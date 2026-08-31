/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package asyncbroker

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/go-chi/chi/v5"
	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-async/api"
	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
	"github.com/llm-d/llm-d-router/pkg/coordinator/steps"
)

// newAsyncTestStep builds the step through its factory against a miniredis,
// with poll wake-up for determinism. extra merges over the base params.
func newAsyncTestStep(t *testing.T, extra map[string]any) (*Step, *redis.Client) {
	t.Helper()
	mr := miniredis.RunT(t)
	params := map[string]any{
		"redis_url":   "redis://" + mr.Addr(),
		"wakeup_mode": "poll",
		"routes": []any{
			map[string]any{"model": "test-model", "queue": "team-default-queue", "tier": "interactive"},
		},
		"objectives": map[string]any{
			"interactive": map[string]any{"reserved": "interactive-reserved", "overflow": "interactive-overflow"},
		},
		"quota": map[string]any{"limits": map[string]any{"limited-team": 1}},
	}
	for k, v := range extra {
		params[k] = v
	}
	step, err := New(nil, params)
	require.NoError(t, err)
	rdb := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	t.Cleanup(func() { _ = rdb.Close() })
	return step.(*Step), rdb
}

// asyncReqCtx builds a RequestContext the way the coordinator server does.
func asyncReqCtx(t *testing.T, body string, headers map[string]string) (*pipeline.RequestContext, *httptest.ResponseRecorder) {
	t.Helper()
	var parsed map[string]any
	require.NoError(t, json.Unmarshal([]byte(body), &parsed))
	model, _ := parsed["model"].(string)
	stream, _ := parsed["stream"].(bool)
	h := http.Header{}
	for k, v := range headers {
		h.Set(k, v)
	}
	rec := httptest.NewRecorder()
	return &pipeline.RequestContext{
		RequestID:        "req-test-1",
		OriginalPath:     "/v1/chat/completions",
		OriginalHeaders:  h,
		OriginalBody:     []byte(body),
		Body:             parsed,
		Model:            model,
		Stream:           stream,
		KVTransferParams: map[string]any{},
		ResponseWriter:   rec,
	}, rec
}

func TestAsyncBrokerConfigValidation(t *testing.T) {
	mr := miniredis.RunT(t)
	base := func() map[string]any {
		return map[string]any{"redis_url": "redis://" + mr.Addr()}
	}
	tests := []struct {
		name    string
		mutate  func(map[string]any)
		wantErr string
	}{
		{
			name:    "missing redis_url",
			mutate:  func(p map[string]any) { delete(p, "redis_url") },
			wantErr: "redis_url is required",
		},
		{
			name:    "bad wakeup_mode",
			mutate:  func(p map[string]any) { p["wakeup_mode"] = "sometimes" },
			wantErr: "wakeup_mode",
		},
		{
			name:    "unknown param key",
			mutate:  func(p map[string]any) { p["bogus_key"] = true },
			wantErr: "bogus_key",
		},
		{
			name:    "forward_headers must not include the mode header",
			mutate:  func(p map[string]any) { p["forward_headers"] = []any{"x-ap-mode"} },
			wantErr: "forward_headers",
		},
		{
			name:    "forward_headers must not include the fairness header",
			mutate:  func(p map[string]any) { p["forward_headers"] = []any{"x-llm-d-inference-fairness-id"} },
			wantErr: "forward_headers",
		},
		{
			name: "timeouts keyed by unknown mode",
			mutate: func(p map[string]any) {
				p["timeouts"] = map[string]any{"passthrough": map[string]any{"max_seconds": 5}}
			},
			wantErr: "timeouts",
		},
		{
			name:    "negative wait_cap_seconds",
			mutate:  func(p map[string]any) { p["wait_cap_seconds"] = -1 },
			wantErr: "wait_cap_seconds",
		},
		{
			name: "negative timeout seconds",
			mutate: func(p map[string]any) {
				p["timeouts"] = map[string]any{"wait": map[string]any{"default_seconds": -5}}
			},
			wantErr: "timeouts.wait",
		},
		{
			name: "negative quota window",
			mutate: func(p map[string]any) {
				p["quota"] = map[string]any{"window_seconds": -60}
			},
			wantErr: "quota.window_seconds",
		},
		{
			name: "negative quota limit",
			mutate: func(p map[string]any) {
				p["quota"] = map[string]any{"limits": map[string]any{"team-a": -1}}
			},
			wantErr: "quota.limits.team-a",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			params := base()
			tc.mutate(params)
			_, err := New(nil, params)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.wantErr)
		})
	}

	// The injected connector and format params must not trip unknown-key
	// rejection.
	params := base()
	params[steps.ParamKVConnector] = "kv-shared-storage"
	params[steps.ParamECConnector] = "ec-shared-storage"
	params["use_openai_format"] = true
	_, err := New(nil, params)
	require.NoError(t, err)
}

func TestAsyncBrokerNoModeHeaderIsNoOp(t *testing.T) {
	step, rdb := newAsyncTestStep(t, nil)
	reqCtx, rec := asyncReqCtx(t, `{"model":"test-model"}`, map[string]string{
		"X-Team":                        "team-a",
		"x-llm-d-inference-fairness-id": "spoofed",
	})

	require.NoError(t, step.Execute(t.Context(), reqCtx))

	// Untouched: no response written, headers preserved verbatim (including
	// a client-supplied fairness header, which is gateway-baseline behavior
	// for requests that did not opt in), nothing in Redis.
	assert.Empty(t, rec.Body.String())
	assert.Equal(t, "spoofed", reqCtx.OriginalHeaders.Get("x-llm-d-inference-fairness-id"))
	keys, err := rdb.Keys(t.Context(), "*").Result()
	require.NoError(t, err)
	assert.Empty(t, keys)
}

func TestAsyncBrokerRejections(t *testing.T) {
	tests := []struct {
		name    string
		body    string
		headers map[string]string
		wantMsg string
	}{
		{
			name:    "unknown mode",
			body:    `{"model":"test-model"}`,
			headers: map[string]string{defaultModeHeader: "sideways"},
			wantMsg: "unknown " + defaultModeHeader + " value",
		},
		{
			name:    "stream in queued mode",
			body:    `{"model":"test-model","stream":true}`,
			headers: map[string]string{defaultModeHeader: "enqueue"},
			wantMsg: "stream is only supported in passthrough mode",
		},
		{
			name:    "missing model",
			body:    `{"messages":[]}`,
			headers: map[string]string{defaultModeHeader: "enqueue"},
			wantMsg: "model is required",
		},
		{
			name:    "tenant with colon",
			body:    `{"model":"test-model"}`,
			headers: map[string]string{defaultModeHeader: "enqueue", "X-Team": "a:b"},
			wantMsg: "must not contain",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			step, _ := newAsyncTestStep(t, nil)
			reqCtx, rec := asyncReqCtx(t, tc.body, tc.headers)
			err := step.Execute(t.Context(), reqCtx)
			require.True(t, errors.Is(err, pipeline.ErrPipelineDone), "want ErrPipelineDone, got %v", err)
			assert.Equal(t, http.StatusBadRequest, rec.Code)
			assert.Contains(t, rec.Body.String(), tc.wantMsg)
		})
	}
}

func TestAsyncBrokerEnqueue(t *testing.T) {
	step, rdb := newAsyncTestStep(t, nil)
	reqCtx, rec := asyncReqCtx(t, `{"model":"test-model","messages":[{"role":"user","content":"hi"}]}`,
		map[string]string{
			defaultModeHeader:           "enqueue",
			"X-Team":                    "team-a",
			"X-Request-Timeout-Seconds": "120",
			"x-llm-d-slo-ttft-ms":       "800",
			"traceparent":               "00-abc-def-01",
		})

	err := step.Execute(t.Context(), reqCtx)
	require.True(t, errors.Is(err, pipeline.ErrPipelineDone))
	require.Equal(t, http.StatusAccepted, rec.Code, rec.Body.String())

	var resp map[string]string
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	assert.Equal(t, "req-test-1", resp["id"])
	assert.Equal(t, "pending", resp["status"])
	assert.Equal(t, "req-test-1", rec.Header().Get("x-request-id"))

	// The envelope landed on the routed queue with per-request result key,
	// endpoint, tenant metadata, and the forwarded SLO header but never the
	// mode header.
	members, err := rdb.ZRange(t.Context(), "team-default-queue", 0, -1).Result()
	require.NoError(t, err)
	require.Len(t, members, 1)
	var envelope struct {
		Internal struct {
			ResultQueueName string `json:"result_queue_name"`
		} `json:"internal"`
		Data api.RequestMessage `json:"data"`
	}
	require.NoError(t, json.Unmarshal([]byte(members[0]), &envelope))
	assert.Equal(t, resultKey("team-a", "req-test-1"), envelope.Internal.ResultQueueName)
	assert.Equal(t, envelopeID("team-a", "req-test-1"), envelope.Data.ID)
	assert.Equal(t, "/v1/chat/completions", envelope.Data.Endpoint)
	assert.Equal(t, "team-a", envelope.Data.Metadata["userid"])
	assert.Equal(t, "00-abc-def-01", envelope.Data.Metadata["traceparent"])
	assert.Equal(t, "test-model", envelope.Data.Payload["model"])
	assert.Equal(t, "800", envelope.Data.Headers["x-llm-d-slo-ttft-ms"])
	for k := range envelope.Data.Headers {
		assert.NotEqual(t, "x-ap-mode", strings.ToLower(k))
	}
	assert.InDelta(t, time.Now().Add(120*time.Second).Unix(), envelope.Data.Deadline, 5)

	// Pending marker: the producer's active-token key exists, tenant scoped.
	exists, err := rdb.Exists(t.Context(), api.RequestActiveTokenKey(envelopeID("team-a", "req-test-1"))).Result()
	require.NoError(t, err)
	assert.Equal(t, int64(1), exists)
}

func TestAsyncBrokerWaitDeliversResult(t *testing.T) {
	step, rdb := newAsyncTestStep(t, nil)
	reqCtx, rec := asyncReqCtx(t, `{"model":"test-model"}`,
		map[string]string{defaultModeHeader: "wait", "X-Team": "team-a"})

	// Pre-load the result so the wait loop's first check finds it.
	res, err := json.Marshal(api.ResultMessage{StatusCode: 200, Payload: `{"object":"chat.completion"}`})
	require.NoError(t, err)
	key := resultKey("team-a", "req-test-1")
	require.NoError(t, rdb.LPush(t.Context(), key, string(res)).Err())

	err = step.Execute(t.Context(), reqCtx)
	require.True(t, errors.Is(err, pipeline.ErrPipelineDone))
	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Body.String(), "chat.completion")

	// Delivered on the held connection: the mailbox is reclaimed.
	exists, err := rdb.Exists(t.Context(), key).Result()
	require.NoError(t, err)
	assert.Equal(t, int64(0), exists)
}

func TestAsyncBrokerWaitCapFallsBackToPending(t *testing.T) {
	step, _ := newAsyncTestStep(t, map[string]any{"wait_cap_seconds": 1})
	reqCtx, rec := asyncReqCtx(t, `{"model":"test-model"}`,
		map[string]string{defaultModeHeader: "wait", "X-Team": "team-a"})

	start := time.Now()
	err := step.Execute(t.Context(), reqCtx)
	require.True(t, errors.Is(err, pipeline.ErrPipelineDone))
	assert.GreaterOrEqual(t, time.Since(start), time.Second)
	assert.Equal(t, http.StatusAccepted, rec.Code)
	assert.Contains(t, rec.Body.String(), "pending")
}

func TestAsyncBrokerWaitDeadlineAnswersTimeout(t *testing.T) {
	step, _ := newAsyncTestStep(t, map[string]any{
		"timeouts": map[string]any{"wait": map[string]any{"default_seconds": 1}},
	})
	reqCtx, rec := asyncReqCtx(t, `{"model":"test-model"}`,
		map[string]string{defaultModeHeader: "wait", "X-Team": "team-a"})

	start := time.Now()
	err := step.Execute(t.Context(), reqCtx)
	require.True(t, errors.Is(err, pipeline.ErrPipelineDone))
	assert.GreaterOrEqual(t, time.Since(start), time.Second)
	assert.Equal(t, http.StatusGatewayTimeout, rec.Code)
	assert.Contains(t, rec.Body.String(), api.ErrCodeDeadlineExceeded)
}

func TestAsyncBrokerDeadlineClamping(t *testing.T) {
	readDeadline := func(t *testing.T, rdb *redis.Client) int64 {
		t.Helper()
		members, err := rdb.ZRange(t.Context(), "team-default-queue", 0, -1).Result()
		require.NoError(t, err)
		require.Len(t, members, 1)
		var envelope struct {
			Data api.RequestMessage `json:"data"`
		}
		require.NoError(t, json.Unmarshal([]byte(members[0]), &envelope))
		return envelope.Data.Deadline
	}
	headers := map[string]string{
		defaultModeHeader:           "enqueue",
		"X-Team":                    "team-a",
		"X-Request-Timeout-Seconds": "1000000",
	}

	t.Run("unclamped when no max configured", func(t *testing.T) {
		step, rdb := newAsyncTestStep(t, nil)
		reqCtx, rec := asyncReqCtx(t, `{"model":"test-model"}`, headers)
		require.True(t, errors.Is(step.Execute(t.Context(), reqCtx), pipeline.ErrPipelineDone))
		require.Equal(t, http.StatusAccepted, rec.Code, rec.Body.String())
		assert.InDelta(t, time.Now().Add(1000000*time.Second).Unix(), readDeadline(t, rdb), 5)
	})

	t.Run("clamped when max configured", func(t *testing.T) {
		step, rdb := newAsyncTestStep(t, map[string]any{
			"timeouts": map[string]any{"enqueue": map[string]any{"max_seconds": 300}},
		})
		reqCtx, rec := asyncReqCtx(t, `{"model":"test-model"}`, headers)
		require.True(t, errors.Is(step.Execute(t.Context(), reqCtx), pipeline.ErrPipelineDone))
		require.Equal(t, http.StatusAccepted, rec.Code, rec.Body.String())
		assert.InDelta(t, time.Now().Add(300*time.Second).Unix(), readDeadline(t, rdb), 5)
	})

	t.Run("partial entry keeps the per-mode default", func(t *testing.T) {
		// An entry setting only max_seconds must not disturb the enqueue
		// default of one hour: each bound falls back independently.
		step, rdb := newAsyncTestStep(t, map[string]any{
			"timeouts": map[string]any{"enqueue": map[string]any{"max_seconds": 7200}},
		})
		reqCtx, rec := asyncReqCtx(t, `{"model":"test-model"}`, map[string]string{
			defaultModeHeader: "enqueue",
			"X-Team":          "team-a",
		})
		require.True(t, errors.Is(step.Execute(t.Context(), reqCtx), pipeline.ErrPipelineDone))
		require.Equal(t, http.StatusAccepted, rec.Code, rec.Body.String())
		assert.InDelta(t, time.Now().Add(3600*time.Second).Unix(), readDeadline(t, rdb), 5)
	})

	t.Run("wait max defaults to one hour", func(t *testing.T) {
		cfg := &asyncBrokerConfig{}
		assert.Equal(t, int64(defaultWaitMaxSecs), cfg.timeoutBounds(asyncModeWait).MaxSeconds)
		// A default configured above the fallback max raises the max with it.
		cfg.Timeouts = map[string]asyncTimeoutBounds{"wait": {DefaultSeconds: 7200}}
		assert.Equal(t, int64(7200), cfg.timeoutBounds(asyncModeWait).MaxSeconds)
		// Enqueue stays unclamped.
		assert.Equal(t, int64(0), cfg.timeoutBounds(asyncModeEnqueue).MaxSeconds)
	})
}

func TestAsyncBrokerPassthroughStampsAndClassifies(t *testing.T) {
	step, _ := newAsyncTestStep(t, nil)

	// First request for the limited tenant: reserved, client-supplied
	// identity headers replaced, pipeline continues (nil error). The request
	// context stays open so the quota slot is held.
	ctx1, cancel1 := context.WithCancel(t.Context())
	reqCtx1, rec1 := asyncReqCtx(t, `{"model":"test-model"}`, map[string]string{
		defaultModeHeader:               "passthrough",
		"X-Team":                        "limited-team",
		"x-llm-d-inference-objective":   "self-assigned",
		"x-llm-d-inference-fairness-id": "spoofed",
	})
	require.NoError(t, step.Execute(ctx1, reqCtx1))
	assert.Empty(t, rec1.Body.String())
	assert.Equal(t, "interactive-reserved", reqCtx1.OriginalHeaders.Get("x-llm-d-inference-objective"))
	assert.Equal(t, "limited-team", reqCtx1.OriginalHeaders.Get("x-llm-d-inference-fairness-id"))

	// Second concurrent request exceeds the reserved limit of 1: overflow.
	reqCtx2, _ := asyncReqCtx(t, `{"model":"test-model"}`, map[string]string{
		defaultModeHeader: "passthrough",
		"X-Team":          "limited-team",
	})
	require.NoError(t, step.Execute(t.Context(), reqCtx2))
	assert.Equal(t, "interactive-overflow", reqCtx2.OriginalHeaders.Get("x-llm-d-inference-objective"))

	// Releasing the first request frees the slot for a later one.
	cancel1()
	require.Eventually(t, func() bool {
		reqCtx3, _ := asyncReqCtx(t, `{"model":"test-model"}`, map[string]string{
			defaultModeHeader: "passthrough",
			"X-Team":          "limited-team",
		})
		ctx3, cancel3 := context.WithCancel(t.Context())
		defer cancel3()
		if err := step.Execute(ctx3, reqCtx3); err != nil {
			return false
		}
		return reqCtx3.OriginalHeaders.Get("x-llm-d-inference-objective") == "interactive-reserved"
	}, 2*time.Second, 50*time.Millisecond)

	// An unlimited tenant is always reserved and never touches counters.
	reqCtx4, _ := asyncReqCtx(t, `{"model":"test-model"}`, map[string]string{
		defaultModeHeader: "passthrough",
		"X-Team":          "team-free",
	})
	require.NoError(t, step.Execute(t.Context(), reqCtx4))
	assert.Equal(t, "interactive-reserved", reqCtx4.OriginalHeaders.Get("x-llm-d-inference-objective"))
	assert.Equal(t, "team-free", reqCtx4.OriginalHeaders.Get("x-llm-d-inference-fairness-id"))
}

func TestAsyncBrokerRoutes(t *testing.T) {
	step, rdb := newAsyncTestStep(t, nil)
	r := chi.NewRouter()
	step.RegisterRoutes(r)

	do := func(method, path, tenant string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(method, path, nil)
		if tenant != "" {
			req.Header.Set("X-Team", tenant)
		}
		rec := httptest.NewRecorder()
		r.ServeHTTP(rec, req)
		return rec
	}

	// Models derive from the configured routes.
	rec := do(http.MethodGet, "/v1/models", "")
	require.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Body.String(), "test-model")

	// Unknown id: gone.
	rec = do(http.MethodGet, "/v1/requests/nope", "team-a")
	assert.Equal(t, http.StatusGone, rec.Code)

	// Pending: active token exists, no result yet.
	require.NoError(t, rdb.Set(t.Context(), api.RequestActiveTokenKey(envelopeID("team-a", "pending-id")), "1", 0).Err())
	rec = do(http.MethodGet, "/v1/requests/pending-id", "team-a")
	assert.Equal(t, http.StatusAccepted, rec.Code)

	// The pending check is tenant scoped: the wrong tenant cannot learn
	// that the id exists.
	rec = do(http.MethodGet, "/v1/requests/pending-id", "team-b")
	assert.Equal(t, http.StatusGone, rec.Code)

	// Ready: delivered verbatim, and the mailbox TTL shrinks to the grace
	// window instead of the original TTL.
	res, err := json.Marshal(api.ResultMessage{StatusCode: 200, Payload: `{"done":true}`})
	require.NoError(t, err)
	key := resultKey("team-a", "ready-id")
	require.NoError(t, rdb.LPush(t.Context(), key, string(res)).Err())
	require.NoError(t, rdb.Expire(t.Context(), key, time.Hour).Err())
	rec = do(http.MethodGet, "/v1/requests/ready-id", "team-a")
	require.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Body.String(), "done")
	ttl, err := rdb.TTL(t.Context(), key).Result()
	require.NoError(t, err)
	assert.LessOrEqual(t, ttl, resultFetchGraceTTL)

	// Wrong tenant finds nothing.
	rec = do(http.MethodGet, "/v1/requests/ready-id", "team-b")
	assert.Equal(t, http.StatusGone, rec.Code)

	// Delete is tenant scoped too: the wrong tenant cannot reclaim the result.
	rec = do(http.MethodDelete, "/v1/requests/ready-id", "team-b")
	assert.Equal(t, http.StatusNoContent, rec.Code)
	exists, err := rdb.Exists(t.Context(), key).Result()
	require.NoError(t, err)
	assert.Equal(t, int64(1), exists)

	// The read routes apply the enqueue path's tenant and id validation.
	rec = do(http.MethodGet, "/v1/requests/bad*id", "team-a")
	assert.Equal(t, http.StatusBadRequest, rec.Code)
	rec = do(http.MethodGet, "/v1/requests/ready-id", "team:a")
	assert.Equal(t, http.StatusBadRequest, rec.Code)
	rec = do(http.MethodDelete, "/v1/requests/ready-id", "team:a")
	assert.Equal(t, http.StatusBadRequest, rec.Code)

	// A stored result with a status outside the valid HTTP range answers 502
	// instead of panicking in WriteHeader.
	badRes, err := json.Marshal(api.ResultMessage{StatusCode: 42, Payload: `{}`})
	require.NoError(t, err)
	require.NoError(t, rdb.LPush(t.Context(), resultKey("team-a", "corrupt-id"), string(badRes)).Err())
	rec = do(http.MethodGet, "/v1/requests/corrupt-id", "team-a")
	assert.Equal(t, http.StatusBadGateway, rec.Code)
	assert.Contains(t, rec.Body.String(), "MALFORMED_RESULT")

	// Delete reclaims immediately.
	rec = do(http.MethodDelete, "/v1/requests/ready-id", "team-a")
	assert.Equal(t, http.StatusNoContent, rec.Code)
	exists, err = rdb.Exists(t.Context(), key).Result()
	require.NoError(t, err)
	assert.Equal(t, int64(0), exists)

	// Deleting a still queued request stamps its cancellation marker (the
	// cancel script copies the active token, so only in-flight ids cancel).
	rec = do(http.MethodDelete, "/v1/requests/pending-id", "team-a")
	assert.Equal(t, http.StatusNoContent, rec.Code)
	cancelled, err := rdb.Exists(t.Context(), api.RequestCancellationKey(envelopeID("team-a", "pending-id"))).Result()
	require.NoError(t, err)
	assert.Equal(t, int64(1), cancelled)

	// Result error codes map onto client statuses.
	res, err = json.Marshal(api.ResultMessage{ErrorCode: api.ErrCodeDeadlineExceeded, ErrorMessage: "too slow"})
	require.NoError(t, err)
	require.NoError(t, rdb.LPush(t.Context(), resultKey("team-a", "late-id"), string(res)).Err())
	rec = do(http.MethodGet, "/v1/requests/late-id", "team-a")
	assert.Equal(t, http.StatusGatewayTimeout, rec.Code)
	assert.Contains(t, rec.Body.String(), api.ErrCodeDeadlineExceeded)
}

func TestAsyncBrokerFetchGraceConfig(t *testing.T) {
	seed := func(t *testing.T, rdb *redis.Client, id string) string {
		t.Helper()
		res, err := json.Marshal(api.ResultMessage{StatusCode: 200, Payload: `{"done":true}`})
		require.NoError(t, err)
		key := resultKey("team-a", id)
		require.NoError(t, rdb.LPush(t.Context(), key, string(res)).Err())
		require.NoError(t, rdb.Expire(t.Context(), key, time.Hour).Err())
		return key
	}
	fetch := func(t *testing.T, step *Step, id string) {
		t.Helper()
		r := chi.NewRouter()
		step.RegisterRoutes(r)
		req := httptest.NewRequest(http.MethodGet, "/v1/requests/"+id, nil)
		req.Header.Set("X-Team", "team-a")
		rec := httptest.NewRecorder()
		r.ServeHTTP(rec, req)
		require.Equal(t, http.StatusOK, rec.Code)
	}

	t.Run("custom grace shrinks the mailbox TTL", func(t *testing.T) {
		step, rdb := newAsyncTestStep(t, map[string]any{"fetch_grace_seconds": 5})
		key := seed(t, rdb, "grace-id")
		fetch(t, step, "grace-id")
		ttl, err := rdb.TTL(t.Context(), key).Result()
		require.NoError(t, err)
		assert.Greater(t, ttl, time.Duration(0))
		assert.LessOrEqual(t, ttl, 5*time.Second)
	})

	t.Run("zero grace deletes on delivery", func(t *testing.T) {
		step, rdb := newAsyncTestStep(t, map[string]any{"fetch_grace_seconds": 0})
		key := seed(t, rdb, "eager-id")
		fetch(t, step, "eager-id")
		exists, err := rdb.Exists(t.Context(), key).Result()
		require.NoError(t, err)
		assert.Equal(t, int64(0), exists)
	})
}

// TestAsyncQuotaFailsOpen covers the Redis-error branch: a limited tenant is
// classified reserved (with the error surfaced) when the quota check cannot
// run, so an outage never blocks live traffic.
func TestAsyncQuotaFailsOpen(t *testing.T) {
	mr := miniredis.RunT(t)
	rdb := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	t.Cleanup(func() { _ = rdb.Close() })
	q := &asyncQuotaClassifier{
		rdb: rdb,
		cfg: asyncQuotaConfig{
			Prefix: "quota:", Attribute: "userid",
			Limits: map[string]int{"team-a": 1}, WindowSeconds: 60,
		},
		logger: logr.Discard(),
	}
	mr.Close()

	class, release, err := q.classify(t.Context(), "team-a")
	require.Error(t, err)
	assert.Equal(t, asyncClassificationReserved, class)
	assert.Nil(t, release)
}

// TestAsyncBrokerRetryReattaches covers re-POSTs of a client-chosen id: a
// live copy is reattached to, a cancelled but still queued copy is revived,
// a cancelled tombstone routes to a fresh enqueue, and a stored result is
// delivered to a wait retry. None of them enqueue a duplicate.
func TestAsyncBrokerRetryReattaches(t *testing.T) {
	const queue = "team-default-queue"
	post := func(t *testing.T, step *Step, mode, id string) *httptest.ResponseRecorder {
		t.Helper()
		reqCtx, rec := asyncReqCtx(t, `{"model":"test-model"}`, map[string]string{
			defaultModeHeader: mode, "X-Team": "team-a", "X-Request-Id": id,
		})
		reqCtx.RequestID = id
		err := step.Execute(t.Context(), reqCtx)
		require.True(t, errors.Is(err, pipeline.ErrPipelineDone))
		return rec
	}
	queueLen := func(t *testing.T, rdb *redis.Client) int64 {
		t.Helper()
		n, err := rdb.ZCard(t.Context(), queue).Result()
		require.NoError(t, err)
		return n
	}

	t.Run("live id is not re-enqueued", func(t *testing.T) {
		step, rdb := newAsyncTestStep(t, nil)
		post(t, step, "enqueue", "job-r1")
		rec := post(t, step, "enqueue", "job-r1")
		require.Equal(t, http.StatusAccepted, rec.Code, rec.Body.String())
		assert.Equal(t, int64(1), queueLen(t, rdb))
	})

	t.Run("cancelled queued copy is revived", func(t *testing.T) {
		step, rdb := newAsyncTestStep(t, nil)
		post(t, step, "enqueue", "job-r2")
		eid := envelopeID("team-a", "job-r2")
		require.NoError(t, step.sub.CancelRequests(t.Context(), []string{eid}))
		rec := post(t, step, "enqueue", "job-r2")
		require.Equal(t, http.StatusAccepted, rec.Code, rec.Body.String())
		exists, err := rdb.Exists(t.Context(), api.RequestCancellationKey(eid)).Result()
		require.NoError(t, err)
		assert.Equal(t, int64(0), exists, "retry should clear the cancel marker")
		assert.Equal(t, int64(1), queueLen(t, rdb))
	})

	t.Run("cancelled tombstone runs fresh", func(t *testing.T) {
		step, rdb := newAsyncTestStep(t, nil)
		res, err := json.Marshal(api.ResultMessage{ErrorCode: api.ErrCodeCancelled, ErrorMessage: "cancelled"})
		require.NoError(t, err)
		key := resultKey("team-a", "job-r3")
		require.NoError(t, rdb.LPush(t.Context(), key, string(res)).Err())
		rec := post(t, step, "enqueue", "job-r3")
		require.Equal(t, http.StatusAccepted, rec.Code, rec.Body.String())
		exists, err := rdb.Exists(t.Context(), key).Result()
		require.NoError(t, err)
		assert.Equal(t, int64(0), exists, "retry should clear the tombstone")
		assert.Equal(t, int64(1), queueLen(t, rdb))
	})

	t.Run("wait retry delivers the stored result", func(t *testing.T) {
		step, rdb := newAsyncTestStep(t, nil)
		res, err := json.Marshal(api.ResultMessage{StatusCode: 200, Payload: `{"object":"chat.completion"}`})
		require.NoError(t, err)
		require.NoError(t, rdb.LPush(t.Context(), resultKey("team-a", "job-r4"), string(res)).Err())
		rec := post(t, step, "wait", "job-r4")
		require.Equal(t, http.StatusOK, rec.Code, rec.Body.String())
		assert.Contains(t, rec.Body.String(), "chat.completion")
		assert.Equal(t, int64(0), queueLen(t, rdb), "no enqueue for a completed request")
	})
}
