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

package steps

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
	"github.com/llm-d/llm-d-async/api"
	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// newAsyncTestStep builds the step through its factory against a miniredis,
// with poll wake-up for determinism. extra merges over the base params.
func newAsyncTestStep(t *testing.T, extra map[string]any) (*AsyncBrokerStep, *redis.Client) {
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
	step, err := NewAsyncBrokerStep(nil, params)
	require.NoError(t, err)
	rdb := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	t.Cleanup(func() { _ = rdb.Close() })
	return step.(*AsyncBrokerStep), rdb
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
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			params := base()
			tc.mutate(params)
			_, err := NewAsyncBrokerStep(nil, params)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.wantErr)
		})
	}

	// The injected connector and format params must not trip unknown-key
	// rejection.
	params := base()
	params[ParamKVConnector] = "kv-shared-storage"
	params[ParamECConnector] = "ec-shared-storage"
	params["use_openai_format"] = true
	_, err := NewAsyncBrokerStep(nil, params)
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
			headers: map[string]string{"X-AP-Mode": "sideways"},
			wantMsg: "unknown X-AP-Mode value",
		},
		{
			name:    "stream in queued mode",
			body:    `{"model":"test-model","stream":true}`,
			headers: map[string]string{"X-AP-Mode": "enqueue"},
			wantMsg: "stream is only supported in passthrough mode",
		},
		{
			name:    "missing model",
			body:    `{"messages":[]}`,
			headers: map[string]string{"X-AP-Mode": "enqueue"},
			wantMsg: "model is required",
		},
		{
			name:    "tenant with colon",
			body:    `{"model":"test-model"}`,
			headers: map[string]string{"X-AP-Mode": "enqueue", "X-Team": "a:b"},
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
			"X-AP-Mode":                 "enqueue",
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
	assert.Equal(t, "/v1/chat/completions", envelope.Data.Endpoint)
	assert.Equal(t, "team-a", envelope.Data.Metadata["team"])
	assert.Equal(t, "00-abc-def-01", envelope.Data.Metadata["traceparent"])
	assert.Equal(t, "test-model", envelope.Data.Payload["model"])
	assert.Equal(t, "800", envelope.Data.Headers["x-llm-d-slo-ttft-ms"])
	for k := range envelope.Data.Headers {
		assert.NotEqual(t, "x-ap-mode", strings.ToLower(k))
	}
	assert.InDelta(t, time.Now().Add(120*time.Second).Unix(), envelope.Data.Deadline, 5)

	// Pending marker: the producer's active-token key exists.
	exists, err := rdb.Exists(t.Context(), api.RequestActiveTokenKey("req-test-1")).Result()
	require.NoError(t, err)
	assert.Equal(t, int64(1), exists)
}

func TestAsyncBrokerWaitDeliversResult(t *testing.T) {
	step, rdb := newAsyncTestStep(t, nil)
	reqCtx, rec := asyncReqCtx(t, `{"model":"test-model"}`,
		map[string]string{"X-AP-Mode": "wait", "X-Team": "team-a"})

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
		map[string]string{"X-AP-Mode": "wait", "X-Team": "team-a"})

	start := time.Now()
	err := step.Execute(t.Context(), reqCtx)
	require.True(t, errors.Is(err, pipeline.ErrPipelineDone))
	assert.GreaterOrEqual(t, time.Since(start), time.Second)
	assert.Equal(t, http.StatusAccepted, rec.Code)
	assert.Contains(t, rec.Body.String(), "pending")
}

func TestAsyncBrokerPassthroughStampsAndClassifies(t *testing.T) {
	step, _ := newAsyncTestStep(t, nil)

	// First request for the limited tenant: reserved, client-supplied
	// identity headers replaced, pipeline continues (nil error). The request
	// context stays open so the quota slot is held.
	ctx1, cancel1 := context.WithCancel(t.Context())
	reqCtx1, rec1 := asyncReqCtx(t, `{"model":"test-model"}`, map[string]string{
		"X-AP-Mode":                     "passthrough",
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
		"X-AP-Mode": "passthrough",
		"X-Team":    "limited-team",
	})
	require.NoError(t, step.Execute(t.Context(), reqCtx2))
	assert.Equal(t, "interactive-overflow", reqCtx2.OriginalHeaders.Get("x-llm-d-inference-objective"))

	// Releasing the first request frees the slot for a later one.
	cancel1()
	require.Eventually(t, func() bool {
		reqCtx3, _ := asyncReqCtx(t, `{"model":"test-model"}`, map[string]string{
			"X-AP-Mode": "passthrough",
			"X-Team":    "limited-team",
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
		"X-AP-Mode": "passthrough",
		"X-Team":    "team-free",
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
	require.NoError(t, rdb.Set(t.Context(), api.RequestActiveTokenKey("pending-id"), "1", 0).Err())
	rec = do(http.MethodGet, "/v1/requests/pending-id", "team-a")
	assert.Equal(t, http.StatusAccepted, rec.Code)

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

	// Delete reclaims immediately.
	rec = do(http.MethodDelete, "/v1/requests/ready-id", "team-a")
	assert.Equal(t, http.StatusNoContent, rec.Code)
	exists, err := rdb.Exists(t.Context(), key).Result()
	require.NoError(t, err)
	assert.Equal(t, int64(0), exists)

	// Result error codes map onto client statuses.
	res, err = json.Marshal(api.ResultMessage{ErrorCode: api.ErrCodeDeadlineExceeded, ErrorMessage: "too slow"})
	require.NoError(t, err)
	require.NoError(t, rdb.LPush(t.Context(), resultKey("team-a", "late-id"), string(res)).Err())
	rec = do(http.MethodGet, "/v1/requests/late-id", "team-a")
	assert.Equal(t, http.StatusGatewayTimeout, rec.Code)
	assert.Contains(t, rec.Body.String(), api.ErrCodeDeadlineExceeded)
}
