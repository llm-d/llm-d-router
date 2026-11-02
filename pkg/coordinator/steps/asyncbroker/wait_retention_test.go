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
	"sync"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/go-chi/chi/v5"
	"github.com/llm-d/llm-d-async/api"
	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/require"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

type waitResultReadHook func(context.Context) error

func (waitResultReadHook) DialHook(next redis.DialHook) redis.DialHook { return next }

func (h waitResultReadHook) ProcessHook(next redis.ProcessHook) redis.ProcessHook {
	return func(ctx context.Context, cmd redis.Cmder) error {
		if cmd.Name() == "lrange" {
			if err := h(ctx); err != nil {
				return err
			}
		}
		return next(ctx, cmd)
	}
}

func (waitResultReadHook) ProcessPipelineHook(next redis.ProcessPipelineHook) redis.ProcessPipelineHook {
	return next
}

func TestAsyncBrokerConcurrentWaitsReceiveResult(t *testing.T) {
	step, rdb := newAsyncTestStep(t, nil)
	t.Cleanup(func() { _ = step.rdb.Close() })
	headers := map[string]string{
		defaultModeHeader: "enqueue", defaultTenantHeader: "team-a",
		reqcommon.RequestIDHeaderKey: "req-test-1", defaultTimeoutHeader: "2",
	}
	queued, queuedRec := asyncReqCtx(t, `{"model":"test-model"}`, headers)
	require.ErrorIs(t, step.Execute(t.Context(), queued), pipeline.ErrPipelineDone)
	require.Equal(t, http.StatusAccepted, queuedRec.Code)

	// Separate Redis clients model waiters on different coordinator replicas.
	retryStep := *step
	retryStep.rdb = redis.NewClient(step.rdb.Options())
	t.Cleanup(func() { _ = retryStep.rdb.Close() })
	ctx, cancel := context.WithCancel(t.Context())
	var workers sync.WaitGroup
	t.Cleanup(func() { cancel(); workers.Wait() })
	headers[defaultModeHeader] = "wait"
	var parked, resume [2]chan struct{}
	var done [2]chan error
	var responses [2]*httptest.ResponseRecorder
	for i, s := range []*Step{step, &retryStep} {
		parked[i], resume[i], done[i] = make(chan struct{}), make(chan struct{}), make(chan error, 1)
		reads := 0
		s.rdb.AddHook(waitResultReadHook(func(ctx context.Context) error {
			reads++
			// The retry lookup sees the live token; the next mailbox read is
			// inside waitForResult, after the request has attached.
			if reads == 2 {
				close(parked[i])
				select {
				case <-resume[i]:
				case <-ctx.Done():
					return ctx.Err()
				}
			}
			return nil
		}))
		req, rec := asyncReqCtx(t, `{"model":"test-model"}`, headers)
		responses[i] = rec
		workers.Go(func() { done[i] <- s.Execute(ctx, req) })
	}
	for _, ready := range parked {
		select {
		case <-ready:
		case <-time.After(5 * time.Second):
			t.Fatal("wait request did not attach")
		}
	}

	key := resultKey("team-a", "req-test-1")
	activeKey := api.RequestActiveTokenKey(envelopeID("team-a", "req-test-1"))
	result, err := json.Marshal(api.ResultMessage{StatusCode: http.StatusOK, Payload: `{"answer":"complete"}`})
	require.NoError(t, err)
	_, err = rdb.TxPipelined(t.Context(), func(pipe redis.Pipeliner) error {
		pipe.LPush(t.Context(), key, result)
		pipe.Expire(t.Context(), key, 5*time.Minute)
		pipe.Del(t.Context(), activeKey)
		return nil
	})
	require.NoError(t, err)

	// The second attached waiter cannot read until the first has delivered.
	for i := range resume {
		close(resume[i])
		select {
		case err = <-done[i]:
		case <-time.After(5 * time.Second):
			t.Fatal("wait request did not finish")
		}
		require.Equal(t, http.StatusOK, responses[i].Code, "waiter %d: %s", i, responses[i].Body.String())
		require.ErrorIs(t, err, pipeline.ErrPipelineDone)
		require.JSONEq(t, `{"answer":"complete"}`, responses[i].Body.String())
	}
	n, err := rdb.ZCard(t.Context(), "team-default-queue").Result()
	require.NoError(t, err)
	require.Equal(t, int64(1), n, "reattached waits must not submit more work")
}

func TestAsyncBrokerWaitDeadlineResultIsRetained(t *testing.T) {
	tests := []struct {
		name     string
		res      api.ResultMessage
		wantDone bool
		wantTTL  time.Duration
	}{
		{
			name:     "success keeps the grace",
			res:      api.ResultMessage{StatusCode: http.StatusOK, Payload: `{"answer":"boundary"}`},
			wantDone: true,
			wantTTL:  resultFetchGraceTTL,
		},
		{
			name:    "error keeps the floor",
			res:     api.ResultMessage{StatusCode: http.StatusServiceUnavailable, Payload: `{"error":"overloaded"}`},
			wantTTL: 5 * time.Second,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			step, rdb := newAsyncTestStep(t, nil)
			t.Cleanup(func() { _ = step.rdb.Close() })
			req, rec := asyncReqCtx(t, `{"model":"test-model"}`, nil)
			key := resultKey("team-a", req.RequestID)
			result, err := json.Marshal(tc.res)
			require.NoError(t, err)
			firstRead := true
			step.rdb.AddHook(waitResultReadHook(func(ctx context.Context) error {
				if firstRead {
					firstRead = false
					// Hold the initial Redis read until the wait deadline, then publish
					// before the final lookup runs with its independent context. The
					// stored TTL lies between the wait floor and the grace.
					<-ctx.Done()
					_, err = rdb.TxPipelined(t.Context(), func(pipe redis.Pipeliner) error {
						pipe.LPush(t.Context(), key, result)
						pipe.Expire(t.Context(), key, 30*time.Second)
						return nil
					})
					return ctx.Err()
				}
				return nil
			}))

			got := step.waitForResult(t.Context(), req, "team-a", 10*time.Millisecond)
			require.NoError(t, err)
			require.Equal(t, tc.wantDone, errors.Is(got, pipeline.ErrPipelineDone), "pipeline result: %v", got)
			require.Equal(t, tc.res.StatusCode, rec.Code)
			require.JSONEq(t, tc.res.Payload, rec.Body.String())
			ttl, err := rdb.TTL(t.Context(), key).Result()
			require.NoError(t, err)
			require.Equal(t, tc.wantTTL, ttl, "final-deadline delivery must bound the result")
		})
	}
}

func TestAsyncBrokerWaitResultLifecycle(t *testing.T) {
	for _, reclaim := range []string{"expiry", "delete"} {
		t.Run(reclaim, func(t *testing.T) {
			mr := miniredis.RunT(t)
			built, err := New(nil, map[string]any{
				"redis_url": "redis://" + mr.Addr(), "wakeup_mode": "poll",
				"default_queue": "team-default-queue",
			})
			require.NoError(t, err)
			step := built.(*Step)
			rdb := step.rdb
			t.Cleanup(func() { _ = rdb.Close() })
			key := resultKey("team-a", "req-test-1")
			result, err := json.Marshal(api.ResultMessage{StatusCode: http.StatusOK, Payload: `{"answer":"retained"}`})
			require.NoError(t, err)
			// No expiry on the stored result: the AP's result_ttl_seconds
			// defaults to 0.
			require.NoError(t, rdb.LPush(t.Context(), key, result).Err())
			headers := map[string]string{
				defaultModeHeader: "wait", defaultTenantHeader: "team-a",
				reqcommon.RequestIDHeaderKey: "req-test-1",
			}
			for i := range 2 {
				if i > 0 {
					mr.FastForward(resultFetchGraceTTL / 2)
				}
				req, rec := asyncReqCtx(t, `{"model":"test-model"}`, headers)
				require.ErrorIs(t, step.Execute(t.Context(), req), pipeline.ErrPipelineDone)
				require.Equal(t, http.StatusOK, rec.Code)
				require.JSONEq(t, `{"answer":"retained"}`, rec.Body.String())
				ttl, err := rdb.TTL(t.Context(), key).Result()
				require.NoError(t, err)
				require.Equal(t, resultFetchGraceTTL, ttl, "wait delivery must bound the result to the grace window")
			}
			n, err := rdb.ZCard(t.Context(), "team-default-queue").Result()
			require.NoError(t, err)
			require.Zero(t, n, "retrying within the grace window must not submit work")

			if reclaim == "expiry" {
				mr.FastForward(resultFetchGraceTTL)
			} else {
				router := chi.NewRouter()
				step.RegisterRoutes(router)
				req := httptest.NewRequest(http.MethodDelete, "/v1/requests/req-test-1", nil)
				req.Header.Set(defaultTenantHeader, "team-a")
				rec := httptest.NewRecorder()
				router.ServeHTTP(rec, req)
				require.Equal(t, http.StatusNoContent, rec.Code)
			}
			headers[defaultModeHeader] = "enqueue"
			req, rec := asyncReqCtx(t, `{"model":"test-model"}`, headers)
			require.ErrorIs(t, step.Execute(t.Context(), req), pipeline.ErrPipelineDone)
			require.Equal(t, http.StatusAccepted, rec.Code)
			n, err = rdb.ZCard(t.Context(), "team-default-queue").Result()
			require.NoError(t, err)
			require.Equal(t, int64(1), n, "a reclaimed result must allow a fresh submission with the same id")
		})
	}
}

func TestAsyncBrokerWaitErrorExpiresAfterFloor(t *testing.T) {
	mr := miniredis.RunT(t)
	built, err := New(nil, map[string]any{
		"redis_url": "redis://" + mr.Addr(), "wakeup_mode": "poll",
		"default_queue": "team-default-queue",
	})
	require.NoError(t, err)
	step := built.(*Step)
	rdb := step.rdb
	t.Cleanup(func() { _ = rdb.Close() })
	result, err := json.Marshal(api.ResultMessage{StatusCode: http.StatusServiceUnavailable, Payload: `{"error":"overloaded"}`})
	require.NoError(t, err)
	require.NoError(t, rdb.LPush(t.Context(), resultKey("team-a", "req-test-1"), result).Err())
	headers := map[string]string{
		defaultModeHeader: "wait", defaultTenantHeader: "team-a",
		reqcommon.RequestIDHeaderKey: "req-test-1", defaultTimeoutHeader: "1",
	}
	// A client retrying every 2s reads the stored error at 0s, 2s and 4s. The
	// floor runs from the first delivery, so the retry at 6s is enqueued.
	for range 3 {
		req, rec := asyncReqCtx(t, `{"model":"test-model"}`, headers)
		require.NotErrorIs(t, step.Execute(t.Context(), req), pipeline.ErrPipelineDone)
		require.Equal(t, http.StatusServiceUnavailable, rec.Code)
		mr.FastForward(2 * time.Second)
	}
	n, err := rdb.ZCard(t.Context(), "team-default-queue").Result()
	require.NoError(t, err)
	require.Zero(t, n, "reading the stored error must not submit work")

	headers[defaultModeHeader] = "enqueue"
	req, rec := asyncReqCtx(t, `{"model":"test-model"}`, headers)
	require.ErrorIs(t, step.Execute(t.Context(), req), pipeline.ErrPipelineDone)
	require.Equal(t, http.StatusAccepted, rec.Code)
	n, err = rdb.ZCard(t.Context(), "team-default-queue").Result()
	require.NoError(t, err)
	require.Equal(t, int64(1), n, "a retry past the floor must be enqueued again")
}

func TestAsyncBrokerWaitZeroFetchGraceKeepsResultForAttachedWaiters(t *testing.T) {
	step, rdb := newAsyncTestStep(t, map[string]any{"fetch_grace_seconds": 0})
	t.Cleanup(func() { _ = step.rdb.Close() })
	req, rec := asyncReqCtx(t, `{"model":"test-model"}`, nil)
	key := resultKey("team-a", req.RequestID)
	result, err := json.Marshal(api.ResultMessage{StatusCode: http.StatusOK, Payload: `{"answer":"floor"}`})
	require.NoError(t, err)
	require.NoError(t, rdb.LPush(t.Context(), key, result).Err())

	require.ErrorIs(t, step.waitForResult(t.Context(), req, "team-a", time.Second), pipeline.ErrPipelineDone)
	require.Equal(t, http.StatusOK, rec.Code)
	ttl, err := rdb.TTL(t.Context(), key).Result()
	require.NoError(t, err)
	require.Equal(t, 5*time.Second, ttl, "a zero fetch grace must not delete the result under other attached waiters")
}
