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
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-async/api"
	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// miniredis does not emit keyspace notifications, so these tests publish the
// notification an LPUSH would fire themselves. That covers the waiter's own
// plumbing (multiplexing, registration, refcounted unsubscribe); Redis's
// event generation is the e2e side of the contract.
func TestAsyncResultWaiterMultiplexesWaiters(t *testing.T) {
	mr := miniredis.RunT(t)
	rdb := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	t.Cleanup(func() { _ = rdb.Close() })
	ctx, cancel := context.WithCancel(t.Context())
	t.Cleanup(cancel)
	w := newAsyncResultWaiter(ctx, rdb, 0, logr.Discard())

	const key = "results:req:team-a:job-1"
	wake1, cleanup1, err := w.register(ctx, key)
	require.NoError(t, err)
	wake2, cleanup2, err := w.register(ctx, key)
	require.NoError(t, err)

	// Publish until the wake fires: the subscription lands on a separate
	// connection, so a single publish could race it.
	waitWake := func(ch <-chan struct{}, name string) {
		t.Helper()
		deadline := time.After(2 * time.Second)
		for {
			require.NoError(t, rdb.Publish(ctx, w.prefix+key, "lpush").Err())
			select {
			case <-ch:
				return
			case <-deadline:
				t.Fatalf("%s did not wake", name)
			case <-time.After(20 * time.Millisecond):
			}
		}
	}

	// One notification wakes every waiter parked on the key.
	waitWake(wake1, "first waiter")
	waitWake(wake2, "second waiter")

	// The first waiter leaving must not tear down the subscription the
	// second still needs.
	cleanup1()
	waitWake(wake2, "second waiter after first cleanup")
	cleanup2()
}

func TestAsyncBrokerWaitNotifyDeliversResult(t *testing.T) {
	step, rdb := newAsyncTestStep(t, map[string]any{"wakeup_mode": "notify"})
	reqCtx, rec := asyncReqCtx(t, `{"model":"test-model"}`, map[string]string{
		"X-AP-Mode": "wait", "X-Team": "team-a",
	})
	start := time.Now()
	done := make(chan error, 1)
	go func() { done <- step.Execute(t.Context(), reqCtx) }()

	res, err := json.Marshal(api.ResultMessage{StatusCode: 200, Payload: `{"done":true}`})
	require.NoError(t, err)
	key := resultKey("team-a", "req-test-1")
	require.NoError(t, rdb.LPush(t.Context(), key, string(res)).Err())

	deadline := time.After(3 * time.Second)
	for {
		select {
		case err := <-done:
			require.True(t, errors.Is(err, pipeline.ErrPipelineDone))
			require.Equal(t, http.StatusOK, rec.Code, rec.Body.String())
			assert.Contains(t, rec.Body.String(), "done")
			// Well under the 2s backup poll proves the wake-up delivered,
			// not the fallback ticker.
			assert.Less(t, time.Since(start), waitBackupPollInterval)
			return
		case <-deadline:
			t.Fatal("wait did not deliver the result")
		case <-time.After(20 * time.Millisecond):
			require.NoError(t, rdb.Publish(t.Context(), "__keyspace@0__:"+key, "lpush").Err())
		}
	}
}
