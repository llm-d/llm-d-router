// Copyright 2025 The llm-d Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package kvevents

import (
	"context"
	"testing"
	"time"
)

func TestReplayLimiterBoundsConcurrentReplays(t *testing.T) {
	limiter := newReplayLimiter(1)
	if !limiter.acquire(context.Background()) {
		t.Fatal("first replay slot was not acquired")
	}

	acquired := make(chan bool, 1)
	go func() {
		acquired <- limiter.acquire(context.Background())
	}()

	select {
	case <-acquired:
		t.Fatal("second replay acquired the occupied slot")
	case <-time.After(50 * time.Millisecond):
	}

	limiter.release()
	select {
	case ok := <-acquired:
		if !ok {
			t.Fatal("second replay failed to acquire the released slot")
		}
	case <-time.After(time.Second):
		t.Fatal("second replay did not acquire the released slot")
	}
	limiter.release()
}

func TestReplayLimiterHonorsCancellation(t *testing.T) {
	limiter := newReplayLimiter(1)
	if !limiter.acquire(context.Background()) {
		t.Fatal("first replay slot was not acquired")
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if limiter.acquire(ctx) {
		t.Fatal("canceled replay acquired a slot")
	}
	limiter.release()
}
