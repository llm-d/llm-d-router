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

package redis

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestByteBudgetReserveIsAtomic(t *testing.T) {
	now := time.Unix(0, 0)
	b := &byteBudget{rate: 100, tokens: 100, last: now, now: func() time.Time { return now }}
	var total atomic.Int64
	var wg sync.WaitGroup
	for range 20 {
		wg.Go(func() {
			_, reserved, ok := b.reserve(50, 1)
			if ok {
				total.Add(int64(reserved))
			}
		})
	}
	wg.Wait()
	// Two 50-byte reservations exhaust the bucket; a third would need tokens > 0
	// to overdraft. Concurrent callers must not each observe the full balance.
	require.Equal(t, int64(100), total.Load())
}
