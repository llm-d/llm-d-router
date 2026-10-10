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

package thunderagent

import (
	"context"
	"fmt"
	"testing"
	"time"

	fwkfc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
)

// BenchmarkPickAdmittedHeads measures one dispatch cycle when every waiting
// head belongs to an admitted session, the common case.
func BenchmarkPickAdmittedHeads(b *testing.B) {
	const pods, sessions, queues = 32, 2000, 64
	a := newTestAgent(testConfig())
	now := time.Now()
	a.mgr.mu.Lock()
	for i := range sessions {
		ep := a.mgr.ensureEndpointLocked(fmt.Sprintf("default/pod-%d", i%pods), 1e9, now)
		s := a.mgr.bindLocked(fmt.Sprintf("s%d", i), ep)
		s.committedTokens, s.turnCount, s.lastResponseAt = 1000, 1, now
	}
	a.mgr.mu.Unlock()

	qs := make([]fwkfc.FlowQueueAccessor, queues)
	for i := range queues {
		qs[i] = makeQueue(fmt.Sprintf("s%d", i), now, 400)
	}
	band := bandOf(qs...)
	ctx := context.Background()

	b.ResetTimer()
	for b.Loop() {
		if _, err := a.Pick(ctx, band); err != nil {
			b.Fatal(err)
		}
	}
}
