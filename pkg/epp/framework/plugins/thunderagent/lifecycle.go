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
	"strings"
	"time"

	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// isSessionFinal reports whether the request carries the session-final
// marker. The marker rides on the session's real last turn; a standalone
// release request (final header on an empty prompt) also works because it
// flows through as a tiny turn.
func isSessionFinal(request *fwksched.InferenceRequest, header string) bool {
	v := strings.TrimSpace(request.Headers[header])
	return strings.EqualFold(v, "true") || v == "1"
}

func (a *ThunderAgent) runEviction(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			a.evictIdle(time.Now())
		}
	}
}

// evictIdle drops programs with no request in flight and no activity within
// the TTL. Programs with in-flight requests are kept so the ResponseBody
// bookkeeping for those requests still finds its state.
func (a *ThunderAgent) evictIdle(now time.Time) {
	t := a.table
	var evicted int
	t.mu.Lock()
	for id, st := range t.programs {
		if st.inflightTokens > 0 {
			continue
		}
		if now.Sub(st.lastActivity) <= t.ttl {
			continue
		}
		delete(t.programs, id)
		evicted++
	}
	t.mu.Unlock()
	a.metrics.ttlEvictions.Add(float64(evicted))
}
