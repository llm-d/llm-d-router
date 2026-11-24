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

package agentidentity

import (
	"context"
	"hash/maphash"
	"sync"
	"time"
)

const (
	// activeSessionWindow is how recently a session must have sent a request
	// to count as active.
	activeSessionWindow = 5 * time.Minute
	// activeSessionRefresh is how often expired sessions are dropped and the
	// gauge is recomputed.
	activeSessionRefresh = 15 * time.Second
	// maxActiveSessions bounds memory. Sessions first seen while the set is
	// full are not counted, so the gauge is then a lower bound.
	maxActiveSessions = 100000
)

// activeSessions counts distinct session identities seen within
// activeSessionWindow. It keeps one 8-byte seeded hash per session with its
// last-seen time and harness, nothing else: no raw IDs and no per-session
// statistics.
type activeSessions struct {
	seed maphash.Seed // per process, so hashes are not comparable across restarts

	mu       sync.Mutex
	sessions map[uint64]activeSession
	limit    int
}

type activeSession struct {
	lastSeen time.Time
	harness  string
}

func newActiveSessions(limit int) *activeSessions {
	return &activeSessions{
		seed:     maphash.MakeSeed(),
		sessions: make(map[uint64]activeSession),
		limit:    limit,
	}
}

// observe records that the session sent a request at now. A session keeps the
// harness of its first request.
func (a *activeSessions) observe(id, harness string, now time.Time) {
	key := maphash.String(a.seed, id)
	a.mu.Lock()
	defer a.mu.Unlock()
	if s, ok := a.sessions[key]; ok {
		s.lastSeen = now
		a.sessions[key] = s
		return
	}
	if len(a.sessions) >= a.limit {
		return
	}
	a.sessions[key] = activeSession{lastSeen: now, harness: harness}
}

// expire drops sessions not seen within the window and returns the remaining
// count per harness.
func (a *activeSessions) expire(now time.Time) map[string]int {
	a.mu.Lock()
	defer a.mu.Unlock()
	counts := make(map[string]int)
	for key, s := range a.sessions {
		if now.Sub(s.lastSeen) > activeSessionWindow {
			delete(a.sessions, key)
			continue
		}
		counts[s.harness]++
	}
	return counts
}

// runActiveSessions refreshes the gauge until ctx is cancelled.
func (p *Plugin) runActiveSessions(ctx context.Context) {
	ticker := time.NewTicker(activeSessionRefresh)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			p.refreshActiveSessions(time.Now())
		}
	}
}

// refreshActiveSessions sets one series per harness that has active sessions
// and removes the series of harnesses that no longer have any. It is called
// from one goroutine only.
func (p *Plugin) refreshActiveSessions(now time.Time) {
	counts := p.active.expire(now)
	for harness, n := range counts {
		p.activeGauge.WithLabelValues(harness).Set(float64(n))
	}
	for harness := range p.activeHarnesses {
		if _, ok := counts[harness]; !ok {
			p.activeGauge.DeleteLabelValues(harness)
		}
	}
	p.activeHarnesses = counts
}
