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

package kvcachehintv2

import (
	"sync"
	"time"
)

// budgetTracker caps the sum of live protected tokens across retention
// scopes. Each scope holds at most one reservation; a reservation lives until
// its expiry, mirroring the directive TTL the backend enforces. Expired
// reservations are purged lazily on the next reserve call.
type budgetTracker struct {
	mu     sync.Mutex
	budget int64
	total  int64
	scopes map[string]budgetEntry
	// now is replaceable in tests.
	now func() time.Time
}

// budgetEntry is one scope's live reservation.
type budgetEntry struct {
	tokens int64
	expiry time.Time
}

func newBudgetTracker(budget int64) *budgetTracker {
	return &budgetTracker{
		budget: budget,
		scopes: make(map[string]budgetEntry),
		now:    time.Now,
	}
}

// reserve charges tokens against the budget under the scope and returns the
// charged token count. A scope's reservation renews in place: the charge is
// the larger of the existing and requested tokens, and the expiry extends to
// cover the new duration. Growth that would exceed the budget falls back to
// the existing charge, so a protected session keeps refreshing its TTL even
// under a full budget; a scope with no reservation is rejected instead, and
// the caller emits no directive.
func (b *budgetTracker) reserve(scope string, tokens int64, duration time.Duration) (int64, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()

	now := b.now()
	for s, entry := range b.scopes {
		if !entry.expiry.After(now) {
			b.total -= entry.tokens
			delete(b.scopes, s)
		}
	}

	existing, exists := b.scopes[scope]
	charged := max(tokens, existing.tokens)
	delta := charged - existing.tokens
	if !exists {
		delta = charged
	}
	if delta > 0 && b.total+delta > b.budget {
		if !exists {
			return 0, false
		}
		charged = existing.tokens
		delta = 0
	}

	expiry := now.Add(duration)
	if exists && existing.expiry.After(expiry) {
		expiry = existing.expiry
	}
	b.scopes[scope] = budgetEntry{tokens: charged, expiry: expiry}
	b.total += delta
	return charged, true
}

// usage returns the live protected token total and the configured budget.
func (b *budgetTracker) usage() (used, budget int64) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.total, b.budget
}
