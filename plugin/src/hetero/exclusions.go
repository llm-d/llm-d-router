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

package hetero

import (
	"sync"
	"time"
)

// PairKey names a prefill/decode pair for exclusion. Pairs with compatibility keys on both
// sides are tracked by key pair, so a failure caused by configuration excludes every pod
// pair with the same keys; otherwise by pod names.
func PairKey(prefill, decode Pod) string {
	if prefill.CompatKey != "" && decode.CompatKey != "" {
		return "key:" + prefill.CompatKey + "->" + decode.CompatKey
	}
	return "pod:" + prefill.Name + "->" + decode.Name
}

type pairState struct {
	failures int
	until    time.Time
}

// Exclusions counts consecutive failed handoffs per pair and excludes a pair for a fixed
// time once the count reaches the threshold. A success resets the count. A failed or
// rejected handoff is a hard exclusion, not a cost: while a pair is excluded the decider
// treats it as unable to exchange KV.
type Exclusions struct {
	mu        sync.Mutex
	threshold int
	ttl       time.Duration
	pairs     map[string]*pairState
}

// NewExclusions returns a tracker. A threshold of zero disables exclusion.
func NewExclusions(threshold int, ttl time.Duration) *Exclusions {
	return &Exclusions{threshold: threshold, ttl: ttl, pairs: map[string]*pairState{}}
}

// Record adds the outcome of one disaggregated request for the pair and reports whether
// this outcome started an exclusion.
func (e *Exclusions) Record(pair string, ok bool, now time.Time) bool {
	if e == nil || e.threshold <= 0 {
		return false
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	st := e.pairs[pair]
	if ok {
		if st != nil && !now.Before(st.until) {
			delete(e.pairs, pair)
		} else if st != nil {
			st.failures = 0
		}
		return false
	}
	if st == nil {
		st = &pairState{}
		e.pairs[pair] = st
	}
	if now.Before(st.until) {
		return false
	}
	st.failures++
	if st.failures >= e.threshold {
		st.failures = 0
		st.until = now.Add(e.ttl)
		return true
	}
	return false
}

// Excluded reports whether the pair is excluded at now.
func (e *Exclusions) Excluded(pair string, now time.Time) bool {
	if e == nil || e.threshold <= 0 {
		return false
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	st := e.pairs[pair]
	return st != nil && now.Before(st.until)
}

// Active returns the pairs excluded at now.
func (e *Exclusions) Active(now time.Time) []string {
	if e == nil {
		return nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	var out []string
	for k, st := range e.pairs {
		if now.Before(st.until) {
			out = append(out, k)
		}
	}
	return out
}
