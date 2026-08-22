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

package metrics

import "sync"

// The model_name label is populated from the request body, which no
// coordinator config validates against a closed set. Prometheus *Vec types
// never evict label combinations, so an unbounded number of distinct model
// names would grow the time series set without limit and exhaust memory.
// boundedLabel caps the distinct values a label may take; values beyond the
// cap collapse to overflowValue. The cap matches EPP's cap for the same
// reason and for consistency between the two components.
const (
	maxModelLabelValues = 1000
	overflowValue       = "other"
)

type boundedLabel struct {
	mu    sync.RWMutex
	seen  map[string]struct{}
	limit int
}

func newBoundedLabel(limit int) *boundedLabel {
	return &boundedLabel{
		seen:  make(map[string]struct{}),
		limit: limit,
	}
}

// bound returns v if it has already been admitted or there is room to admit
// it, otherwise overflowValue. A value, once admitted, always returns itself,
// so paired calls (e.g. running-request increment and decrement) stay
// balanced.
func (b *boundedLabel) bound(v string) string {
	b.mu.RLock()
	_, ok := b.seen[v]
	full := len(b.seen) >= b.limit
	b.mu.RUnlock()
	if ok {
		return v
	}
	if full {
		return overflowValue
	}

	b.mu.Lock()
	defer b.mu.Unlock()
	if _, ok := b.seen[v]; ok {
		return v
	}
	if len(b.seen) >= b.limit {
		return overflowValue
	}
	b.seen[v] = struct{}{}
	return v
}

var modelLabelLimiter = newBoundedLabel(maxModelLabelValues)

// boundModel maps a request-derived model name to the label value emitted on
// coordinator metrics. Empty resolves to ModelUnknown before the cap is
// consulted, so a client that never sends "model" cannot exhaust the cap on
// its own.
func boundModel(modelName string) string {
	if modelName == "" {
		return ModelUnknown
	}
	return modelLabelLimiter.bound(modelName)
}
