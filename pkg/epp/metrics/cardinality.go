/*
Copyright 2025 The Kubernetes Authors.

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

// Model name labels are populated from the request body, which is not validated
// against a closed set of served models. Prometheus *Vec types never evict label
// combinations, so an unbounded number of distinct model names would grow the time
// series set without limit and exhaust memory. boundedLabel caps the number of
// distinct values a label may take; values beyond the cap collapse to overflowValue.
//
// Names configured through InferenceModelRewrite rules (exact-match sources and
// rewrite targets) are pinned: they always emit their real label and do not count
// against the cap, so a flood of unconfigured names cannot displace a configured
// model into the overflow bucket.
const (
	maxModelLabelValues = 1000
	overflowValue       = "other"
)

type boundedLabel struct {
	mu     sync.RWMutex
	seen   map[string]struct{}
	pinned map[string]struct{}
	max    int
}

func newBoundedLabel(max int) *boundedLabel {
	return &boundedLabel{
		seen:   make(map[string]struct{}),
		pinned: make(map[string]struct{}),
		max:    max,
	}
}

// bound returns v if it is pinned, already admitted, or there is room to admit
// it, otherwise overflowValue. A value, once admitted, always returns itself, so
// paired calls (e.g. running-request increment and decrement) stay balanced. The
// one exception is a value that folded to overflowValue and is later pinned by a
// new rewrite rule: pairs in flight across that instant can leave a small,
// bounded residue on the overflow series.
func (b *boundedLabel) bound(v string) string {
	if v == "" {
		return v
	}
	b.mu.RLock()
	_, pin := b.pinned[v]
	_, ok := b.seen[v]
	full := len(b.seen) >= b.max
	b.mu.RUnlock()
	if pin || ok {
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
	if len(b.seen) >= b.max {
		return overflowValue
	}
	b.seen[v] = struct{}{}
	return v
}

// pin marks v as always admitted without consuming a cap slot. Pins are never
// removed: a model deconfigured at runtime keeps emitting its real label, which
// matches Prometheus semantics (its existing series persist regardless) and
// keeps paired gauge calls balanced.
func (b *boundedLabel) pin(v string) {
	if v == "" {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.pinned[v] = struct{}{}
}

var modelLabelLimiter = newBoundedLabel(maxModelLabelValues)

// Fairness IDs are populated from a client request header (or an agent-identity attribute), so
// like model names their cardinality is not operator-bounded. They label per-request and
// flow-control metrics; without a cap, every distinct fairness ID ever observed permanently
// grows the time series set. maxFairnessLabelValues bounds the distinct fairness_id label
// values; values beyond the cap collapse to overflowValue.
const maxFairnessLabelValues = 1000

var fairnessLabelLimiter = newBoundedLabel(maxFairnessLabelValues)

// boundFairnessID caps the request-derived fairness_id label.
func boundFairnessID(fairnessID string) string {
	return fairnessLabelLimiter.bound(fairnessID)
}

// PreAdmitModelLabels pins the given model names so they always emit their real
// label value on model-labeled metrics, regardless of how many unconfigured
// names have been admitted. The datastore calls this when InferenceModelRewrite
// rules are loaded or updated.
func PreAdmitModelLabels(names ...string) {
	for _, n := range names {
		modelLabelLimiter.pin(n)
	}
}

// boundModel caps a single request-derived model-name label.
func boundModel(modelName string) string {
	return modelLabelLimiter.bound(modelName)
}

// boundModels caps the model-name labels shared by request metrics.
func boundModels(modelName, targetModelName string) (string, string) {
	return modelLabelLimiter.bound(modelName), modelLabelLimiter.bound(targetModelName)
}
