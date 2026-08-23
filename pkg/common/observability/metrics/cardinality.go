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

// OverflowValue is the label value emitted by BoundedLabel.Bound once the cap
// fills. Callers that need to distinguish an "unknown" (empty) request-derived
// label value from the cap-overflow value should keep their own sentinel and
// gate empty inputs at the wrapper layer.
const OverflowValue = "other"

// BoundedLabel caps the distinct values a Prometheus label may take. Values beyond
// the cap collapse to OverflowValue. Values passed to Pin are always admitted and
// do not count against the cap; a pinned value continues to emit its real label
// even after the cap has filled.
//
// The primitive is safe for concurrent use. It is agnostic of the label domain:
// caller packages construct their own limiters, choose their cap, and wrap Bound
// with any domain-specific empty-input handling.
type BoundedLabel struct {
	mu     sync.RWMutex
	seen   map[string]struct{}
	pinned map[string]struct{}
	limit  int
}

// NewBoundedLabel returns a BoundedLabel that admits up to limit distinct
// unpinned values before folding further values into OverflowValue.
func NewBoundedLabel(limit int) *BoundedLabel {
	return &BoundedLabel{
		seen:   make(map[string]struct{}),
		pinned: make(map[string]struct{}),
		limit:  limit,
	}
}

// Bound returns v if it is empty, pinned, already admitted, or there is room
// to admit it; otherwise OverflowValue. A value, once admitted, always returns
// itself, so paired calls (e.g. running-request increment and decrement) stay
// balanced. The one exception is a value that folded to OverflowValue and is
// later pinned: pairs in flight across that instant can leave a small bounded
// residue on the overflow series.
func (b *BoundedLabel) Bound(v string) string {
	if v == "" {
		return v
	}
	b.mu.RLock()
	_, pin := b.pinned[v]
	_, ok := b.seen[v]
	full := len(b.seen) >= b.limit
	b.mu.RUnlock()
	if pin || ok {
		return v
	}
	if full {
		return OverflowValue
	}

	b.mu.Lock()
	defer b.mu.Unlock()
	if _, ok := b.seen[v]; ok {
		return v
	}
	if len(b.seen) >= b.limit {
		return OverflowValue
	}
	b.seen[v] = struct{}{}
	return v
}

// Pin marks v as always admitted without consuming a cap slot. Pins are never
// removed: a value deconfigured at runtime keeps emitting its real label,
// which matches Prometheus semantics (its existing series persist regardless)
// and keeps paired gauge calls balanced.
func (b *BoundedLabel) Pin(v string) {
	if v == "" {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.pinned[v] = struct{}{}
}
