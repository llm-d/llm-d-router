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

import (
	"fmt"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/common/request"
)

func TestBoundedLabel(t *testing.T) {
	b := NewBoundedLabel(2)

	require.Equal(t, "a", b.Bound("a"), "first value admitted")
	require.Equal(t, "b", b.Bound("b"), "second value admitted")
	require.Equal(t, OverflowValue, b.Bound("c"), "value beyond cap collapses to overflow")
	require.Equal(t, "a", b.Bound("a"), "already-admitted value still returns itself after cap")
	require.Equal(t, OverflowValue, b.Bound("d"), "further unseen values keep collapsing")
	require.Equal(t, "", b.Bound(""), "empty value passes through without consuming a slot")
}

// Pinned names model operator-configured values (for example EPP's
// InferenceModelRewrite sources and targets): they must emit their real label
// even when the cap is exhausted by unconfigured values, and must not consume
// cap slots themselves.
func TestBoundedLabelPin(t *testing.T) {
	b := NewBoundedLabel(2)

	b.Pin("configured")
	b.Pin("configured") // idempotent
	require.Equal(t, "a", b.Bound("a"), "pin does not consume a cap slot")
	require.Equal(t, "b", b.Bound("b"), "cap still has room for a second unconfigured name")
	require.Equal(t, OverflowValue, b.Bound("c"), "cap full for unconfigured names")
	require.Equal(t, "configured", b.Bound("configured"), "pinned name survives a full cap")

	b.Pin("late")
	require.Equal(t, "late", b.Bound("late"), "name pinned after the cap fills still emits its real label")
}

func TestBoundedLabelConcurrentAdmissionsUnderCap(t *testing.T) {
	b := NewBoundedLabel(1000)
	got := make([]string, 500)
	var wg sync.WaitGroup
	for i := 0; i < 500; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			got[i] = b.Bound(fmt.Sprintf("m%d", i))
		}(i)
	}
	wg.Wait()
	for i, g := range got {
		require.Equal(t, fmt.Sprintf("m%d", i), g)
	}
}

// Overflow is reported only when the cap folds a value. A client sending the literal overflow
// value takes a slot like any other ID while the cap has room, and is folded once it is full.
func TestBoundedLabelBoundWithOverflow(t *testing.T) {
	type step struct {
		in       string
		want     string
		overflow bool
	}
	tests := []struct {
		name  string
		steps []step
	}{
		{name: "literal overflow value under the cap", steps: []step{
			{in: "", want: ""},
			{in: OverflowValue, want: OverflowValue},
			{in: "a", want: "a"},
			{in: "b", want: OverflowValue, overflow: true},
			{in: "a", want: "a"},
			{in: OverflowValue, want: OverflowValue},
			{in: "pinned", want: "pinned"},
		}},
		{name: "literal overflow value after the cap fills", steps: []step{
			{in: "a", want: "a"},
			{in: "b", want: "b"},
			{in: OverflowValue, want: OverflowValue, overflow: true},
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			b := NewBoundedLabel(2)
			b.Pin("pinned")
			for _, s := range test.steps {
				got, overflow := b.BoundWithOverflow(s.in)
				require.Equal(t, s.want, got, "value %q", s.in)
				require.Equal(t, s.overflow, overflow, "overflow for %q", s.in)
			}
		})
	}
}

// Racing admissions near a full cap must report exactly one overflow per folded value, or the
// overflow counter drifts from the series it describes.
func TestBoundedLabelBoundWithOverflowConcurrent(t *testing.T) {
	const n = 500
	b := NewBoundedLabel(1)
	start := make(chan struct{})
	var wg sync.WaitGroup
	var mu sync.Mutex
	folded, overflows := 0, 0
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			got, overflow := b.BoundWithOverflow(fmt.Sprintf("m%d", i))
			mu.Lock()
			defer mu.Unlock()
			if got == OverflowValue {
				folded++
			}
			if overflow {
				overflows++
			}
		}(i)
	}
	close(start)
	wg.Wait()
	require.Equal(t, n-1, folded)
	require.Equal(t, n-1, overflows)
}

// The fallback fairness ID stays visible once client IDs fill the cap, unless the
// operator disabled per-ID labels with a cap of 0.
func TestFairnessIDLimitPinsDefaultFlow(t *testing.T) {
	t.Cleanup(func() { SetFairnessIDLabelLimit(DefaultFairnessIDLabelLimit) })

	SetFairnessIDLabelLimit(1)
	require.Equal(t, "team-a", BoundFairnessID("team-a"))
	got, overflow := BoundFairnessIDWithOverflow("team-b")
	require.Equal(t, OverflowValue, got)
	require.True(t, overflow)
	got, overflow = BoundFairnessIDWithOverflow(request.DefaultFairnessID)
	require.Equal(t, request.DefaultFairnessID, got)
	require.False(t, overflow)

	SetFairnessIDLabelLimit(0)
	require.Equal(t, OverflowValue, BoundFairnessID(request.DefaultFairnessID))
}
