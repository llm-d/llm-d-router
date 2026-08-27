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

package cardinality

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
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

// Pinned values must emit their real label even when the cap is exhausted by
// unpinned values, and must not consume cap slots themselves.
func TestBoundedLabelPin(t *testing.T) {
	b := NewBoundedLabel(2)

	b.Pin("configured")
	b.Pin("configured") // idempotent
	require.Equal(t, "a", b.Bound("a"), "pin does not consume a cap slot")
	require.Equal(t, "b", b.Bound("b"), "cap still has room for a second unpinned value")
	require.Equal(t, OverflowValue, b.Bound("c"), "cap full for unpinned values")
	require.Equal(t, "configured", b.Bound("configured"), "pinned value survives a full cap")

	b.Pin("late")
	require.Equal(t, "late", b.Bound("late"), "value pinned after the cap fills still emits its real label")
}

// SetFairnessIDLabelLimit replaces the shared fairness limiter, so the cap
// applies and previously admitted values are cleared.
func TestSetFairnessIDLabelLimit(t *testing.T) {
	t.Cleanup(func() { SetFairnessIDLabelLimit(DefaultFairnessIDLabelLimit) })

	SetFairnessIDLabelLimit(2)
	require.Equal(t, "a", BoundFairnessID("a"))
	require.Equal(t, "b", BoundFairnessID("b"))
	require.Equal(t, OverflowValue, BoundFairnessID("c"), "value beyond the cap collapses to overflow")

	SetFairnessIDLabelLimit(0)
	require.Equal(t, OverflowValue, BoundFairnessID("a"), "a zero cap collapses every value to overflow")

	SetFairnessIDLabelLimit(10)
	for i := 0; i < 5; i++ {
		require.NotEqual(t, OverflowValue, BoundFairnessID(fmt.Sprintf("tenant-%d", i)),
			"raising the cap admits new values again")
	}
}
