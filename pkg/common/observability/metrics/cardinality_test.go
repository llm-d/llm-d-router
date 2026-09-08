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
