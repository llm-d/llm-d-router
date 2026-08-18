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

import (
	"fmt"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestBoundedLabel_AdmitsUpToCapThenOverflows(t *testing.T) {
	b := newBoundedLabel(3)
	require.Equal(t, "a", b.bound("a"))
	require.Equal(t, "b", b.bound("b"))
	require.Equal(t, "c", b.bound("c"))
	// Fourth distinct value spills to overflow.
	require.Equal(t, overflowValue, b.bound("d"))
	// Already-admitted values keep their real label.
	require.Equal(t, "a", b.bound("a"))
}

func TestBoundedLabel_ConcurrentAdmissionsUnderCap(t *testing.T) {
	b := newBoundedLabel(1000)
	var wg sync.WaitGroup
	for i := 0; i < 500; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			v := fmt.Sprintf("m%d", i)
			require.Equal(t, v, b.bound(v))
		}(i)
	}
	wg.Wait()
}

func TestBoundModel_EmptyIsUnknown(t *testing.T) {
	// Empty model name resolves to ModelUnknown before touching the cap, so
	// a flood of empty-model requests can never exhaust the cap.
	require.Equal(t, ModelUnknown, boundModel(""))
}
