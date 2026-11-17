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

package debugdump

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"
)

func TestNames(t *testing.T) {
	names := Names(map[types.NamespacedName]int{
		{Namespace: "default", Name: "b"}: 1,
		{Namespace: "default", Name: "a"}: 2,
	})
	assert.ElementsMatch(t, []string{"default/a", "default/b"}, names)
}

func TestMarshalSortsAndCaps(t *testing.T) {
	names := make([]string, 0, MaxEndpoints+5)
	for i := MaxEndpoints + 4; i >= 0; i-- {
		names = append(names, fmt.Sprintf("default/ep-%03d", i))
	}

	payload, err := Marshal(names)
	require.NoError(t, err)

	var state State
	require.NoError(t, json.Unmarshal(payload, &state))
	assert.Equal(t, MaxEndpoints+5, state.TotalEndpoints)
	assert.Equal(t, MaxEndpoints, state.MaxEndpoints)
	require.Len(t, state.Endpoints, MaxEndpoints)
	assert.Equal(t, "default/ep-000", state.Endpoints[0])
	assert.Equal(t, fmt.Sprintf("default/ep-%03d", MaxEndpoints-1), state.Endpoints[MaxEndpoints-1])
}

func TestMarshalEmpty(t *testing.T) {
	payload, err := Marshal(nil)
	require.NoError(t, err)

	var state State
	require.NoError(t, json.Unmarshal(payload, &state))
	assert.Empty(t, state.Endpoints)
	assert.Equal(t, 0, state.TotalEndpoints)
	assert.Equal(t, MaxEndpoints, state.MaxEndpoints)
}
