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

package thunderagent

import (
	"bytes"
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFactory(t *testing.T) {
	t.Run("defaults", func(t *testing.T) {
		p, err := Factory("test", nil, nil)
		require.NoError(t, err)
		a, ok := p.(*ThunderAgent)
		require.True(t, ok)
		assert.Equal(t, float64(4194304), a.capacityTokens)
		assert.Equal(t, 0.9, a.utilThreshold)
		assert.Equal(t, time.Duration(0), a.table.actingHalfLife)
		assert.Equal(t, "x-session-final", a.sessionFinalHeader)
	})

	t.Run("header names are normalized", func(t *testing.T) {
		params := json.NewDecoder(bytes.NewBufferString(`{"sessionFinalHeader": " X-Session-Final "}`))
		p, err := Factory("test", params, nil)
		require.NoError(t, err)
		assert.Equal(t, "x-session-final", p.(*ThunderAgent).sessionFinalHeader)
	})

	invalid := map[string]string{
		"capacity":       `{"capacityTokens": -1}`,
		"half life":      `{"actingHalfLifeSeconds": -1}`,
		"util threshold": `{"utilThreshold": 1.5}`,
		"buffer tokens":  `{"bufferTokensPerProgram": -1}`,
		"starvation":     `{"headWaitStarvationMs": -1}`,
		"ttl":            `{"evictionTtlSeconds": 0}`,
		"sweep":          `{"evictionSweepSeconds": 0}`,
		"final header":   `{"sessionFinalHeader": " "}`,
	}
	for name, raw := range invalid {
		t.Run("invalid "+name, func(t *testing.T) {
			_, err := Factory("test", json.NewDecoder(bytes.NewBufferString(raw)), nil)
			assert.Error(t, err)
		})
	}
}

func TestDumpState(t *testing.T) {
	a := newTestAgent(testConfig())
	endpoints := newTestEndpoints("pod1")

	req := newRequest("program-a", 400)
	require.NoError(t, a.PreRequest(context.Background(), req, schedulingResultFor(endpoints[0])))

	raw, err := a.DumpState()
	require.NoError(t, err)
	var state dumpState
	require.NoError(t, json.Unmarshal(raw, &state))
	assert.Equal(t, 1, state.TotalPrograms)
	assert.Equal(t, int64(100), state.TotalInflightTokens)
	require.Contains(t, state.Pods, endpoints[0].GetMetadata().ID.String())
	assert.Equal(t, 1, state.Pods[endpoints[0].GetMetadata().ID.String()].Programs)
}
