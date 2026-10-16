/*
Copyright 2026 The Kubernetes Authors.
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

package prefixcacheaffinity

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// A1-I01: with the load and cache inputs held fixed, the value the plugin
// resolved decides which branch of the TTFT load gate runs. The cache-warm
// endpoint carries 100000 in-flight tokens so the estimate is
// 100000/throughput*1000 ms: 6278 ms at the built-in default and 4000 ms at a
// measured 25000 tokens/sec, against a maxTTFTPenaltyMs of 5000.
func TestA1I01_EffectiveThroughputDrivesTheLoadGateBranch(t *testing.T) {
	fixedInput := func() []fwksched.Endpoint {
		return []fwksched.Endpoint{
			makeEndpoint("cached", 100, 0, 100000),
			makeEndpoint("cold", 0, 0, 0),
		}
	}

	// Default value: the gate reopens all endpoints because the cache-warm one
	// looks saturated.
	plugin, err := newFactoryPlugin(t, map[string]any{paramMaxTTFTPenaltyMs: 5000})
	require.NoError(t, err)
	require.Equal(t, SourceDefault, plugin.config.PeakPrefillThroughputSource)
	assert.Equal(t, 2, len(plugin.Filter(context.Background(), nil, fixedInput())),
		"the default throughput must break stickiness on this fixed input")

	// Measured value for this deployment: the same input now keeps stickiness,
	// so the number in the record is what routing acts on.
	record := writeRecord(t, calibrationRecordJSON(map[string]string{keyPeakPrefillTokensPerSecond: "25000"}))
	plugin, err = newFactoryPlugin(t, map[string]any{
		paramMaxTTFTPenaltyMs:              5000,
		paramPrefillCalibrationFile:        record,
		paramPrefillCalibrationFingerprint: measuredFingerprint,
	})
	require.NoError(t, err)
	require.Equal(t, SourceCalibrated, plugin.config.PeakPrefillThroughputSource)
	require.Equal(t, 25000.0, plugin.config.PeakPrefillThroughput)
	narrowed := plugin.Filter(context.Background(), nil, fixedInput())
	require.Equal(t, 1, len(narrowed), "the measured throughput must keep the sticky set")
	assert.Equal(t, "cached", narrowed[0].GetMetadata().ID.Name)

	// The same record on a deployment whose fingerprint differs is not applied,
	// and the branch flips back: the record file itself does not route traffic.
	plugin, err = newFactoryPlugin(t, map[string]any{
		paramMaxTTFTPenaltyMs:              5000,
		paramPrefillCalibrationFile:        record,
		paramPrefillCalibrationFingerprint: "sha256:another-deployment",
	})
	require.NoError(t, err)
	assert.Equal(t, SourceDefault, plugin.config.PeakPrefillThroughputSource)
	assert.Equal(t, 2, len(plugin.Filter(context.Background(), nil, fixedInput())))
}
