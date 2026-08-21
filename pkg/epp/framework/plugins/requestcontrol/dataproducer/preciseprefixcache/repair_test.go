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

package preciseprefixcache

import (
	"testing"

	"github.com/llm-d/llm-d-router/pkg/kvevents"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNormalizeFullReportRepairConfig(t *testing.T) {
	config, err := normalizeFullReportRepairConfig(FullReportRepairConfig{})
	require.NoError(t, err)
	assert.Equal(t, defaultFullReportThreshold, config.FullReportThreshold)
	assert.Equal(t, defaultMinMissingBlocks, config.MinMissingBlocks)

	_, err = normalizeFullReportRepairConfig(FullReportRepairConfig{FullReportThreshold: 1.1})
	assert.ErrorContains(t, err, "fullReportThreshold")
	_, err = normalizeFullReportRepairConfig(FullReportRepairConfig{MinMissingBlocks: -1})
	assert.ErrorContains(t, err, "minMissingBlocks")
}

func TestValidateFullReportRepairPrerequisites(t *testing.T) {
	valid := kvevents.DefaultConfig()
	require.NoError(t, validateFullReportRepairPrerequisites(valid))

	withoutDiscovery := kvevents.DefaultConfig()
	withoutDiscovery.DiscoverPods = false
	assert.ErrorContains(t, validateFullReportRepairPrerequisites(withoutDiscovery), "discoverPods")

	globalSocket := kvevents.DefaultConfig()
	globalSocket.ZMQEndpoint = "tcp://127.0.0.1:5557"
	assert.ErrorContains(t, validateFullReportRepairPrerequisites(globalSocket), "global-socket")

	sglang := kvevents.DefaultConfig()
	sglang.EngineType = "sglang"
	assert.ErrorContains(t, validateFullReportRepairPrerequisites(sglang), "vllm")

	withReplay := kvevents.DefaultConfig()
	withReplay.PodDiscoveryConfig.ReplaySocketPort = 6000
	assert.ErrorContains(t, validateFullReportRepairPrerequisites(withReplay), "replaySocketPort")
}

func TestFullReportRepairForceWaitsForMinimumDeficit(t *testing.T) {
	r := newFullReportRepair(FullReportRepairConfig{FullReportThreshold: 0.80, MinMissingBlocks: 32})
	const endpoint = "10.0.0.1:8000"
	r.observe(endpoint, kvevents.StreamEventAttached)
	r.observe(endpoint, kvevents.StreamEventMissingParent)

	request, _ := r.shouldRequest(endpoint, repairMatch{total: 100, confirmed: 70})
	assert.False(t, request, "30 missing blocks are below the floor")
	request, reason := r.shouldRequest(endpoint, repairMatch{total: 200, confirmed: 168})
	assert.True(t, request)
	assert.Equal(t, "integrity", reason, "the short request must not consume the force bit")
}
