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
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	testclock "k8s.io/utils/clock/testing"

	"github.com/llm-d/llm-d-router/pkg/kvevents"
)

// newTestRepair returns a vLLM repair tracker with the default thresholds whose
// supported endpoints have already observed an origin-tagged store.
func newTestRepair(cooldown time.Duration, supported ...string) *fullReportRepair {
	r := newFullReportRepair(FullReportRepairConfig{
		FullReportThreshold: defaultFullReportThreshold,
		MinMissingBlocks:    defaultMinMissingBlocks,
	}, cooldown, vllmFullReport)
	for _, endpoint := range supported {
		r.observe(endpoint, kvevents.StreamEventReportSupported)
	}
	return r
}

func TestNormalizeFullReportRepairConfig(t *testing.T) {
	config, cooldown, err := normalizeFullReportRepairConfig(FullReportRepairConfig{})
	require.NoError(t, err)
	assert.Equal(t, defaultFullReportThreshold, config.FullReportThreshold)
	assert.Equal(t, defaultMinMissingBlocks, config.MinMissingBlocks)
	assert.Equal(t, defaultReportCooldown, cooldown)

	_, cooldown, err = normalizeFullReportRepairConfig(FullReportRepairConfig{Cooldown: "3s"})
	require.NoError(t, err)
	assert.Equal(t, 3*time.Second, cooldown)

	_, _, err = normalizeFullReportRepairConfig(FullReportRepairConfig{FullReportThreshold: 1.1})
	assert.ErrorContains(t, err, "fullReportThreshold")
	_, _, err = normalizeFullReportRepairConfig(FullReportRepairConfig{MinMissingBlocks: -1})
	assert.ErrorContains(t, err, "minMissingBlocks")
	_, _, err = normalizeFullReportRepairConfig(FullReportRepairConfig{Cooldown: "not-a-duration"})
	assert.ErrorContains(t, err, "cooldown")
	_, _, err = normalizeFullReportRepairConfig(FullReportRepairConfig{Cooldown: "-1s"})
	assert.ErrorContains(t, err, "cooldown")
}

func TestFullReportRequestFor(t *testing.T) {
	request, err := fullReportRequestFor(kvevents.DefaultConfig())
	require.NoError(t, err)
	assert.NotNil(t, request)

	withoutDiscovery := kvevents.DefaultConfig()
	withoutDiscovery.DiscoverPods = false
	_, err = fullReportRequestFor(withoutDiscovery)
	assert.ErrorContains(t, err, "discoverPods")

	globalSocket := kvevents.DefaultConfig()
	globalSocket.ZMQEndpoint = "tcp://127.0.0.1:5557"
	_, err = fullReportRequestFor(globalSocket)
	assert.ErrorContains(t, err, "global-socket")

	sglang := kvevents.DefaultConfig()
	sglang.EngineType = "sglang"
	_, err = fullReportRequestFor(sglang)
	assert.ErrorContains(t, err, "vllm")

	withReplay := kvevents.DefaultConfig()
	withReplay.PodDiscoveryConfig.ReplaySocketPort = 6000
	_, err = fullReportRequestFor(withReplay)
	assert.ErrorContains(t, err, "replaySocketPort")
}

func TestFullReportRepairForceBypassesMinimumDeficit(t *testing.T) {
	const endpoint = "10.0.0.1:8000"
	r := newTestRepair(0, endpoint)
	r.observe(endpoint, kvevents.StreamEventMissingParent, kvevents.StreamBlock{Hash: 42})

	_, request := r.shouldRequest(endpoint, repairMatch{total: 100, confirmed: 70})
	assert.True(t, request, "an integrity fault bypasses the ordinary gap floor")
	reason, request := r.shouldRequest(endpoint, repairMatch{total: 8, confirmed: 7})
	assert.True(t, request, "an integrity fault needs no minimum prompt length")
	assert.Equal(t, "integrity", reason)
}

func TestFullReportRepairCooldown(t *testing.T) {
	clk := testclock.NewFakePassiveClock(time.Now())
	const endpoint = "10.0.0.1:8000"
	r := newTestRepair(10*time.Second, endpoint)
	r.clock = clk
	match := repairMatch{total: 200, confirmed: 100}

	reason, request := r.shouldRequest(endpoint, match)
	assert.True(t, request)
	assert.Equal(t, "threshold", reason)
	_, request = r.shouldRequest(endpoint, match)
	assert.True(t, request, "deciding does not start the cooldown")
	require.True(t, r.reserve(endpoint))
	assert.False(t, r.reserve(endpoint), "a request that decided concurrently cannot start a second report")
	_, request = r.shouldRequest(endpoint, match)
	assert.False(t, request, "a request within the cooldown is suppressed")

	// A fault observed during the cooldown survives until the window closes.
	r.observe(endpoint, kvevents.StreamEventMissingParent, kvevents.StreamBlock{Hash: 42})
	_, request = r.shouldRequest(endpoint, match)
	assert.False(t, request)

	clk.SetTime(clk.Now().Add(11 * time.Second))
	reason, request = r.shouldRequest(endpoint, match)
	assert.True(t, request)
	assert.Equal(t, "integrity", reason, "the preserved fault is retried after the cooldown")
}

func TestFullReportRepairIntegritySurvivesReport(t *testing.T) {
	const endpoint = "10.0.0.1:8000"
	r := newTestRepair(0, endpoint)
	r.observe(endpoint, kvevents.StreamEventMissingParent, kvevents.StreamBlock{Hash: 42})
	_, requested := r.shouldRequest(endpoint, repairMatch{total: 200, confirmed: 168})
	require.True(t, requested)
	require.True(t, r.reserve(endpoint))
	reason, requested := r.shouldRequest(endpoint, repairMatch{total: 200, confirmed: 168})
	assert.True(t, requested, "requesting a report does not prove the missing lineage was repaired")
	assert.Equal(t, "integrity", reason)
}

func TestFullReportRepairLifecycle(t *testing.T) {
	const endpoint = "10.0.0.1:8000"
	match := repairMatch{total: 200, confirmed: 100}
	r := newTestRepair(0)

	r.observe(endpoint, kvevents.StreamEventStored, kvevents.StreamBlock{Hash: 42})
	r.observe(endpoint, kvevents.StreamEventRemoved, kvevents.StreamBlock{Hash: 42})
	assert.Empty(t, r.endpoints, "stores and removals alone keep no endpoint state")

	r.observe(endpoint, kvevents.StreamEventMissingParent, kvevents.StreamBlock{Hash: 42})
	_, requested := r.shouldRequest(endpoint, match)
	assert.False(t, requested, "untagged reports cannot be safely reference-counted")
	r.observe(endpoint, kvevents.StreamEventReportSupported)
	reason, requested := r.shouldRequest(endpoint, match)
	assert.True(t, requested)
	assert.Equal(t, "integrity", reason)

	// Cache resets and retired subscribers both clear the endpoint.
	r.observe(endpoint, kvevents.StreamEventCleared)
	assert.Empty(t, r.endpoints)
	r.observe(endpoint, kvevents.StreamEventMissingParent, kvevents.StreamBlock{Hash: 43})
	_, requested = r.shouldRequest(endpoint, match)
	assert.False(t, requested, "the stream must tag origins again after a reset")
	r.observe(endpoint, kvevents.StreamEventReportSupported)
	reason, requested = r.shouldRequest(endpoint, match)
	assert.True(t, requested, "a fault observed after a reset is repairable")
	assert.Equal(t, "integrity", reason)
}

func TestFullReportRepairResolvesOnlyAffectedBlocks(t *testing.T) {
	const endpoint = "pod"
	r := newTestRepair(0, endpoint)
	a := kvevents.StreamBlock{Hash: 42, DeviceTier: "gpu", GroupIdx: 0}
	b := kvevents.StreamBlock{Hash: 43, DeviceTier: "gpu", GroupIdx: 0}
	r.observe(endpoint, kvevents.StreamEventMissingParent, a, b)
	match := repairMatch{total: 200, confirmed: 168}
	for _, unrelated := range []kvevents.StreamBlock{
		{Hash: 99, DeviceTier: "gpu", GroupIdx: 0},
		{Hash: 42, DeviceTier: "cpu", GroupIdx: 0},
		{Hash: 42, DeviceTier: "gpu", GroupIdx: 1},
	} {
		r.observe(endpoint, kvevents.StreamEventStored, unrelated)
		r.observe(endpoint, kvevents.StreamEventRemoved, unrelated)
		_, requested := r.shouldRequest(endpoint, match)
		assert.True(t, requested)
	}
	r.observe(endpoint, kvevents.StreamEventStored, a)
	_, requested := r.shouldRequest(endpoint, match)
	assert.True(t, requested, "the other missing block remains unresolved")
	r.observe(endpoint, kvevents.StreamEventRemoved, b)
	_, requested = r.shouldRequest(endpoint, match)
	assert.False(t, requested)
}
