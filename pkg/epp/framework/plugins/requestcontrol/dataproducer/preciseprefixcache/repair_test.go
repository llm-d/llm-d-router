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
	r := newFullReportRepair(FullReportRepairConfig{FullReportThreshold: 0.80, MinMissingBlocks: 32}, 0)
	const endpoint = "10.0.0.1:8000"
	r.observe(endpoint, kvevents.StreamEventAttached)
	r.observe(endpoint, kvevents.StreamEventMissingParent)

	request, _ := r.shouldRequest(endpoint, repairMatch{total: 100, confirmed: 70})
	assert.False(t, request, "30 missing blocks are below the floor")
	request, reason := r.shouldRequest(endpoint, repairMatch{total: 200, confirmed: 168})
	assert.True(t, request)
	assert.Equal(t, "integrity", reason, "the short request must not consume the force bit")
}

func TestFullReportRepairCooldown(t *testing.T) {
	clk := testclock.NewFakePassiveClock(time.Now())
	r := newFullReportRepair(FullReportRepairConfig{FullReportThreshold: 0.80, MinMissingBlocks: 32}, 10*time.Second)
	r.clock = clk
	const endpoint = "10.0.0.1:8000"
	r.observe(endpoint, kvevents.StreamEventAttached)
	match := repairMatch{total: 200, confirmed: 100}

	request, reason := r.shouldRequest(endpoint, match)
	assert.True(t, request)
	assert.Equal(t, "threshold", reason)
	request, _ = r.shouldRequest(endpoint, match)
	assert.False(t, request, "a second request within the cooldown is suppressed")

	// A fault observed during the cooldown survives until the window closes.
	r.observe(endpoint, kvevents.StreamEventMissingParent)
	request, _ = r.shouldRequest(endpoint, match)
	assert.False(t, request)

	clk.SetTime(clk.Now().Add(11 * time.Second))
	request, reason = r.shouldRequest(endpoint, match)
	assert.True(t, request)
	assert.Equal(t, "integrity", reason, "the preserved fault is consumed after the cooldown")
}
