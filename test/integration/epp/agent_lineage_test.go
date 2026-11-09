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

package epp

import (
	"encoding/json"
	"fmt"
	"slices"
	"testing"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	fwkepp "github.com/llm-d/llm-d-router/test/framework/epp"
)

// agentLineageConfigTemplate enables the agent-lineage plugin on the shared
// test configuration. compose controls requestHandler.composeFairnessIDs.
func agentLineageConfigTemplate(compose bool) string {
	composeLine := ""
	if compose {
		composeLine = "  composeFairnessIDs: true"
	}
	return fmt.Sprintf(`
apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
  - type: queue-scorer
  - type: kv-cache-utilization-scorer
  - type: prefix-cache-scorer
  - type: lora-affinity-scorer
  - type: openai-parser
  - type: agent-lineage
  - type: mock-metrics-source
  - type: core-metrics-extractor
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: queue-scorer
      - pluginRef: kv-cache-utilization-scorer
      - pluginRef: prefix-cache-scorer
      - pluginRef: lora-affinity-scorer
requestHandler:
  propagatePriority: true
%s
  parsers:
  - pluginRef: openai-parser
dataLayer:
  injectDefaults: false
  sources:
  - pluginRef: mock-metrics-source
`, composeLine)
}

// lineageRequest describes one logical request of an equivalence run.
type lineageRequest struct {
	name    string
	headers map[string]string
}

// runLineageRequests drives each request through the harness and returns the
// composed fairness ID and source each request handed to the sidecar, in
// order. A request is admitted when the router forwards it, so a returned ID
// for every request also asserts identical admission across the header
// shapes.
func runLineageRequests(t *testing.T, h *TestHarness, reqs []lineageRequest) (ids, sources []string) {
	t.Helper()

	bodyMap := map[string]any{
		"max_tokens": 100, "model": modelMyModelTarget, "prompt": "lineage-equivalence", "temperature": 0,
	}
	body, err := json.Marshal(bodyMap)
	require.NoError(t, err)

	ids = make([]string, 0, len(reqs))
	sources = make([]string, 0, len(reqs))
	for i, lr := range reqs {
		t.Run(lr.name, func(t *testing.T) {
			headers := map[string]string{
				"hi":                         "mom",
				reqcommon.RequestIDHeaderKey: fmt.Sprintf("lineage-req-%d", i),
				metadata.ObjectiveKey:        modelMyModel,
				metadata.ModelNameRewriteKey: modelMyModelTarget,
			}
			for k, v := range lr.headers {
				headers[k] = v
			}
			requests := fwkepp.ReqRaw(headers, string(body))
			// One ext_proc stream per logical request: a stream serves a
			// single request lifecycle.
			client, err := extProcPb.NewExternalProcessorClient(h.GRPCConn()).Process(t.Context())
			require.NoError(t, err)
			responses, err := fwkepp.StreamedRequest(t, client, requests, 2)
			require.NoError(t, err)
			require.Len(t, responses, 2, "request %q must be routed: admission differs across header shapes", lr.name)
			setHeaders := responses[0].GetRequestHeaders().GetResponse().GetHeaderMutation().GetSetHeaders()
			id := headerValue(setHeaders, tracing.FairnessIDHandoffHeader)
			require.NotEmpty(t, id, "request %q carries no fairness-ID handoff", lr.name)
			source := headerValue(setHeaders, tracing.FairnessSourceHandoffHeader)
			require.NotEmpty(t, source, "request %q carries no fairness-source handoff", lr.name)
			ids = append(ids, id)
			sources = append(sources, source)
		})
	}
	return ids, sources
}

// TestAgentLineage_ComposedEquivalence drives the same agent tree through
// the four harness header shapes under two tenants. The composed fairness ID
// is the flow-control queue key, so equal IDs assert one queue per program:
// a three-child fork collapses to its root under every harness, the fork
// stays separate from the tenant's chat turn, and tenants never share keys.
func TestAgentLineage_ComposedEquivalence(t *testing.T) {
	const (
		tenantP = "tenant-p"
		tenantQ = "tenant-q"
	)
	codexTurn := func(session, thread string) string {
		payload, err := json.Marshal(map[string]any{"session_id": session, "thread_id": thread, "request_kind": "turn"})
		require.NoError(t, err)
		return string(payload)
	}

	stamped := []lineageRequest{
		// Claude Code: one session ID shared by the whole tree.
		{name: "claude-code root", headers: map[string]string{metadata.FlowFairnessIDKey: tenantP, "x-claude-code-session-id": "cc-root"}},
		{name: "claude-code sub-agent", headers: map[string]string{metadata.FlowFairnessIDKey: tenantP, "x-claude-code-session-id": "cc-root"}},
		// Codex: turn metadata carries the root session; the flat
		// session-id differs between root and sub-agent threads.
		{name: "codex root thread", headers: map[string]string{metadata.FlowFairnessIDKey: tenantP, "x-codex-turn-metadata": codexTurn("codex-root", "t0"), "session-id": "derived-cache-key"}},
		{name: "codex sub-agent thread", headers: map[string]string{metadata.FlowFairnessIDKey: tenantP, "x-codex-turn-metadata": codexTurn("codex-root", "t1"), "session-id": "codex-root", "thread-id": "t1"}},
		// opencode: fresh session per child, parent named one hop up.
		{name: "opencode root", headers: map[string]string{metadata.FlowFairnessIDKey: tenantP, "x-opencode-session-id": "oc-root", "x-session-affinity": "oc-root"}},
		{name: "opencode sub-agent 1", headers: map[string]string{metadata.FlowFairnessIDKey: tenantP, "x-opencode-session-id": "oc-s1", "x-opencode-parent-session-id": "oc-root", "x-parent-session-id": "oc-root", "x-session-affinity": "oc-root"}},
		{name: "opencode sub-agent 2", headers: map[string]string{metadata.FlowFairnessIDKey: tenantP, "x-opencode-session-id": "oc-s2", "x-opencode-parent-session-id": "oc-root", "x-parent-session-id": "oc-root", "x-session-affinity": "oc-root"}},
		{name: "opencode sub-agent 3", headers: map[string]string{metadata.FlowFairnessIDKey: tenantP, "x-opencode-session-id": "oc-s3", "x-opencode-parent-session-id": "oc-root", "x-parent-session-id": "oc-root", "x-session-affinity": "oc-root"}},
		{name: "opencode grandchild", headers: map[string]string{metadata.FlowFairnessIDKey: tenantP, "x-opencode-session-id": "oc-ss1", "x-parent-session-id": "oc-s1"}},
		// The tenant's interactive chat turn: a separate program queue.
		{name: "tenant-p chat turn", headers: map[string]string{metadata.FlowFairnessIDKey: tenantP, "x-claude-code-session-id": "chat-session"}},
		// A second tenant with the same session value keeps its own queue.
		{name: "tenant-q claude-code root", headers: map[string]string{metadata.FlowFairnessIDKey: tenantQ, "x-claude-code-session-id": "cc-root"}},
	}

	unstamped := []lineageRequest{
		// No fairness header: the same fork composes onto the default
		// tenant. Four opencode session IDs collapse to one program key.
		{name: "unstamped opencode root", headers: map[string]string{"x-opencode-session-id": "oc2-root"}},
		{name: "unstamped opencode sub-agent 1", headers: map[string]string{"x-opencode-session-id": "oc2-s1", "x-parent-session-id": "oc2-root"}},
		{name: "unstamped opencode sub-agent 2", headers: map[string]string{"x-opencode-session-id": "oc2-s2", "x-parent-session-id": "oc2-root"}},
		{name: "unstamped opencode sub-agent 3", headers: map[string]string{"x-opencode-session-id": "oc2-s3", "x-parent-session-id": "oc2-root"}},
		// Headerless traffic lands in the default flow.
		{name: "headerless", headers: map[string]string{}},
	}

	ctx := t.Context()
	h := NewTestHarness(ctx, t, WithTracing(), WithStandardMode(), WithConfigText(agentLineageConfigTemplate(true))).WithBaseResources()
	h.WithPods([]PodState{
		P(0, 3, 0.2),
		P(1, 0, 0.1),
		P(2, 10, 0.2),
	}).WaitForSync(3, modelMyModel).WaitForReadyPodsMetric(3)

	ids, sources := runLineageRequests(t, h, append(append([]lineageRequest{}, stamped...), unstamped...))

	stampedIDs := ids[:len(stamped)]
	unstampedIDs := ids[len(stamped):]
	stampedSources := sources[:len(stamped)]
	unstampedSources := sources[len(stamped):]

	want := []string{
		tenantP + "/cc-root",      // claude-code root
		tenantP + "/cc-root",      // claude-code sub-agent
		tenantP + "/codex-root",   // codex root thread
		tenantP + "/codex-root",   // codex sub-agent thread
		tenantP + "/oc-root",      // opencode root
		tenantP + "/oc-root",      // opencode sub-agent 1
		tenantP + "/oc-root",      // opencode sub-agent 2
		tenantP + "/oc-root",      // opencode sub-agent 3
		tenantP + "/oc-root",      // opencode grandchild
		tenantP + "/chat-session", // tenant-p chat turn
		tenantQ + "/cc-root",      // tenant-q claude-code root
	}
	require.Equal(t, want, stampedIDs, "stamped traffic: one queue per program under every harness")

	wantUnstamped := []string{
		"default-tenant/oc2-root",
		"default-tenant/oc2-root",
		"default-tenant/oc2-root",
		"default-tenant/oc2-root",
		reqcommon.DefaultFairnessID,
	}
	require.Equal(t, wantUnstamped, unstampedIDs, "unstamped traffic: the fork collapses onto the default tenant and headerless lands in the default flow")

	wantSources := append(
		slices.Repeat([]string{tracing.AttributionSourceHeader}, len(stamped)),
		slices.Repeat([]string{tracing.AttributionSourceDerivedRoot}, len(unstamped)-1)...,
	)
	wantSources = append(wantSources, tracing.AttributionSourceDefault)
	require.Equal(t, wantSources, append(stampedSources, unstampedSources...), "attribution source follows the resolution branch")
}

// TestAgentLineage_CompositionDisabled drives the same fork with
// requestHandler.composeFairnessIDs off: the collapse still holds (the root
// keys the queue bare) and a fairness header keys the flow as sent.
func TestAgentLineage_CompositionDisabled(t *testing.T) {
	reqs := []lineageRequest{
		{name: "opencode root", headers: map[string]string{"x-opencode-session-id": "oc3-root"}},
		{name: "opencode sub-agent", headers: map[string]string{"x-opencode-session-id": "oc3-s1", "x-parent-session-id": "oc3-root"}},
		{name: "opencode grandchild", headers: map[string]string{"x-opencode-session-id": "oc3-ss1", "x-parent-session-id": "oc3-s1"}},
		{name: "fairness header as sent", headers: map[string]string{metadata.FlowFairnessIDKey: "tenant-z", "x-opencode-session-id": "s"}},
	}

	ctx := t.Context()
	h := NewTestHarness(ctx, t, WithTracing(), WithStandardMode(), WithConfigText(agentLineageConfigTemplate(false))).WithBaseResources()
	h.WithPods([]PodState{
		P(0, 3, 0.2),
		P(1, 0, 0.1),
		P(2, 10, 0.2),
	}).WaitForSync(3, modelMyModel).WaitForReadyPodsMetric(3)

	ids, sources := runLineageRequests(t, h, reqs)

	require.Equal(t, []string{
		"oc3-root",
		"oc3-root",
		"oc3-root",
		"tenant-z",
	}, ids, "composition disabled: the fork collapses on the bare root and the fairness header keys the flow as sent")
	require.Equal(t, []string{
		tracing.AttributionSourceDerivedRoot,
		tracing.AttributionSourceDerivedRoot,
		tracing.AttributionSourceDerivedRoot,
		tracing.AttributionSourceHeader,
	}, sources, "composition disabled: the source follows the resolution branch")
}
