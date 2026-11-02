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
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"
	crmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	eppRunner "github.com/llm-d/llm-d-router/cmd/epp/runner"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	dlmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/mocks"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	fwknet "github.com/llm-d/llm-d-router/test/framework/net"
)

// endpointHealthConfigBase routes at random among the endpoints the health
// filter keeps. The saturation detector's staleness threshold is raised so that
// its filter cannot drop an endpoint whose polls fail, leaving the health filter
// as the only check under test. The slots hold the health filter's plugin entry
// and its profile reference, or nothing.
const endpointHealthConfigBase = `
apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
  - type: openai-parser
  - type: mock-metrics-source
  - type: random-picker
  - type: utilization-detector
    parameters:
      metricsStalenessThreshold: 1h
%s
requestHandler:
  parsers:
  - pluginRef: openai-parser
dataLayer:
  injectDefaults: false
  sources:
  - pluginRef: mock-metrics-source
flowControl:
  saturationDetector:
    pluginRef: utilization-detector
schedulingProfiles:
  - name: default
    plugins:
%s
    - pluginRef: random-picker
`

const healthFilterName = "health"

func endpointHealthConfig(parameters string) string {
	if parameters == "" {
		return fmt.Sprintf(endpointHealthConfigBase, "", "")
	}
	plugin := fmt.Sprintf("  - type: endpoint-health-filter\n    name: %s\n    parameters:\n%s", healthFilterName, parameters)
	return fmt.Sprintf(endpointHealthConfigBase, plugin, "    - pluginRef: "+healthFilterName)
}

var errBadGateway = errors.New("unexpected status code: 502")

// setupHealthHarness starts an EPP over four healthy endpoints and returns the
// address and ID of the one the tests make fail.
func setupHealthHarness(t *testing.T, configText string) (*TestHarness, string, types.NamespacedName) {
	t.Helper()
	pods := []PodState{
		P(0, 0, 0, modelMyModelTarget),
		P(1, 0, 0, modelMyModelTarget),
		P(2, 0, 0, modelMyModelTarget),
		P(3, 0, 0, modelMyModelTarget),
	}
	h := NewTestHarness(t.Context(), t, WithStandardMode(), WithConfigText(configText)).WithBaseResources()
	h.WithPods(pods).WaitForSync(len(pods), modelMyModel).WaitForMetricsDelivery()
	return h, "192.168.1.2:8000", types.NamespacedName{Namespace: h.Namespace, Name: "pod-1-rank-0"}
}

// routeAndRespond sends one request and answers it with the status that
// respond returns for the endpoint the EPP picked. It returns that endpoint.
func routeAndRespond(t *testing.T, h *TestHarness, respond func(endpoint string) string) string {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	client, err := extProcPb.NewExternalProcessorClient(h.grpcConn).Process(ctx)
	require.NoError(t, err)

	for _, req := range newRoutedRequest(nil) {
		require.NoError(t, client.Send(req))
	}
	headersResp, err := client.Recv()
	require.NoError(t, err)
	endpoint := headerValue(headersResp.GetRequestHeaders().GetResponse().GetHeaderMutation().GetSetHeaders(), metadata.DestinationEndpointKey)
	require.NotEmpty(t, endpoint, "request headers response must set the destination endpoint")
	_, err = client.Recv()
	require.NoError(t, err)

	status := respond(endpoint)
	body := `{"error":{"message":"upstream unavailable"}}`
	if status == "200" {
		body = `{"choices":[{"finish_reason":"stop","index":0,"message":{"content":"hi","role":"assistant"}}],"model":"` +
			modelMyModelTarget + `","object":"chat.completion","usage":{"completion_tokens":2,"prompt_tokens":3,"total_tokens":5}}`
	}
	for _, req := range ReqResponseOnly(map[string]string{"content-type": "application/json", ":status": status}, body) {
		require.NoError(t, client.Send(req))
	}
	for range 2 {
		_, err = client.Recv()
		require.NoError(t, err)
	}
	require.NoError(t, client.CloseSend())
	return endpoint
}

func respondOK(string) string { return "200" }

// gatheredValue returns the value of the named counter or gauge series with the
// given labels, or zero when the series does not exist.
func gatheredValue(t *testing.T, name string, labels map[string]string) float64 {
	t.Helper()
	families, err := crmetrics.Registry.Gather()
	require.NoError(t, err)
	for _, family := range families {
		if family.GetName() != name {
			continue
		}
	series:
		for _, m := range family.GetMetric() {
			got := make(map[string]string, len(m.GetLabel()))
			for _, l := range m.GetLabel() {
				got[l.GetName()] = l.GetValue()
			}
			for k, v := range labels {
				if got[k] != v {
					continue series
				}
			}
			if m.GetCounter() != nil {
				return m.GetCounter().GetValue()
			}
			return m.GetGauge().GetValue()
		}
	}
	return 0
}

func healthCounter(t *testing.T, name, reason string) float64 {
	t.Helper()
	return gatheredValue(t, name, map[string]string{"plugin_name": healthFilterName, "reason": reason})
}

// TestEndpointHealth_WithoutFilterFailingEndpointKeepsItsShare reproduces the
// failure the health filter exists for: without it, an endpoint whose polls and
// responses all fail keeps receiving its share of requests.
func TestEndpointHealth_WithoutFilterFailingEndpointKeepsItsShare(t *testing.T) {
	h, failingAddr, failingID := setupHealthHarness(t, endpointHealthConfig(""))
	h.SetPodPollErrors(map[types.NamespacedName]error{failingID: errBadGateway})
	respond := func(endpoint string) string {
		if endpoint == failingAddr {
			return "502"
		}
		return "200"
	}

	hits := 0
	for range 60 {
		if routeAndRespond(t, h, respond) == failingAddr {
			hits++
		}
	}
	assert.Positive(t, hits, "the failing endpoint keeps being picked")
}

// TestEndpointHealth_PollFailuresEject stops routing to an endpoint once its
// polls have failed for pollFailureDuration, and resumes after the first
// successful poll.
func TestEndpointHealth_PollFailuresEject(t *testing.T) {
	h, failingAddr, failingID := setupHealthHarness(t, endpointHealthConfig(`
      pollSource: mock-metrics-source
      pollFailureDuration: 300ms
      responseFailureThreshold: 0`))

	ejectionsBefore := healthCounter(t, "llm_d_epp_endpoint_health_ejections_total", "poll")
	readmissionsBefore := healthCounter(t, "llm_d_epp_endpoint_health_readmissions_total", "poll")
	h.SetPodPollErrors(map[types.NamespacedName]error{failingID: errBadGateway})
	require.Eventually(t, func() bool {
		routeAndRespond(t, h, respondOK)
		return healthCounter(t, "llm_d_epp_endpoint_health_ejections_total", "poll") == ejectionsBefore+1
	}, 10*time.Second, 50*time.Millisecond, "the endpoint was not marked unhealthy")
	assert.InDelta(t, 1, gatheredValue(t, "llm_d_epp_endpoint_health_excluded_endpoints",
		map[string]string{"plugin_name": healthFilterName}), 0)

	for range 40 {
		require.NotEqual(t, failingAddr, routeAndRespond(t, h, respondOK))
	}

	h.SetPodPollErrors(nil)
	require.Eventually(t, func() bool {
		return routeAndRespond(t, h, respondOK) == failingAddr
	}, 10*time.Second, time.Millisecond, "the endpoint was not readmitted after its polls recovered")
	assert.InDelta(t, readmissionsBefore+1, healthCounter(t, "llm_d_epp_endpoint_health_readmissions_total", "poll"), 0)
}

// TestEndpointHealth_FailedResponsesEject ejects an endpoint that answers
// requests with 502 while its polls keep succeeding, and routes to it again
// once the ejection ends.
func TestEndpointHealth_FailedResponsesEject(t *testing.T) {
	const ejection = 3 * time.Second
	h, failingAddr, _ := setupHealthHarness(t, endpointHealthConfig(fmt.Sprintf(`
      pollFailureThreshold: 0
      responseFailureThreshold: 3
      ejectionDuration: %s
      maxEjectionDuration: %s`, ejection, ejection)))

	failing := true
	respond := func(endpoint string) string {
		if failing && endpoint == failingAddr {
			return "502"
		}
		return "200"
	}

	ejectionsBefore := healthCounter(t, "llm_d_epp_endpoint_health_ejections_total", "response")
	readmissionsBefore := healthCounter(t, "llm_d_epp_endpoint_health_readmissions_total", "response")
	require.Eventually(t, func() bool {
		routeAndRespond(t, h, respond)
		return healthCounter(t, "llm_d_epp_endpoint_health_ejections_total", "response") == ejectionsBefore+1
	}, 10*time.Second, time.Millisecond, "the endpoint was not ejected")
	ejectedAt := time.Now()

	for range 20 {
		require.NotEqual(t, failingAddr, routeAndRespond(t, h, respond))
	}
	require.Less(t, time.Since(ejectedAt), ejection, "the requests must run inside the ejection window")

	failing = false
	require.Eventually(t, func() bool {
		return routeAndRespond(t, h, respond) == failingAddr
	}, ejection+10*time.Second, 50*time.Millisecond, "the endpoint was not readmitted after the ejection")
	assert.InDelta(t, readmissionsBefore+1, healthCounter(t, "llm_d_epp_endpoint_health_readmissions_total", "response"), 0)
}

// TestEndpointHealth_FailsOpenWhenEveryEndpointFails keeps routing when every
// endpoint is unhealthy.
func TestEndpointHealth_FailsOpenWhenEveryEndpointFails(t *testing.T) {
	h, _, _ := setupHealthHarness(t, endpointHealthConfig(`
      pollSource: mock-metrics-source
      pollFailureDuration: 300ms
      responseFailureThreshold: 0`))

	errs := map[types.NamespacedName]error{}
	for i := range 4 {
		errs[types.NamespacedName{Namespace: h.Namespace, Name: fmt.Sprintf("pod-%d-rank-0", i)}] = errBadGateway
	}
	failOpenLabels := map[string]string{"plugin_name": healthFilterName}
	failOpensBefore := gatheredValue(t, "llm_d_epp_endpoint_health_fail_open_total", failOpenLabels)
	h.SetPodPollErrors(errs)

	require.Eventually(t, func() bool {
		routeAndRespond(t, h, respondOK)
		return gatheredValue(t, "llm_d_epp_endpoint_health_fail_open_total", failOpenLabels) > failOpensBefore
	}, 10*time.Second, 50*time.Millisecond, "the filter did not fail open")

	routed := map[string]bool{}
	for range 40 {
		routed[routeAndRespond(t, h, respondOK)] = true
	}
	assert.Greater(t, len(routed), 1, "requests are spread over the unhealthy endpoints")
}

// TestEndpointHealth_UnknownPollSourceFailsStartup rejects a pollSource that
// names no configured polling data source instead of running without poll
// health.
func TestEndpointHealth_UnknownPollSourceFailsStartup(t *testing.T) {
	opts := defaultEppServerOptions(t, "unused", endpointHealthConfig(`
      pollSource: no-such-source`))
	lis, err := fwknet.ReserveListener()
	require.NoError(t, err)
	t.Cleanup(func() { _ = lis.Close() })
	source := dlmocks.NewDataSource(fwkplugin.TypedName{Type: mockDataSourceType, Name: mockDataSourceType})

	_, _, _, err = eppRunner.NewTestRunnerSetup(t.Context(), testEnv.Config, opts, source, lis)
	require.ErrorContains(t, err, "poll-status/no-such-source")
}
