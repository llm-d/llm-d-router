/*
Copyright 2026 The Kubernetes Authors.

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

package inflightload

import (
	"encoding/json"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/utils/ptr"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	rh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

func streamProducer(t testing.TB) *InFlightLoadProducer {
	t.Helper()
	p, err := InFlightLoadProducerFactory("stream-growth", json.NewDecoder(strings.NewReader(
		`{"trackStreamingOutputTokens":true,"outputTokenHeadroom":2048}`)), testutils.NewTestHandle(t.Context()))
	require.NoError(t, err)
	return p.(*InFlightLoadProducer)
}

func TestStreamingTokenLifecycle(t *testing.T) {
	for _, end := range []string{"eos", "cancel", "racing-cancel"} {
		t.Run(end, func(t *testing.T) {
			p := streamProducer(t)
			prefill, decode := newStubSchedulingEndpoint("p"), newStubSchedulingEndpoint("d")
			prefill.GetMetadata().Labels = map[string]string{bylabel.RoleLabel: "prefill"}
			decode.GetMetadata().Labels = map[string]string{bylabel.RoleLabel: "decode"}
			for _, e := range []fwksched.Endpoint{prefill, decode} {
				e.Put(p.prefixMatchInfoDK, attrprefix.NewPrefixCacheMatchInfo(700, 735, 32))
			}
			req := makeTokenRequest(end, 23498)
			req.Body.MaxOutputTokens = ptr.To(int64(65536))
			req.SchedulingResult = &fwksched.SchedulingResult{ProfileResults: map[string]*fwksched.ProfileRunResult{
				"prefill": {TargetEndpoints: []fwksched.Endpoint{prefill}},
				"decode":  {TargetEndpoints: []fwksched.Endpoint{decode}},
			}}
			require.NoError(t, p.PreRequest(t.Context(), req, req.SchedulingResult))
			pid, did := prefill.GetMetadata().ID.String(), decode.GetMetadata().ID.String()
			require.Equal(t, int64(1120), p.GetTokens(pid))
			require.Equal(t, int64(25546), p.GetTokens(did))
			p.ResponseBody(t.Context(), req, &requestcontrol.Response{StartOfStream: true}, nil)
			require.Zero(t, p.GetTokens(pid))
			require.Equal(t, int64(25546), p.GetTokens(did))
			for _, n := range []int{1, 25, 25, 12, 0, 59903} {
				p.ResponseBody(t.Context(), req, &requestcontrol.Response{Usage: rh.Usage{CompletionTokens: n}}, nil)
			}
			require.Equal(t, int64(85449), p.GetTokens(did))
			switch end {
			case "racing-cancel":
				var wg sync.WaitGroup
				for i := range 50 {
					wg.Go(func() {
						p.ResponseBody(t.Context(), req, &requestcontrol.Response{Usage: rh.Usage{CompletionTokens: 60000 + i}}, nil)
					})
				}
				wg.Go(func() { p.PluginState.Delete(req.RequestID) })
				wg.Wait()
			case "cancel":
				p.PluginState.Delete(req.RequestID)
			default:
				p.ResponseBody(t.Context(), req, &requestcontrol.Response{EndOfStream: true}, nil)
			}
			p.ResponseBody(t.Context(), req, &requestcontrol.Response{Usage: rh.Usage{CompletionTokens: 65536}}, nil)
			p.PluginState.Delete(req.RequestID)
			require.Zero(t, p.GetTokens(did))
			require.Zero(t, p.GetRequests(did))
		})
	}
}

func TestStreamingTokenConfig(t *testing.T) {
	for _, raw := range []string{
		`{"trackStreamingOutputTokens":true}`,
		`{"trackStreamingOutputTokens":true,"outputTokenHeadroom":-1}`,
		`{"trackStreamingOutputTokens":true,"outputTokenHeadroom":2048,"addEstimatedOutputTokens":true}`,
		`{"trackStreamingOutputTokens":true,"outputTokenHeadroom":2048,"maxEstimatedOutputTokens":32768}`,
		`{"outputTokenHeadroom":2048}`,
	} {
		_, err := InFlightLoadProducerFactory("stream", json.NewDecoder(strings.NewReader(raw)), testutils.NewTestHandle(t.Context()))
		require.Error(t, err, raw)
	}
}

func TestStreamingChargeByEndpointRole(t *testing.T) {
	p := streamProducer(t)
	req := makeTokenRequest("roles", 1000)
	for _, role := range []string{"prefill", "encode-prefill", "decode", "prefill-decode", ""} {
		t.Run(role, func(t *testing.T) {
			ep := newStubSchedulingEndpoint("worker")
			ep.GetMetadata().Labels = map[string]string{bylabel.RoleLabel: role}
			ep.Put(p.prefixMatchInfoDK, attrprefix.NewPrefixCacheMatchInfo(20, 30, 32))
			want := int64(3048)
			if role == bylabel.RolePrefill || role == bylabel.RoleEncodePrefill {
				want = 360
			}
			require.Equal(t, want, p.estimateRequestTokens(ep, req, 1000))
		})
	}
}

func TestStreamingUsageRequestedOnOpenAIOnly(t *testing.T) {
	for _, protocol := range []string{"openai", "anthropic", "non-streaming"} {
		t.Run(protocol, func(t *testing.T) {
			p := streamProducer(t)
			req := makeTokenRequest(protocol, 100)
			req.Body.Stream = protocol != "non-streaming"
			payload := rh.PayloadMap{"stream_options": map[string]any{"include_usage": false, "other_option": "preserved"}}
			req.Body.Payload = payload
			if protocol == "anthropic" {
				req.Body.Messages = &rh.MessagesRequest{}
			} else {
				req.Body.ChatCompletions = &rh.ChatCompletionsRequest{}
			}
			e := newStubSchedulingEndpoint("decode")
			result := &fwksched.SchedulingResult{ProfileResults: map[string]*fwksched.ProfileRunResult{"decode": {TargetEndpoints: []fwksched.Endpoint{e}}}}
			require.NoError(t, p.PreRequest(t.Context(), req, result))
			opts := payload["stream_options"].(map[string]any)
			require.Equal(t, protocol == "openai", opts["include_usage"])
			require.Equal(t, "preserved", opts["other_option"])
			if protocol == "openai" {
				require.Equal(t, true, opts["continuous_usage_stats"])
			} else {
				require.NotContains(t, opts, "continuous_usage_stats")
			}
			p.PluginState.Delete(req.RequestID)
		})
	}
}
