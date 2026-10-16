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

package inflightload

import (
	"testing"

	"github.com/stretchr/testify/require"

	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
)

func TestAdmissionCostsPreserveRolesWithoutAccounting(t *testing.T) {
	for _, tc := range []struct {
		name      string
		role      string
		addOutput bool
		warm      int64
		cold      int64
	}{
		{name: "monolithic", addOutput: true, warm: 64 + UnknownOutputTokens, cold: 128 + UnknownOutputTokens},
		{name: "prefill", addOutput: true, role: bylabel.RolePrefill, warm: 64, cold: 128},
		{name: "decode", addOutput: true, role: bylabel.RoleDecode, warm: UnknownOutputTokens, cold: UnknownOutputTokens},
		{name: "combined", addOutput: true, role: bylabel.RolePrefillDecode, warm: 64 + UnknownOutputTokens, cold: 128 + UnknownOutputTokens},
		{name: "input-only", warm: 64, cold: 128},
		{name: "decode-input-only", role: bylabel.RoleDecode, warm: 64, cold: 128},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := newTestProducer(t)
			p.addEstimatedOutputTokens = tc.addOutput
			endpoint := newStubSchedulingEndpoint(tc.name)
			endpoint.GetMetadata().Labels = map[string]string{bylabel.RoleLabel: tc.role}
			endpoint.Put(p.prefixMatchInfoDK, attrprefix.NewPrefixCacheMatchInfo(1, 2, 64))
			endpoints := []fwksched.Endpoint{endpoint}
			request := makeTokenRequest("queued-"+tc.name, 128)
			require.NoError(t, p.PrepareForAdmission(t.Context(), request, endpoints))
			value, _ := endpoint.Get(p.uncachedRequestTokensDk)
			require.Equal(t, tc.warm, value.(*attrconcurrency.UncachedRequestTokens).Tokens)
			require.NoError(t, p.PrepareWithoutPrefix(t.Context(), request, endpoints))
			value, _ = endpoint.Get(p.uncachedRequestTokensDk)
			require.Equal(t, tc.cold, value.(*attrconcurrency.UncachedRequestTokens).Tokens)
			require.NoError(t, p.Produce(t.Context(), request, endpoints))
			value, _ = endpoint.Get(p.uncachedRequestTokensDk)
			require.Equal(t, tc.warm, value.(*attrconcurrency.UncachedRequestTokens).Tokens)
			require.Zero(t, p.GetTokens(endpoint.GetMetadata().ID.String()))
			require.Zero(t, p.GetRequests(endpoint.GetMetadata().ID.String()))
		})
	}
}

func TestUncachedInputTokensPartialBlockStaysWithinColdCost(t *testing.T) {
	endpoint := newStubSchedulingEndpoint("partial")
	endpoint.Put(attrprefix.PrefixCacheMatchInfoDataKey, attrprefix.NewPrefixCacheMatchInfo(0, 1, 64))
	require.Equal(t, int64(1), uncachedInputTokens(endpoint, 1, attrprefix.PrefixCacheMatchInfoDataKey))
}

func TestAdmissionPartialBlockCostWithOutputEstimate(t *testing.T) {
	for _, tc := range []struct {
		name      string
		addOutput bool
		want      int64
	}{
		{name: "input-only", want: 1},
		{name: "input-and-output", addOutput: true, want: 1 + UnknownOutputTokens},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := newTestProducer(t)
			p.addEstimatedOutputTokens = tc.addOutput
			endpoint := newStubSchedulingEndpoint("partial-output")
			endpoint.Put(p.prefixMatchInfoDK, attrprefix.NewPrefixCacheMatchInfo(0, 1, 64))
			request := makeTokenRequest("partial-output", 1)
			endpoints := []fwksched.Endpoint{endpoint}
			require.NoError(t, p.PrepareWithoutPrefix(t.Context(), request, endpoints))
			cold, _ := endpoint.Get(p.uncachedRequestTokensDk)
			require.Equal(t, tc.want, cold.(*attrconcurrency.UncachedRequestTokens).Tokens)
			require.NoError(t, p.Produce(t.Context(), request, endpoints))
			actual, _ := endpoint.Get(p.uncachedRequestTokensDk)
			require.Equal(t, tc.want, actual.(*attrconcurrency.UncachedRequestTokens).Tokens)
		})
	}
}
