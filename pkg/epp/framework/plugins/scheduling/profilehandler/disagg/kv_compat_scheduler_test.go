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

package disagg_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/go-logr/logr/testr"
	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker/maxscore"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/disagg"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/scheduling"
	"github.com/llm-d/llm-d-router/test/utils"
)

const kvCompatKeyLabel = "llm-d.ai/kv-compat-key"

func kvCompatSchedEndpoint(name, role, key string) (fwksched.Endpoint, fwkdl.Endpoint) {
	md := &fwkdl.EndpointMetadata{
		ID:      k8stypes.NamespacedName{Name: name},
		Address: name,
		Labels:  map[string]string{bylabel.RoleLabel: role, kvCompatKeyLabel: key},
	}
	metrics := &fwkdl.Metrics{UpdateTime: time.Now()}
	return fwksched.NewEndpoint(md, metrics, fwkdl.NewAttributes()), fwkdl.NewEndpoint(md, metrics)
}

// TestKVCompatSchedule runs the scheduler with the stock prefix-based decider behind
// kv-compat-pd-decider, and kv-compat-filter in the prefill profile.
func TestKVCompatSchedule(t *testing.T) {
	ctx := log.IntoContext(context.Background(), testr.New(t))
	incompatible, incompatibleDL := kvCompatSchedEndpoint("prefill-k2", bylabel.RolePrefill, "k2")
	compatible, compatibleDL := kvCompatSchedEndpoint("prefill-k1", bylabel.RolePrefill, "k1")
	decodeEp, decodeDL := kvCompatSchedEndpoint("decode-k1", bylabel.RoleDecode, "k1")

	tests := []struct {
		name        string
		pool        []fwksched.Endpoint
		poolDL      []fwkdl.Endpoint
		wantPrefill string
	}{
		{"no compatible prefill endpoint: decode-only, no error",
			[]fwksched.Endpoint{incompatible, decodeEp}, []fwkdl.Endpoint{incompatibleDL, decodeDL}, ""},
		{"compatible endpoint present: the filter removes the incompatible one",
			[]fwksched.Endpoint{incompatible, compatible, decodeEp}, []fwkdl.Endpoint{incompatibleDL, compatibleDL, decodeDL}, "prefill-k1"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stock, err := disagg.NewPrefixBasedPDDecider(disagg.PrefixBasedPDDeciderConfig{NonCachedTokens: 2})
			require.NoError(t, err)
			compat, err := disagg.NewKVCompatPDDecider(stock, disagg.KVCompatPDDeciderParameters{})
			require.NoError(t, err)
			for _, ep := range tt.poolDL {
				require.NoError(t, compat.Extract(ctx, fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: ep}))
			}
			handle := utils.NewTestHandle(ctx)
			handle.AddPlugin("compat", compat)
			filter, err := disagg.KVCompatFilterPluginFactory("compat-filter", jsonDecoder(`{"decider":"compat"}`), handle)
			require.NoError(t, err)

			prefixScorer, err := prefix.New(ctx, prefix.PrefixCacheScorerPluginType, "")
			require.NoError(t, err)
			datalayer.RegisterScopeSpecs([]fwkplugin.Plugin{prefixScorer})
			prefillProfile := scheduling.NewSchedulerProfile().
				WithFilters(bylabel.NewPrefillRole(), filter.(fwksched.Filter)).
				WithPicker(maxscore.NewMaxScorePicker(picker.DefaultMaxNumOfEndpoints))
			require.NoError(t, prefillProfile.AddPlugins(scheduling.NewWeightedScorer(prefixScorer, 50)))
			decodeProfile := scheduling.NewSchedulerProfile().
				WithFilters(bylabel.NewDecodeRole()).
				WithPicker(maxscore.NewMaxScorePicker(picker.DefaultMaxNumOfEndpoints))
			require.NoError(t, decodeProfile.AddPlugins(scheduling.NewWeightedScorer(prefixScorer, 0)))
			handler := disagg.NewDisaggProfileHandler(decode, prefill, "", compat, nil)
			scheduler := scheduling.NewSchedulerWithConfig(scheduling.NewSchedulerConfig(handler,
				map[string]fwksched.SchedulerProfile{prefill: prefillProfile, decode: decodeProfile}))

			req := &fwksched.InferenceRequest{RequestID: uuid.NewString(), TargetModel: "m", Body: completionsBody(testLongPromptForKVCompat)}
			inputTokens := len(testLongPromptForKVCompat) / averageCharactersPerToken
			for _, ep := range tt.pool {
				ep.Put(attrprefix.PrefixCacheMatchInfoDataKey, attrprefix.NewPrefixCacheMatchInfo(0, inputTokens, 1))
			}
			got, err := scheduler.Schedule(ctx, req, tt.pool)
			require.NoError(t, err)
			assert.Equal(t, "decode-k1", got.ProfileResults[decode].TargetEndpoints[0].GetMetadata().ID.Name)
			prefillRes := got.ProfileResults[prefill]
			if tt.wantPrefill == "" {
				assert.Nil(t, prefillRes)
				return
			}
			require.NotNil(t, prefillRes)
			assert.Equal(t, tt.wantPrefill, prefillRes.TargetEndpoints[0].GetMetadata().ID.Name)
		})
	}
}

const testLongPromptForKVCompat = "a long prompt that the stock decider splits at two uncached tokens"

func jsonDecoder(raw string) *json.Decoder {
	return json.NewDecoder(strings.NewReader(raw))
}
