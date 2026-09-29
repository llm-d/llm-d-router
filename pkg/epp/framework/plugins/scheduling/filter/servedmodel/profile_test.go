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

package servedmodel_test

import (
	"encoding/base64"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmodels "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/models"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/servedmodel"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/sessionaffinity"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker/maxscore"
	"github.com/llm-d/llm-d-router/pkg/epp/scheduling"
)

func TestServedModelFilterPlacementInProfile(t *testing.T) {
	servedModel, err := servedmodel.ServedModelFilterFactory("", nil, nil)
	require.NoError(t, err)
	filter := servedModel.(fwksched.Filter)
	sessionAffinity := sessionaffinity.NewSessionAffinity("session-affinity", "", "default")
	sessionTo := func(endpoint string) map[string]string {
		return map[string]string{"x-session-token": base64.StdEncoding.EncodeToString([]byte("ns/" + endpoint))}
	}

	tests := []struct {
		name      string
		filters   []fwksched.Filter
		endpoints []fwksched.Endpoint
		headers   map[string]string
		want      string
	}{
		{
			name:    "after the role filter, decode picks the decode endpoint with the adapter",
			filters: []fwksched.Filter{bylabel.NewDecodeRole(), filter},
			endpoints: []fwksched.Endpoint{
				endpoint("prefill-0", bylabel.RolePrefill, "llama", "sql-v3"),
				endpoint("decode-0", bylabel.RoleDecode, "llama"),
				endpoint("decode-1", bylabel.RoleDecode, "llama", "sql-v3"),
			},
			want: "decode-1",
		},
		{
			name:    "after the role filter, a prefill endpoint listing the adapter does not hide decode endpoints without a list",
			filters: []fwksched.Filter{bylabel.NewDecodeRole(), filter},
			endpoints: []fwksched.Endpoint{
				endpoint("prefill-0", bylabel.RolePrefill, "llama", "sql-v3"),
				endpoint("decode-0", bylabel.RoleDecode),
			},
			want: "decode-0",
		},
		{
			name:    "before session affinity, a session bound to an endpoint without the adapter moves",
			filters: []fwksched.Filter{filter, sessionAffinity},
			endpoints: []fwksched.Endpoint{
				endpoint("pod-a", "", "llama"),
				endpoint("pod-b", "", "llama", "sql-v3"),
			},
			headers: sessionTo("pod-a"),
			want:    "pod-b",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			plugins := make([]fwkplugin.Plugin, 0, len(test.filters))
			for _, f := range test.filters {
				plugins = append(plugins, f)
			}
			datalayer.RegisterScopeSpecs(plugins)
			profile := scheduling.NewSchedulerProfile().WithFilters(test.filters...).WithPicker(maxscore.NewMaxScorePicker(1))

			request := &fwksched.InferenceRequest{RequestID: "r", TargetModel: "sql-v3", Headers: test.headers}
			result, err := profile.Run(t.Context(), request, test.endpoints)
			require.NoError(t, err)
			require.Len(t, result.TargetEndpoints, 1)
			require.Equal(t, test.want, result.TargetEndpoints[0].GetMetadata().Name)
		})
	}
}

// endpoint lists base first and adapters after it, as vLLM does.
func endpoint(name, role string, models ...string) fwksched.Endpoint {
	var attrs fwkdl.AttributeMap
	if len(models) > 0 {
		collection := attrmodels.ModelDataCollection{{ID: models[0]}}
		for _, adapter := range models[1:] {
			collection = append(collection, attrmodels.ModelData{ID: adapter, Parent: models[0]})
		}
		attrs = fwkdl.NewAttributes()
		attrs.Put(attrmodels.ModelsAttributeKey, collection)
	}
	labels := map[string]string{}
	if role != "" {
		labels[bylabel.RoleLabel] = role
	}
	return fwksched.NewEndpoint(&fwkdl.EndpointMetadata{
		ID:     types.NamespacedName{Namespace: "ns", Name: name},
		Name:   name,
		Labels: labels,
	}, &fwkdl.Metrics{}, attrs)
}
