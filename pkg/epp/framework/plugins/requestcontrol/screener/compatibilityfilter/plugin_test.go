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

package compatibilityfilter

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
)

const (
	testAttribute = "vllm.io/nixl-compat-hash"
	testHeader    = "x-compat-hash"
	testRoleLabel = "llm-d.ai/role"
)

func testScreener(t *testing.T) *Screener {
	t.Helper()
	config := Config{
		AttributeKey: testAttribute,
		HeaderName:   testHeader,
		RoleLabelKey: testRoleLabel,
		RequireRoles: []string{"prefill", "decode"},
	}
	require.NoError(t, config.Validate())
	return newScreener(
		"test-screener",
		config,
		attrmetrics.StringMetricDataKey(testAttribute),
	)
}

func testEndpoint(name, role, hash string) fwksched.Endpoint {
	attributes := fwkdl.NewAttributes()
	if hash != "" {
		attributes.Put(
			attrmetrics.StringMetricDataKey(testAttribute),
			attrmetrics.StringMetricValue(hash),
		)
	}
	return fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{
			Name: name,
			Labels: map[string]string{
				testRoleLabel: role,
			},
		},
		&fwkdl.Metrics{},
		attributes,
	)
}

func TestScreenerConsumesCompatibilityAttribute(t *testing.T) {
	screener := testScreener(t)
	dependencies := screener.Consumes()
	require.Contains(t, dependencies.Required, screener.dataKey)
	assert.IsType(
		t,
		attrmetrics.StringMetricValue(""),
		dependencies.Required[screener.dataKey],
	)
}

func TestScreenPinnedValue(t *testing.T) {
	screener := testScreener(t)
	endpoints := []fwksched.Endpoint{
		testEndpoint("a-prefill", "prefill", "a"),
		testEndpoint("a-decode", "decode", "a"),
		testEndpoint("b-prefill", "prefill", "b"),
		testEndpoint("b-decode", "decode", "b"),
	}
	request := &fwksched.InferenceRequest{
		Headers: map[string]string{testHeader: "b"},
	}

	got := screener.Screen(context.Background(), request, endpoints)
	require.Len(t, got, 2)
	assert.Equal(t, "b-prefill", got[0].GetMetadata().Name)
	assert.Equal(t, "b-decode", got[1].GetMetadata().Name)
}

func TestScreenRequiresRoleCoverage(t *testing.T) {
	screener := testScreener(t)
	endpoints := []fwksched.Endpoint{
		testEndpoint("a-prefill", "prefill", "a"),
		testEndpoint("a-decode", "decode", "a"),
		testEndpoint("b-prefill", "prefill", "b"),
	}
	request := &fwksched.InferenceRequest{
		RequestID: "request-1",
		Headers:   map[string]string{},
	}

	got := screener.Screen(context.Background(), request, endpoints)
	require.Len(t, got, 2)
	assert.Equal(t, "a-prefill", got[0].GetMetadata().Name)
	assert.Equal(t, "a-decode", got[1].GetMetadata().Name)
}

func TestResponseHeaderStampsChosenCompatibility(t *testing.T) {
	screener := testScreener(t)
	request := &fwksched.InferenceRequest{
		RequestID: "request-1",
		Headers:   map[string]string{},
	}
	endpoints := []fwksched.Endpoint{
		testEndpoint("a-prefill", "prefill", "a"),
		testEndpoint("a-decode", "decode", "a"),
	}
	require.Len(
		t,
		screener.Screen(context.Background(), request, endpoints),
		2,
	)

	response := &fwkrc.Response{Headers: map[string]string{}}
	screener.ResponseHeader(context.Background(), request, response, nil)
	assert.Equal(t, "a", response.Headers[testHeader])
	assert.NotContains(t, screener.chosenValues, request.RequestID)
}

func TestScreenFailsClosedWithoutCompatibilityData(t *testing.T) {
	screener := testScreener(t)
	request := &fwksched.InferenceRequest{Headers: map[string]string{}}
	endpoints := []fwksched.Endpoint{
		testEndpoint("prefill", "prefill", ""),
		testEndpoint("decode", "decode", ""),
	}

	assert.Empty(t, screener.Screen(context.Background(), request, endpoints))
}

func TestFactoryAcceptsSlashInAttributeKey(t *testing.T) {
	config := Config{
		AttributeKey: testAttribute,
		HeaderName:   testHeader,
	}
	raw, err := json.Marshal(config)
	require.NoError(t, err)

	plugin, err := Factory(
		"test-screener",
		fwkplugin.StrictDecoder(raw),
		nil,
	)
	require.NoError(t, err)
	screener, ok := plugin.(*Screener)
	require.True(t, ok)

	assert.Equal(
		t,
		attrmetrics.StringMetricDataKey(testAttribute),
		screener.dataKey,
	)
}
