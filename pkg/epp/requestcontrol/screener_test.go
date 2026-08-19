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

package requestcontrol

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

type mockScreener struct {
	name   string
	screen func([]fwksched.Endpoint) []fwksched.Endpoint
}

func (f *mockScreener) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "mock-screener", Name: f.name}
}

func (f *mockScreener) Screen(
	_ context.Context,
	_ *fwksched.InferenceRequest,
	endpoints []fwksched.Endpoint,
) []fwksched.Endpoint {
	return f.screen(endpoints)
}

func TestConfigAddPluginsCollectsScreeners(t *testing.T) {
	screener := &mockScreener{name: "candidate-screener", screen: func(endpoints []fwksched.Endpoint) []fwksched.Endpoint {
		return endpoints
	}}
	config := NewConfig()

	config.AddPlugins(screener)

	require.Len(t, config.screeners, 1)
	assert.Same(t, screener, config.screeners[0])
	var _ fwkrc.Screener = screener
}

func TestRunScreenersIntersectsIndependentSubsets(t *testing.T) {
	endpoints := []fwksched.Endpoint{
		candidateEndpoint("a"),
		candidateEndpoint("b"),
		candidateEndpoint("c"),
	}
	secondInput := []fwksched.Endpoint(nil)
	config := NewConfig().WithScreeners(
		&mockScreener{name: "drop-a", screen: func(endpoints []fwksched.Endpoint) []fwksched.Endpoint {
			return endpoints[1:]
		}},
		&mockScreener{name: "keep-c", screen: func(endpoints []fwksched.Endpoint) []fwksched.Endpoint {
			secondInput = append(secondInput, endpoints...)
			return endpoints[2:]
		}},
	)
	director := &Director{requestControlPlugins: *config}

	result := director.runScreeners(context.Background(), &fwksched.InferenceRequest{}, endpoints)

	require.Len(t, secondInput, 3)
	assert.Equal(t, "a", secondInput[0].GetMetadata().Name)
	require.Len(t, result, 1)
	assert.Equal(t, "c", result[0].GetMetadata().Name)
}

func TestRunScreenersAllObserveOriginalSetAfterEmptyResult(t *testing.T) {
	endpoints := []fwksched.Endpoint{candidateEndpoint("a"), candidateEndpoint("b")}
	secondCalled := false
	config := NewConfig().WithScreeners(
		&mockScreener{name: "empty", screen: func([]fwksched.Endpoint) []fwksched.Endpoint {
			return nil
		}},
		&mockScreener{name: "observe", screen: func(got []fwksched.Endpoint) []fwksched.Endpoint {
			secondCalled = true
			require.Len(t, got, len(endpoints))
			return got
		}},
	)
	director := &Director{requestControlPlugins: *config}

	result := director.runScreeners(context.Background(), &fwksched.InferenceRequest{}, endpoints)

	assert.True(t, secondCalled)
	assert.Empty(t, result)
}

func TestRunScreenersIsolatesPluginInputs(t *testing.T) {
	endpoints := []fwksched.Endpoint{candidateEndpoint("a"), candidateEndpoint("b")}
	var secondInput []fwksched.Endpoint
	config := NewConfig().WithScreeners(
		&mockScreener{name: "mutate-input", screen: func(got []fwksched.Endpoint) []fwksched.Endpoint {
			got[0] = candidateEndpoint("replacement")
			return got
		}},
		&mockScreener{name: "observe", screen: func(got []fwksched.Endpoint) []fwksched.Endpoint {
			secondInput = append(secondInput, got...)
			return got
		}},
	)
	director := &Director{requestControlPlugins: *config}

	director.runScreeners(context.Background(), &fwksched.InferenceRequest{}, endpoints)

	require.Len(t, secondInput, 2)
	assert.Equal(t, "a", secondInput[0].GetMetadata().Name)
	assert.Equal(t, "a", endpoints[0].GetMetadata().Name)
}

func candidateEndpoint(name string) fwksched.Endpoint {
	return fwksched.NewEndpoint(&fwkdl.EndpointMetadata{
		ID:   types.NamespacedName{Namespace: "default", Name: name},
		Name: name,
	}, nil, nil)
}
