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

package requestcontrol

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
	"github.com/llm-d/llm-d-router/pkg/epp/handlers"
)

func TestConfigApplyDataProducerPhases(t *testing.T) {
	a, b, c := newMockDataProducerPlugin("a"), newMockDataProducerPlugin("b"), newMockDataProducerPlugin("c")
	config := NewConfig()
	config.AddPlugins(a, b, c)
	config.ApplyDataProducerPhases(map[string]fwkrc.DataProducerPhase{
		"a": fwkrc.DataProducerPhaseRequestHandling,
		"b": fwkrc.DataProducerPhaseScheduling,
	})
	assert.Equal(t, []fwkrc.DataProducer{a}, config.requestHandlingDataProducers)
	assert.Equal(t, []fwkrc.DataProducer{b, c}, config.dataProducerPlugins)
}

// phaseRecorder records the order in which the director runs producers and
// screeners in one request.
type phaseRecorder struct {
	order []string
}

// recordingProducer is a DataProducer that records its run and the number of
// endpoints it saw, and writes mockProducedDataKey on every endpoint.
type recordingProducer struct {
	name      string
	recorder  *phaseRecorder
	requires  []fwkplugin.DataKey
	produces  fwkplugin.DataKey
	err       error
	endpoints int
}

func (p *recordingProducer) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "recording", Name: p.name}
}

func (p *recordingProducer) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{p.produces: 0}
}

func (p *recordingProducer) Consumes() fwkplugin.DataDependencies {
	deps := fwkplugin.DataDependencies{Required: map[fwkplugin.DataKey]any{}}
	for _, key := range p.requires {
		deps.Required[key] = 0
	}
	return deps
}

func (p *recordingProducer) Produce(_ context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) error {
	p.recorder.order = append(p.recorder.order, p.name)
	p.endpoints = len(endpoints)
	if p.err != nil {
		return p.err
	}
	for _, endpoint := range endpoints {
		endpoint.Put(p.produces, mockProducedDataType{value: 42})
	}
	return nil
}

func requestHandlingTestRequest(t *testing.T, id string) (*handlers.RequestContext, *handlers.Request) {
	t.Helper()
	body, err := json.Marshal(map[string]any{"model": "m", "prompt": "p"})
	require.NoError(t, err)
	request := &handlers.Request{
		ID: id,
		Headers: map[string]string{
			reqcommon.RequestIDHeaderKey: id,
			":path":                      "/v1/completions",
		},
		RawBody: body,
	}
	reqCtx := &handlers.RequestContext{Request: request, Parser: openai.NewOpenAIParser()}
	return reqCtx, request
}

func singlePodSchedulingResult() *fwksched.SchedulingResult {
	return &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{fwksched.NewEndpoint(&fwkdl.EndpointMetadata{
				Address: "192.168.1.100",
				Port:    "8000",
				ID:      types.NamespacedName{Name: "pod1", Namespace: "default"},
			}, nil, fwkdl.NewAttributes())}},
		},
	}
}

// TestDirector_RunsRequestHandlingProducersBeforeScreeners verifies that RequestHandling
// producers run on the located endpoints before screeners, that screeners and
// Scheduling producers see their data, and that Scheduling producers run on
// the screened endpoints.
func TestDirector_RunsRequestHandlingProducersBeforeScreeners(t *testing.T) {
	preKey := fwkplugin.NewDataKey("pre", "recording")
	schedKey := fwkplugin.NewDataKey("sched", "recording")
	recorder := &phaseRecorder{}
	pre := &recordingProducer{name: "pre", recorder: recorder, produces: preKey}
	sched := &recordingProducer{name: "sched", recorder: recorder, produces: schedKey, requires: []fwkplugin.DataKey{preKey}}
	sawPreData := false
	screener := &mockScreener{name: "screener", screen: func(endpoints []fwksched.Endpoint) []fwksched.Endpoint {
		recorder.order = append(recorder.order, "screener")
		_, sawPreData = endpoints[0].Get(preKey)
		return endpoints
	}}
	datalayer.RegisterScopeSpecs([]fwkplugin.Plugin{pre, sched})

	dir, ctx := newSinglePodDirector(t, singlePodSchedulingResult())
	dir.requestControlPlugins = *NewConfig().
		WithAdmissionPlugins(newMockAdmissionPlugin("admit", nil)).
		WithRequestHandlingDataProducers(pre).
		WithScreeners(screener).
		WithDataProducerPlugins(sched)

	reqCtx, request := requestHandlingTestRequest(t, "request-handling-order")
	parseResult, err := reqCtx.Parser.ParseRequest(ctx, request.RawBody, request.Headers)
	require.NoError(t, err)
	_, err = dir.HandleRequest(ctx, reqCtx, parseResult.Body)
	require.NoError(t, err)

	assert.Equal(t, []string{"pre", "screener", "sched"}, recorder.order)
	assert.True(t, sawPreData, "screener must see RequestHandling producer data")
	assert.Equal(t, 1, pre.endpoints)
	assert.Equal(t, 1, sched.endpoints)
}

// TestDirector_RequestHandlingProducerFailureSkipsDependents verifies that a failed
// RequestHandling producer leaves its keys missing for the Scheduling phase.
func TestDirector_RequestHandlingProducerFailureSkipsDependents(t *testing.T) {
	preKey := fwkplugin.NewDataKey("pre", "recording")
	schedKey := fwkplugin.NewDataKey("sched", "recording")
	recorder := &phaseRecorder{}
	pre := &recordingProducer{name: "pre", recorder: recorder, produces: preKey, err: errors.New("unsupported body")}
	sched := &recordingProducer{name: "sched", recorder: recorder, produces: schedKey, requires: []fwkplugin.DataKey{preKey}}
	datalayer.RegisterScopeSpecs([]fwkplugin.Plugin{pre, sched})

	dir, ctx := newSinglePodDirector(t, singlePodSchedulingResult())
	dir.requestControlPlugins = *NewConfig().
		WithAdmissionPlugins(newMockAdmissionPlugin("admit", nil)).
		WithRequestHandlingDataProducers(pre).
		WithDataProducerPlugins(sched)

	reqCtx, request := requestHandlingTestRequest(t, "request-handling-failure")
	parseResult, err := reqCtx.Parser.ParseRequest(ctx, request.RawBody, request.Headers)
	require.NoError(t, err)
	_, err = dir.HandleRequest(ctx, reqCtx, parseResult.Body)
	require.NoError(t, err, "a failed producer must not fail the request")

	assert.Equal(t, []string{"pre"}, recorder.order)
}

// TestDirector_ReleasesRequestHandlingProducerStateWhenScreenersEliminateAll
// verifies that per-request state a RequestHandling producer holds is released when
// screeners remove every candidate.
func TestDirector_ReleasesRequestHandlingProducerStateWhenScreenersEliminateAll(t *testing.T) {
	producer := newRequestStateProducer("state")
	screener := &mockScreener{name: "drop-all", screen: func([]fwksched.Endpoint) []fwksched.Endpoint { return nil }}
	datalayer.RegisterScopeSpecs([]fwkplugin.Plugin{producer})

	dir, ctx := newSinglePodDirector(t, singlePodSchedulingResult())
	dir.requestControlPlugins = *NewConfig().
		WithAdmissionPlugins(newMockAdmissionPlugin("admit", nil)).
		WithRequestHandlingDataProducers(producer).
		WithScreeners(screener).
		WithPreRequestPlugins(producer)

	reqCtx, request := requestHandlingTestRequest(t, "request-handling-eliminated")
	parseResult, err := reqCtx.Parser.ParseRequest(ctx, request.RawBody, request.Headers)
	require.NoError(t, err)
	_, err = dir.HandleRequest(ctx, reqCtx, parseResult.Body)
	require.Error(t, err)

	require.Empty(t, producer.heldIDs(), "per-request state must not outlive HandleRequest")
	released, abandoned := producer.counts()
	assert.Equal(t, 0, released)
	assert.Equal(t, 1, abandoned)
}
