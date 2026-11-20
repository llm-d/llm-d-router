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

package coldsession

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/sessionstate"
)

func newTestFilter(t *testing.T, params string) *ColdSessionFilter {
	t.Helper()
	var decoder *json.Decoder
	if params != "" {
		decoder = json.NewDecoder(strings.NewReader(params))
	}
	plugin, err := Factory("test", decoder, nil)
	require.NoError(t, err)
	filter, ok := plugin.(*ColdSessionFilter)
	require.True(t, ok)
	return filter
}

func endpoint(name string) fwksched.Endpoint {
	return fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: fwkdl.ID(k8stypes.NamespacedName{Name: name})},
		fwkdl.NewMetrics(),
		fwkdl.NewAttributes(),
	)
}

func requestWithSession(requestID string, state sessionstate.SessionState) *fwksched.InferenceRequest {
	request := &fwksched.InferenceRequest{RequestID: requestID}
	request.PutAttribute(sessionstate.SessionStateDataKey.WithNonEmptyProducerName(sessionstate.SessionStateProducerType), state)
	return request
}

func resultWith(endpoint fwksched.Endpoint) *fwksched.SchedulingResult {
	return &fwksched.SchedulingResult{
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"prefill": {TargetEndpoints: []fwksched.Endpoint{endpoint}},
		},
		PrimaryProfileName: "prefill",
	}
}

func TestWarmAndUnknownSessionsAreNotFiltered(t *testing.T) {
	filter := newTestFilter(t, "")
	a, b := endpoint("prefill-a"), endpoint("prefill-b")

	assert.Equal(t, []fwksched.Endpoint{a, b}, filter.Filter(context.Background(), requestWithSession("warm", sessionstate.SessionState{TurnsTaken: 2}), []fwksched.Endpoint{a, b}))
	assert.Equal(t, []fwksched.Endpoint{a, b}, filter.Filter(context.Background(), &fwksched.InferenceRequest{RequestID: "unknown"}, []fwksched.Endpoint{a, b}))
}

func TestColdSessionAvoidsAnEndpointAlreadyBusyWithAColdPrefill(t *testing.T) {
	filter := newTestFilter(t, "")
	a, b := endpoint("prefill-a"), endpoint("prefill-b")

	require.NoError(t, filter.PreRequest(context.Background(), requestWithSession("cold-1", sessionstate.SessionState{}), resultWith(a)))

	got := filter.Filter(context.Background(), requestWithSession("cold-2", sessionstate.SessionState{}), []fwksched.Endpoint{a, b})
	assert.Equal(t, []fwksched.Endpoint{b}, got)

	filter.ResponseBody(context.Background(), requestWithSession("cold-1", sessionstate.SessionState{}), &fwkrc.Response{EndOfStream: true}, nil)
	got = filter.Filter(context.Background(), requestWithSession("cold-3", sessionstate.SessionState{}), []fwksched.Endpoint{a, b})
	assert.Equal(t, []fwksched.Endpoint{a, b}, got)
}

func TestColdSessionFailsOpenWhenEveryEndpointIsBusy(t *testing.T) {
	filter := newTestFilter(t, "")
	a, b := endpoint("prefill-a"), endpoint("prefill-b")
	require.NoError(t, filter.PreRequest(context.Background(), requestWithSession("cold-1", sessionstate.SessionState{}), resultWith(a)))
	require.NoError(t, filter.PreRequest(context.Background(), requestWithSession("cold-2", sessionstate.SessionState{}), resultWith(b)))

	got := filter.Filter(context.Background(), requestWithSession("cold-3", sessionstate.SessionState{}), []fwksched.Endpoint{a, b})
	assert.Equal(t, []fwksched.Endpoint{a, b}, got)
}

func TestPreRequestIsIdempotentAndTTLSweepsAbandonedEntries(t *testing.T) {
	filter := newTestFilter(t, `{"requestTtlSeconds":1}`)
	now := time.Unix(1000, 0)
	filter.now = func() time.Time { return now }
	a := endpoint("prefill-a")

	request := requestWithSession("cold-1", sessionstate.SessionState{})
	require.NoError(t, filter.PreRequest(context.Background(), request, resultWith(a)))
	require.NoError(t, filter.PreRequest(context.Background(), request, resultWith(a)))
	assert.Equal(t, 1, filter.counts[a.GetMetadata().ID])

	now = now.Add(2 * time.Second)
	assert.Equal(t, []fwksched.Endpoint{a}, filter.Filter(context.Background(), requestWithSession("cold-2", sessionstate.SessionState{}), []fwksched.Endpoint{a}))
	assert.Empty(t, filter.counts)
}

func TestValidation(t *testing.T) {
	_, err := Factory("test", json.NewDecoder(strings.NewReader(`{"maxConcurrentColdPrefills":0}`)), nil)
	require.NoError(t, err)
	_, err = Factory("test", json.NewDecoder(strings.NewReader(`{"maxConcurrentColdPrefills":-1}`)), nil)
	require.ErrorContains(t, err, "maxConcurrentColdPrefills")
	_, err = Factory("test", json.NewDecoder(strings.NewReader(`{"requestTtlSeconds":0}`)), nil)
	require.NoError(t, err)
	_, err = Factory("test", json.NewDecoder(strings.NewReader(`{"requestTtlSeconds":-1}`)), nil)
	require.ErrorContains(t, err, "requestTtlSeconds")
}

func TestTypedName(t *testing.T) {
	filter := newTestFilter(t, "")
	assert.Equal(t, fwkplugin.TypedName{Type: ColdSessionFilterType, Name: "test"}, filter.TypedName())
}
