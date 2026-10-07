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

package handlers

import (
	"context"
	"io"
	"net/http"
	"testing"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
)

type responderTestDatastore struct {
	endpoints []fwkdl.Endpoint
}

func (d *responderTestDatastore) PoolGet() (*datalayer.EndpointPool, error) {
	return &datalayer.EndpointPool{}, nil
}

func (d *responderTestDatastore) PodList(_ func(fwkdl.Endpoint) bool) []fwkdl.Endpoint {
	return d.endpoints
}

type testResponder struct {
	response *fwkrc.LocalResponse
	err      error
	calls    int
}

type responderProcessServer struct {
	mockProcessServer
	request  *extProcPb.ProcessingRequest
	received bool
}

func (s *responderProcessServer) Recv() (*extProcPb.ProcessingRequest, error) {
	if s.received {
		return nil, io.EOF
	}
	s.received = true
	return s.request, nil
}

func (r *testResponder) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "test-responder", Name: "test-responder"}
}

func (r *testResponder) Respond(_ context.Context, _ *fwkrc.RequestLine, _ []fwkdl.Endpoint) (*fwkrc.LocalResponse, error) {
	r.calls++
	return r.response, r.err
}

func requestHeaders(method string) *extProcPb.ProcessingRequest_RequestHeaders {
	return &extProcPb.ProcessingRequest_RequestHeaders{
		RequestHeaders: &extProcPb.HttpHeaders{
			Headers: &configPb.HeaderMap{Headers: []*configPb.HeaderValue{
				{Key: ":method", Value: method},
				{Key: ":path", Value: "/v1/models"},
			}},
			EndOfStream: true,
		},
	}
}

func newResponderRequestContext() *RequestContext {
	return &RequestContext{
		Request:  &Request{Headers: make(map[string]string)},
		Response: &Response{},
	}
}

func TestHandleRequestHeadersResponderAnswersGet(t *testing.T) {
	t.Parallel()

	responder := &testResponder{response: &fwkrc.LocalResponse{
		StatusCode: http.StatusOK,
		Headers:    map[string]string{"content-type": "application/json"},
		Body:       []byte(`{"object":"list"}`),
	}}
	server := NewStreamingServer(&responderTestDatastore{}, &mockDirector{}, nil, 0)
	server.SetResponders([]fwkrc.Responder{responder})
	reqCtx := newResponderRequestContext()

	err := server.HandleRequestHeaders(context.Background(), reqCtx, requestHeaders(http.MethodGet))

	require.NoError(t, err)
	assert.Equal(t, 1, responder.calls)
	assert.Equal(t, requestAnsweredLocal, reqCtx.requestState)
	require.NotNil(t, reqCtx.localResp)
	require.NotNil(t, reqCtx.localResp.GetImmediateResponse())
	assert.Equal(t, envoyTypePb.StatusCode_OK, reqCtx.localResp.GetImmediateResponse().Status.Code)
}

func TestImmediateResponseRendersStatusHeadersAndBody(t *testing.T) {
	t.Parallel()

	response := immediateResponse(&fwkrc.LocalResponse{
		StatusCode: http.StatusCreated,
		Headers: map[string]string{
			"content-type": "application/json",
			"x-test":       "value",
		},
		Body: []byte("body"),
	})

	immediate := response.GetImmediateResponse()
	require.NotNil(t, immediate)
	assert.Equal(t, envoyTypePb.StatusCode_Created, immediate.Status.Code)
	assert.Equal(t, []byte("body"), immediate.Body)
	gotHeaders := make(map[string]string, len(immediate.Headers.SetHeaders))
	for _, header := range immediate.Headers.SetHeaders {
		gotHeaders[header.Header.Key] = string(header.Header.RawValue)
	}
	assert.Equal(t, map[string]string{"content-type": "application/json", "x-test": "value"}, gotHeaders)
}

func TestImmediateResponseDefaultsZeroStatusToOK(t *testing.T) {
	t.Parallel()

	response := immediateResponse(&fwkrc.LocalResponse{})

	immediate := response.GetImmediateResponse()
	require.NotNil(t, immediate)
	assert.Equal(t, envoyTypePb.StatusCode_OK, immediate.Status.Code)
}

func TestRequestAnsweredLocalSendsImmediateResponse(t *testing.T) {
	t.Parallel()

	server := &mockProcessServer{}
	reqCtx := &RequestContext{
		requestState: requestAnsweredLocal,
		localResp: immediateResponse(&fwkrc.LocalResponse{
			StatusCode: http.StatusOK,
			Body:       []byte("local"),
		}),
	}

	require.NoError(t, reqCtx.updateStateAndSendIfNeeded(server, logr.Discard()))
	require.Len(t, server.sentResponses, 1)
	assert.Equal(t, []byte("local"), server.sentResponses[0].GetImmediateResponse().Body)
}

func TestHandleRequestHeadersResponderError(t *testing.T) {
	t.Parallel()

	responder := &testResponder{err: errcommon.Error{
		Code: errcommon.ServiceUnavailable,
		Msg:  "model data unavailable",
	}}
	server := NewStreamingServer(&responderTestDatastore{}, &mockDirector{}, nil, 0)
	server.SetResponders([]fwkrc.Responder{responder})

	err := server.HandleRequestHeaders(context.Background(), newResponderRequestContext(), requestHeaders(http.MethodGet))

	require.Error(t, err)
	assert.Equal(t, errcommon.ServiceUnavailable, errcommon.CanonicalCode(err))
	assert.Equal(t, 1, responder.calls)
}

func TestProcessResponderErrorReturns503(t *testing.T) {
	t.Parallel()

	responder := &testResponder{err: errcommon.Error{
		Code: errcommon.ServiceUnavailable,
		Msg:  "model data unavailable",
	}}
	server := NewStreamingServer(&responderTestDatastore{}, &mockDirector{}, nil, 0)
	server.SetResponders([]fwkrc.Responder{responder})
	process := &responderProcessServer{
		request: &extProcPb.ProcessingRequest{
			Request: &extProcPb.ProcessingRequest_RequestHeaders{
				RequestHeaders: requestHeaders(http.MethodGet).RequestHeaders,
			},
		},
	}

	require.NoError(t, server.Process(process))

	require.Len(t, process.sentResponses, 1)
	immediate := process.sentResponses[0].GetImmediateResponse()
	require.NotNil(t, immediate)
	assert.Equal(t, envoyTypePb.StatusCode_ServiceUnavailable, immediate.Status.Code)
	assert.Contains(t, string(immediate.Body), "model data unavailable")
}

func TestHandleRequestHeadersDecliningResponderFallsBack(t *testing.T) {
	t.Parallel()

	responder := &testResponder{}
	server := NewStreamingServer(&responderTestDatastore{}, &mockDirector{}, nil, 0)
	server.SetResponders([]fwkrc.Responder{responder})
	reqCtx := newResponderRequestContext()

	err := server.HandleRequestHeaders(context.Background(), reqCtx, requestHeaders(http.MethodGet))

	require.NoError(t, err)
	assert.Equal(t, 1, responder.calls)
	assert.NotEqual(t, requestAnsweredLocal, reqCtx.requestState)
	assert.NotNil(t, reqCtx.reqHeaderResp)
}

func TestHandleRequestHeadersDoesNotOfferPostToResponder(t *testing.T) {
	t.Parallel()

	responder := &testResponder{response: &fwkrc.LocalResponse{StatusCode: http.StatusOK}}
	server := NewStreamingServer(&responderTestDatastore{}, &mockDirector{}, nil, 0)
	server.SetResponders([]fwkrc.Responder{responder})
	reqCtx := newResponderRequestContext()

	err := server.HandleRequestHeaders(context.Background(), reqCtx, requestHeaders(http.MethodPost))

	require.NoError(t, err)
	assert.Zero(t, responder.calls)
	assert.NotEqual(t, requestAnsweredLocal, reqCtx.requestState)
	assert.NotNil(t, reqCtx.reqHeaderResp)
}
