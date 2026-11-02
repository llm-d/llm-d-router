/*
Copyright 2026 The Kubernetes Authors.
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
	"errors"
	"testing"

	corev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/metadata"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

// mockProcessServer implements ExternalProcessor_ProcessServer for testing.
type mockProcessServer struct {
	sentResponses []*extProcPb.ProcessingResponse
}

func (m *mockProcessServer) Send(resp *extProcPb.ProcessingResponse) error {
	m.sentResponses = append(m.sentResponses, resp)
	return nil
}

// Unused methods to satisfy the interface.
func (m *mockProcessServer) Recv() (*extProcPb.ProcessingRequest, error) {
	return nil, errors.New("sentinel error for mock process server")
}
func (m *mockProcessServer) SetHeader(metadata.MD) error  { return nil }
func (m *mockProcessServer) SendHeader(metadata.MD) error { return nil }
func (m *mockProcessServer) SetTrailer(metadata.MD)       {}
func (m *mockProcessServer) Context() context.Context     { return context.Background() }
func (m *mockProcessServer) SendMsg(any) error            { return nil }
func (m *mockProcessServer) RecvMsg(any) error            { return nil }

func TestUpdateStateAndSendIfNeeded_Evicted(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name                 string
		requestDroppedReason errcommon.RequestDroppedReason
		wantHeader           bool
	}{
		{
			name:                 "with eviction reason",
			requestDroppedReason: errcommon.RequestDroppedReasonEvicted,
			wantHeader:           true,
		},
		{
			name:                 "without eviction reason",
			requestDroppedReason: "",
			wantHeader:           false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			srv := &mockProcessServer{}
			logger := logr.Discard()

			reqCtx := &RequestContext{
				requestState:         requestEvicted,
				requestDroppedReason: tt.requestDroppedReason,
			}

			err := reqCtx.updateStateAndSendIfNeeded(srv, logger, nil)
			require.NoError(t, err)

			require.Len(t, srv.sentResponses, 1, "Should send exactly one response")
			ir := srv.sentResponses[0].GetImmediateResponse()
			require.NotNil(t, ir, "Response should be an ImmediateResponse")
			assert.Equal(t, envoyTypePb.StatusCode_TooManyRequests, ir.Status.Code)
			assert.Equal(t, `{"error":{"message":"request evicted by flow control","type":"rate_limit_error","code":429}}`, string(ir.Body))

			require.NotNil(t, ir.Headers, "Should label the body as JSON")
			gotHeaders := make(map[string]string, len(ir.Headers.SetHeaders))
			for _, header := range ir.Headers.SetHeaders {
				gotHeaders[header.Header.Key] = string(header.Header.RawValue)
			}
			wantHeaders := map[string]string{"content-type": "application/json"}
			if tt.wantHeader {
				wantHeaders[errcommon.RequestDroppedReasonHeaderKey] = string(tt.requestDroppedReason)
			}
			assert.Equal(t, wantHeaders, gotHeaders)
		})
	}
}

func TestUpdateStateAndSendIfNeeded_NotEvicted(t *testing.T) {
	t.Parallel()
	srv := &mockProcessServer{}
	logger := logr.Discard()

	// Normal state — no responses queued, nothing should be sent.
	reqCtx := &RequestContext{
		requestState: requestReceived,
	}

	err := reqCtx.updateStateAndSendIfNeeded(srv, logger, nil)
	require.NoError(t, err)
	assert.Empty(t, srv.sentResponses, "Should not send any response for normal state without queued responses")
}

func TestTerminationCause(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		state  streamRequestState
		ctxErr error
		want   fwkrh.TerminationCause
	}{
		{
			name:   "evicted outranks a cancelled context",
			state:  requestEvicted,
			ctxErr: context.Canceled,
			want:   fwkrh.TerminationCauseEvicted,
		},
		{
			name:   "a cancelled context is the client going away",
			state:  responseReceived,
			ctxErr: context.Canceled,
			want:   fwkrh.TerminationCauseClientDisconnect,
		},
		{
			name:  "anything else is an error",
			state: responseReceived,
			want:  fwkrh.TerminationCauseError,
		},
		{
			name:  "skipped response processing never observes completion",
			state: requestResponseProcessingSkipped,
			want:  fwkrh.TerminationCauseNotObserved,
		},
		{
			name:   "skipped response processing with a cancelled context",
			state:  requestResponseProcessingSkipped,
			ctxErr: context.Canceled,
			want:   fwkrh.TerminationCauseNotObserved,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			reqCtx := &RequestContext{requestState: tt.state}
			assert.Equal(t, tt.want, terminationCause(reqCtx, tt.ctxErr))
		})
	}
}

func TestGRPCStatusFromTrailers(t *testing.T) {
	tests := []struct {
		name     string
		status   string
		useValue bool
		want     string
	}{
		{
			name:   "OK",
			status: grpcframe.StatusOK,
			want:   grpcframe.StatusOK,
		},
		{
			name:   "internal error",
			status: "13",
			want:   "13",
		},
		{
			name: "missing grpc status",
			want: "",
		},
		{
			name:     "internal error from value",
			status:   "13",
			useValue: true,
			want:     "13",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			trailers := &extProcPb.HttpTrailers{
				Trailers: &corev3.HeaderMap{},
			}

			if tt.status != "" {
				header := &corev3.HeaderValue{
					Key: grpcframe.HeaderStatus,
				}
				if tt.useValue {
					header.Value = tt.status
				} else {
					header.RawValue = []byte(tt.status)
				}

				trailers.Trailers.Headers = []*corev3.HeaderValue{header}
			}

			assert.Equal(
				t,
				tt.want,
				grpcStatusFromTrailers(trailers),
			)
		})
	}
}

func TestGRPCStatusFromHeaders(t *testing.T) {
	tests := []struct {
		name   string
		status string
		want   string
	}{
		{
			name:   "trailers-only unavailable",
			status: "14",
			want:   "14",
		},
		{
			name:   "trailers-only OK",
			status: grpcframe.StatusOK,
			want:   grpcframe.StatusOK,
		},
		{
			name: "no grpc status",
			want: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			headers := &extProcPb.HttpHeaders{
				Headers: &corev3.HeaderMap{},
			}
			if tt.status != "" {
				headers.Headers.Headers = []*corev3.HeaderValue{{
					Key:      grpcframe.HeaderStatus,
					RawValue: []byte(tt.status),
				}}
			}

			assert.Equal(t, tt.want, grpcStatusFromHeaders(headers))
			assert.Equal(t, "", grpcStatusFromHeaders(nil))
		})
	}
}

// TestSetGRPCStatusError covers the classification of a non-OK grpc-status:
// the mapped HTTP status sets the request's error code the way an HTTP error
// status does, and the stream's termination is an error.
func TestSetGRPCStatusError(t *testing.T) {
	tests := []struct {
		name       string
		grpcStatus string
		presetCode string
		wantCode   string
		wantCause  fwkrh.TerminationCause
	}{
		{
			name:       "unavailable maps to a model server error",
			grpcStatus: "14",
			wantCode:   errcommon.ModelServerError,
			wantCause:  fwkrh.TerminationCauseError,
		},
		{
			name:       "invalid argument maps to a model server client error",
			grpcStatus: "3",
			wantCode:   errcommon.ModelServerClientError,
			wantCause:  fwkrh.TerminationCauseError,
		},
		{
			name:       "an already-set code is kept",
			grpcStatus: "14",
			presetCode: errcommon.ModelServerClientError,
			wantCode:   errcommon.ModelServerClientError,
			wantCause:  fwkrh.TerminationCauseError,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reqCtx := &RequestContext{responseStatusCode: tt.presetCode}
			setGRPCStatusError(reqCtx, tt.grpcStatus)
			assert.Equal(t, tt.wantCode, reqCtx.responseStatusCode)
			assert.Equal(t, tt.wantCause, reqCtx.TerminationCause)
		})
	}
}
