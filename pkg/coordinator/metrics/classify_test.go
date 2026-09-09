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

package metrics

import (
	"errors"
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
)

// Test doubles: a sentinel error and an upstream-carrying error type that
// stand in for the pipeline package's ErrBadRequest and UpstreamError.
// Keeping them local avoids the metrics package importing pipeline.
var errBadReq = errors.New("test: bad request")

type upstreamStub struct {
	status int
}

func (u *upstreamStub) Error() string { return fmt.Sprintf("upstream %d", u.status) }

var stubOpts = ClassifyOptions{
	BadRequest: errBadReq,
	IsUpstream: func(err error) (int, bool) {
		var u *upstreamStub
		if errors.As(err, &u) {
			return u.status, true
		}
		return 0, false
	},
}

func TestClassifyErrorCode_BadRequest(t *testing.T) {
	require.Equal(t, ErrorCodeBadRequest, ClassifyErrorCode(errBadReq, stubOpts))
	// Wrapped bad-request errors still classify as bad_request.
	wrapped := fmt.Errorf("step: %w", errBadReq)
	require.Equal(t, ErrorCodeBadRequest, ClassifyErrorCode(wrapped, stubOpts))
}

func TestClassifyErrorCode_Upstream4xxBand(t *testing.T) {
	for _, status := range []int{400, 404, 422, 499} {
		require.Equal(t, ErrorCodeUpstream4xx,
			ClassifyErrorCode(&upstreamStub{status: status}, stubOpts),
			"status %d", status)
	}
}

func TestClassifyErrorCode_Upstream5xxBand(t *testing.T) {
	for _, status := range []int{500, 502, 503, 504, 599, 999} {
		require.Equal(t, ErrorCodeUpstream5xx,
			ClassifyErrorCode(&upstreamStub{status: status}, stubOpts),
			"status %d", status)
	}
}

func TestClassifyErrorCode_UpstreamTransportFailure(t *testing.T) {
	// An upstream error with StatusCode 0 means the round trip failed before
	// headers arrived (connection refused, timeout, TCP reset). Route it to
	// the transport bucket, not internal.
	require.Equal(t, ErrorCodeUpstreamTransport,
		ClassifyErrorCode(&upstreamStub{status: 0}, stubOpts))
}

func TestClassifyErrorCode_UpstreamSub4xxFallsThroughToInternal(t *testing.T) {
	// A synthetic 3xx or 2xx wrapped as an upstream error is not a client
	// fault under the current banding and is not a 5xx either; it falls to
	// internal rather than being silently forced into a 4xx bucket.
	for _, status := range []int{200, 301, 399} {
		require.Equal(t, ErrorCodeInternal,
			ClassifyErrorCode(&upstreamStub{status: status}, stubOpts),
			"status %d", status)
	}
}

func TestClassifyErrorCode_UnrecognizedIsInternal(t *testing.T) {
	require.Equal(t, ErrorCodeInternal,
		ClassifyErrorCode(errors.New("some other error"), stubOpts))
}

func TestClassifyErrorCode_NilOptionsSafe(t *testing.T) {
	// A caller who forgets to wire options gets "internal" for every error
	// rather than a nil-pointer panic.
	require.Equal(t, ErrorCodeInternal, ClassifyErrorCode(errors.New("boom"), ClassifyOptions{}))
}
