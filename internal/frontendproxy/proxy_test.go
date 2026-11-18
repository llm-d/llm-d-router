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

package frontendproxy

import (
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpctest"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/gen/vllmpb"
)

func TestProxyLimitsKnownAndChunkedBodies(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
	}))
	defer upstream.Close()
	target, err := url.Parse(upstream.URL)
	require.NoError(t, err)
	public := httptest.NewServer(New(target, 16, false))
	defer public.Close()
	for _, known := range []bool{true, false} {
		req, err := http.NewRequestWithContext(t.Context(), "POST", public.URL, strings.NewReader(strings.Repeat("a", 17)))
		require.NoError(t, err)
		if !known {
			req.ContentLength = -1
		}
		res, err := public.Client().Do(req)
		require.NoError(t, err)
		res.Body.Close()
		require.Equal(t, http.StatusRequestEntityTooLarge, res.StatusCode)
	}
}

func TestGRPCTrailersOnly(t *testing.T) {
	upstream := grpctest.NewH2CServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		grpcframe.WriteError(w, codes.InvalidArgument, "prompt is required")
	}))
	target, err := url.Parse(upstream.URL)
	require.NoError(t, err)
	proxy := grpctest.NewH2CServer(t, New(target, 128, true))
	client := vllmpb.NewInferenceClient(grpctest.Dial(t, proxy.Listener.Addr().String()))
	_, err = client.Generate(t.Context(), &vllmpb.GenerateRequest{})
	require.Equal(t, codes.InvalidArgument, status.Code(err))
}
