/*
Copyright 2025 The llm-d Authors.

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

package http

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

// A scraped pod chooses the redirect target, so the scrape client must stop at
// the redirect response instead of fetching an address the pod picked.
func TestPollDoesNotFollowRedirects(t *testing.T) {
	var targetHits atomic.Int32
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		targetHits.Add(1)
		_, _ = w.Write([]byte("redirected\n"))
	}))
	defer target.Close()

	pod := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL+"/metrics", http.StatusFound)
	}))
	defer pod.Close()

	parsed := false
	parser := func(io.Reader) (string, error) {
		parsed = true
		return "", nil
	}
	src, err := NewHTTPDataSource("http", "/metrics", TLSOptions{}, "test-source", "test", parser)
	require.NoError(t, err)

	u, err := url.Parse(pod.URL)
	require.NoError(t, err)
	endpoint := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{MetricsHost: u.Host}, fwkdl.NewMetrics())

	_, err = src.Poll(context.Background(), endpoint)
	require.Error(t, err, "a redirect response must fail the scrape")
	require.Contains(t, err.Error(), "unexpected status code")
	require.Zero(t, targetHits.Load(), "the redirect target must not be contacted")
	require.False(t, parsed, "the redirect body must not be parsed")
}
