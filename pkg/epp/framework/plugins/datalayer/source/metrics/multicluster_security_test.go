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

package metrics

import (
	"context"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	sourcehttp "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/http"
)

// countingReader serves n whitespace bytes and records how many were read.
type countingReader struct {
	n    int
	read int
}

func (c *countingReader) Read(p []byte) (int, error) {
	if c.n <= 0 {
		return 0, io.EOF
	}
	k := min(len(p), c.n)
	for i := 0; i < k; i++ {
		p[i] = ' '
	}
	c.n -= k
	c.read += k
	return k, nil
}

// A response larger than the cap is read only up to maxResponseBytes.
func TestParseBoundedMetricsCapsResponse(t *testing.T) {
	cr := &countingReader{n: maxResponseBytes + 4096}
	_, err := parseBoundedMetrics(cr)
	require.NoError(t, err)
	require.LessOrEqual(t, cr.read, maxResponseBytes, "must not read past the cap")
}

type noopExtractor struct{}

func (noopExtractor) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "noop", Name: "noop"}
}

func (noopExtractor) Extract(context.Context, fwkdl.PollInput[PrometheusMetricMap]) error {
	return nil
}

func newSource(t *testing.T, params string) *sourcehttp.HTTPDataSource[PrometheusMetricMap] {
	t.Helper()
	var dec *json.Decoder
	if params != "" {
		dec = json.NewDecoder(strings.NewReader(params))
	}
	p, err := MultiClusterMetricsDataSourceFactory("mc", dec, nil)
	require.NoError(t, err)
	src := p.(*sourcehttp.HTTPDataSource[PrometheusMetricMap])
	require.NoError(t, src.AppendExtractor(noopExtractor{}))
	return src
}

// The cross-cluster source verifies the peer certificate by default (rejecting a
// self-signed server), and honors insecureSkipVerify to opt out.
func TestMultiClusterMetricsSourceVerifiesByDefault(t *testing.T) {
	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("# no metrics\n"))
	}))
	defer srv.Close()
	u, err := url.Parse(srv.URL)
	require.NoError(t, err)

	endpoint := func() fwkdl.Endpoint {
		return fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{MetricsHost: u.Host}, fwkdl.NewMetrics())
	}

	// Default: https + verify, so the self-signed cert is rejected.
	require.Error(t, newSource(t, "").Dispatch(context.Background(), endpoint()),
		"self-signed peer must be rejected by default")

	// insecureSkipVerify opts out.
	require.NoError(t, newSource(t, `{"insecureSkipVerify":true}`).Dispatch(context.Background(), endpoint()))
}

// A configured caCertPath flows through to the client, so a peer presenting a cert
// signed by that CA is accepted while verification stays on.
func TestMultiClusterMetricsSourceCACertPath(t *testing.T) {
	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("# no metrics\n"))
	}))
	defer srv.Close()
	u, err := url.Parse(srv.URL)
	require.NoError(t, err)

	caPath := filepath.Join(t.TempDir(), "ca.pem")
	require.NoError(t, os.WriteFile(caPath,
		pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: srv.Certificate().Raw}), 0o600))

	endpoint := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{MetricsHost: u.Host}, fwkdl.NewMetrics())
	src := newSource(t, fmt.Sprintf(`{"caCertPath":%q}`, caPath))
	require.NoError(t, src.Dispatch(context.Background(), endpoint),
		"peer cert signed by the configured CA must be accepted with verification on")
}
