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

package http

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

func TestHTTPDataSource_ProxyURL(t *testing.T) {
	var gotURL, gotHost string
	proxy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotURL = r.URL.String()
		gotHost = r.Host
		_, _ = io.WriteString(w, "ok")
	}))
	defer proxy.Close()
	proxyURL, err := url.Parse(proxy.URL)
	require.NoError(t, err)

	parser := func(r io.Reader) (string, error) {
		b, err := io.ReadAll(r)
		return string(b), err
	}
	// The endpoint address is in TEST-NET-1 and never answers: the poll only
	// succeeds if it goes through the proxy.
	ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		ID:          types.NamespacedName{Namespace: "default", Name: "ep"},
		Address:     "192.0.2.1",
		MetricsHost: "192.0.2.1:9080",
	}, nil)

	s, err := NewHTTPDataSource("http", "/metrics", TLSOptions{}, "test", "proxied", parser, WithProxyURL(proxyURL))
	require.NoError(t, err)
	got, err := s.Poll(context.Background(), ep)
	require.NoError(t, err)
	assert.Equal(t, "ok", got)
	assert.Equal(t, "http://192.0.2.1:9080/metrics", gotURL)
	assert.Equal(t, "192.0.2.1:9080", gotHost)
}

func TestHTTPDataSource_ProxyURLDoesNotChangeSharedTransport(t *testing.T) {
	parser := func(r io.Reader) (int, error) { return 0, nil }
	proxyURL := &url.URL{Scheme: "http", Host: "127.0.0.1:15081"}

	direct, err := NewHTTPDataSource("http", "/metrics", TLSOptions{}, "test", "direct", parser)
	require.NoError(t, err)
	proxied, err := NewHTTPDataSource("http", "/metrics", TLSOptions{}, "test", "proxied", parser, WithProxyURL(proxyURL))
	require.NoError(t, err)
	proxiedTLS, err := NewHTTPDataSource("https", "/metrics", TLSOptions{SkipVerify: true}, "test", "proxied-tls", parser,
		WithProxyURL(proxyURL))
	require.NoError(t, err)

	directTransport := direct.client.(*client).Transport.(*http.Transport)
	assert.Same(t, baseTransport, directTransport)
	assert.Nil(t, baseTransport.Proxy)

	req := &http.Request{URL: &url.URL{Scheme: "http", Host: "192.0.2.1:9080"}}
	for _, s := range []*HTTPDataSource[int]{proxied, proxiedTLS} {
		tr := s.client.(*client).Transport.(*http.Transport)
		assert.NotSame(t, baseTransport, tr, s.TypedName().Name)
		require.NotNil(t, tr.Proxy, s.TypedName().Name)
		got, err := tr.Proxy(req)
		require.NoError(t, err)
		assert.Equal(t, proxyURL, got, s.TypedName().Name)
	}
	assert.True(t, proxiedTLS.client.(*client).Transport.(*http.Transport).TLSClientConfig.InsecureSkipVerify)
}

func TestParseProxyURLOption(t *testing.T) {
	for _, raw := range []string{"", "http://127.0.0.1:15081", "https://proxy.example:3128"} {
		opt, err := ParseProxyURLOption(raw)
		assert.NoError(t, err, raw)
		var o options
		opt(&o)
		if raw == "" {
			assert.Nil(t, o.proxyURL)
		} else {
			assert.Equal(t, raw, o.proxyURL.String())
		}
	}
	for _, raw := range []string{"127.0.0.1:15081", "socks5://127.0.0.1:1080", "http://", "http://%zz"} {
		_, err := ParseProxyURLOption(raw)
		assert.Error(t, err, raw)
		if err != nil {
			assert.True(t, strings.Contains(err.Error(), "invalid proxy URL"), err.Error())
		}
	}
}
