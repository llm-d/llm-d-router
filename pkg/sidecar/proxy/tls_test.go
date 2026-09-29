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

package proxy

import (
	"context"
	"crypto/tls"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/stretchr/testify/require"
)

// serveTLS starts a secure-serving proxy for cfg and returns its address.
func serveTLS(t *testing.T, cfg Config) string {
	t.Helper()

	backend := httptest.NewServer(nil)
	t.Cleanup(backend.Close)
	targetURL, err := url.Parse(backend.URL)
	require.NoError(t, err)

	cfg.Port = "0"
	cfg.DecoderURL = targetURL
	cfg.SecureServing = true

	p := NewProxy(cfg)
	p.allowlistValidator = &AllowlistValidator{enabled: false}

	ctx, cancel := context.WithCancel(context.Background())
	errCh := make(chan error, 1)
	go func() { errCh <- p.Start(ctx) }()

	t.Cleanup(func() {
		cancel()
		<-errCh
	})

	<-p.readyCh
	return p.addr.String()
}

func TestStartHTTP_TLSMinVersionRejectsOlderClient(t *testing.T) {
	addr := serveTLS(t, Config{TLSMinVersion: tls.VersionTLS13})

	_, err := tls.Dial("tcp", addr, &tls.Config{
		InsecureSkipVerify: true, //nolint:gosec // self-signed cert under test
		MaxVersion:         tls.VersionTLS12,
	})
	require.Error(t, err, "TLS 1.2 client must be rejected when the minimum is TLS 1.3")
}

func TestStartHTTP_TLSMinVersionDowngradeIgnored(t *testing.T) {
	addr := serveTLS(t, Config{TLSMinVersion: tls.VersionTLS10})

	_, err := tls.Dial("tcp", addr, &tls.Config{
		InsecureSkipVerify: true, //nolint:gosec // self-signed cert under test
		MaxVersion:         tls.VersionTLS10,
	})
	require.Error(t, err, "TLS 1.0 client must still be rejected when TLSMinVersion requests a downgrade")
}
