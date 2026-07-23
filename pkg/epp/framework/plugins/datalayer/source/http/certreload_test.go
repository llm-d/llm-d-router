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

package http

import (
	"crypto/tls"
	"io"
	"math/big"
	"net/http"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// servedSerial returns the serial of the cert the config would present.
func servedSerial(t *testing.T, cfg *tls.Config) *big.Int {
	t.Helper()
	cert, err := cfg.GetClientCertificate(nil)
	require.NoError(t, err)
	require.NotNil(t, cert)
	return cert.Leaf.SerialNumber
}

// newCertSource builds an https source with a client cert in dir, returning its TLS config.
func newCertSource(t *testing.T, dir string) (*HTTPDataSource[any], *tls.Config) {
	t.Helper()
	parser := func(r io.Reader) (any, error) { return struct{}{}, nil }
	certPath, keyPath := writeCertKeyAt(t, dir, 2), filepath.Join(dir, "key.pem")
	ds, err := NewHTTPDataSource[any]("https", "/metrics",
		TLSOptions{ClientCertPath: certPath, ClientKeyPath: keyPath}, "test-type", "ds", parser)
	require.NoError(t, err)
	return ds, tlsConfigOf(t, ds.client)
}

// transportOf returns the source's current transport.
func transportOf(t *testing.T, c Client) *http.Transport {
	t.Helper()
	rt, ok := c.(*client).Transport.(*reloadingTransport)
	require.True(t, ok)
	return rt.cur.Load()
}

// tlsConfigOf returns the TLS config the source's current transport uses.
func tlsConfigOf(t *testing.T, c Client) *tls.Config {
	t.Helper()
	return transportOf(t, c).TLSClientConfig
}

func TestTLSClientConfig_ReloadServesNewCertAndCyclesConnections(t *testing.T) {
	dir := t.TempDir()
	certPath, keyPath := writeCertKeyAt(t, dir, 2), filepath.Join(dir, "key.pem")

	var cycled atomic.Int32
	cfg, refresh, err := tlsClientConfig(
		TLSOptions{ClientCertPath: certPath, ClientKeyPath: keyPath},
		func() { cycled.Add(1) })
	require.NoError(t, err)

	before := servedSerial(t, cfg)
	// RegisterCallback already fired once, so count from here.
	seedCycles := cycled.Load()

	writeCertKeyAt(t, dir, 3) // rotate in place
	refresh(t.Context())

	// certwatcher invokes the callback in its own goroutine.
	require.Eventually(t, func() bool { return servedSerial(t, cfg).Cmp(before) != 0 },
		10*time.Second, 10*time.Millisecond, "rotated certificate was never served")
	assert.Eventually(t, func() bool { return cycled.Load() > seedCycles },
		10*time.Second, 10*time.Millisecond, "idle connections were never cycled on rotation")
}

func TestRefreshCert(t *testing.T) {
	tests := []struct {
		name        string
		armThrottle bool // call refreshCert once first, so the next call is inside the window
		wantRotated bool
	}{
		{name: "rotation picked up when the throttle has expired", wantRotated: true},
		{name: "rotation ignored inside the throttle window", armThrottle: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			ds, cfg := newCertSource(t, dir)
			before := servedSerial(t, cfg)
			beforeTransport := transportOf(t, ds.client)

			if tt.armThrottle {
				ds.refreshCert(t.Context())
			}
			writeCertKeyAt(t, dir, 3) // rotate in place
			ds.refreshCert(t.Context())

			if !tt.wantRotated {
				assert.Equal(t, 0, servedSerial(t, cfg).Cmp(before), "re-read inside the throttle window")
				assert.True(t, transportOf(t, ds.client) == beforeTransport, "transport swapped without a rotation")
				return
			}
			// certwatcher invokes the callback in its own goroutine.
			require.Eventually(t, func() bool { return servedSerial(t, cfg).Cmp(before) != 0 },
				10*time.Second, 10*time.Millisecond, "rotated certificate was never served")
			// Draining idle connections is not enough, a busy one would keep the old cert.
			require.Eventually(t, func() bool { return transportOf(t, ds.client) != beforeTransport },
				10*time.Second, 10*time.Millisecond, "transport must be replaced so warm connections re-handshake")
		})
	}
}

func TestRefreshCert_NoClientCertIsNoOp(t *testing.T) {
	parser := func(r io.Reader) (any, error) { return struct{}{}, nil }
	ds, err := NewHTTPDataSource[any]("http", "/metrics", TLSOptions{}, "test-type", "ds", parser)
	require.NoError(t, err)
	assert.NotPanics(t, func() { ds.refreshCert(t.Context()) })
}
