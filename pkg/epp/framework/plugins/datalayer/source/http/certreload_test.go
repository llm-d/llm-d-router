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
	"context"
	"crypto/tls"
	"math/big"
	"net/http"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// servedSerial returns the serial of the client cert the config would present.
func servedSerial(t *testing.T, cfg *tls.Config) *big.Int {
	t.Helper()
	cert, err := cfg.GetClientCertificate(nil)
	require.NoError(t, err)
	require.NotNil(t, cert)
	return cert.Leaf.SerialNumber
}

// tlsConfigOf returns the TLS config the source's transport uses.
func tlsConfigOf(t *testing.T, c Client) *tls.Config {
	t.Helper()
	switch tr := c.(*client).Transport.(type) {
	case *tlsReloader:
		return tr.current.Load().TLSClientConfig
	case *http.Transport:
		return tr.TLSClientConfig
	default:
		t.Fatalf("unexpected transport %T", tr)
		return nil
	}
}

func TestTLSReloader_ReloadsClientCert(t *testing.T) {
	dir := t.TempDir()
	certPath, keyPath := writeCertKeyAt(t, dir, 2), filepath.Join(dir, "key.pem")

	r, err := newTLSReloader(TLSOptions{ClientCertPath: certPath, ClientKeyPath: keyPath})
	require.NoError(t, err)
	before := servedSerial(t, r.current.Load().TLSClientConfig)

	writeCertKeyAt(t, dir, 3) // rotate in place
	r.nextCheck.Store(0)      // force the next check
	r.maybeReload(context.Background())

	after := servedSerial(t, r.current.Load().TLSClientConfig)
	assert.NotEqual(t, 0, after.Cmp(before), "rotated certificate was never served")
}

// A transient read failure (e.g. racing an atomic rename mid-rotation) must retry
// on the next scrape, not lock the throttle out for a full interval.
func TestTLSReloader_RetriesAfterReadFailure(t *testing.T) {
	dir := t.TempDir()
	certPath, keyPath := writeCertKeyAt(t, dir, 2), filepath.Join(dir, "key.pem")

	r, err := newTLSReloader(TLSOptions{ClientCertPath: certPath, ClientKeyPath: keyPath})
	require.NoError(t, err)

	require.NoError(t, os.Remove(certPath)) // read will fail this tick
	r.nextCheck.Store(0)
	r.maybeReload(context.Background())
	assert.Less(t, r.nextCheck.Load(), tlsReloadInterval.Nanoseconds(),
		"a read failure locked out retry for a full interval")

	writeCertKeyAt(t, dir, 2) // restore, so the next check succeeds and re-arms the throttle
	r.nextCheck.Store(0)
	r.maybeReload(context.Background())
	assert.GreaterOrEqual(t, r.nextCheck.Load(), tlsReloadInterval.Nanoseconds(),
		"a successful check should advance the throttle")
}
