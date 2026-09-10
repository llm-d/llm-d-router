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

package server

import (
	"context"
	"crypto/tls"
	"fmt"
	"path/filepath"

	"k8s.io/component-base/cli/flag"

	tlsutil "github.com/llm-d/llm-d-router/internal/tls"
	"github.com/llm-d/llm-d-router/pkg/common"
)

// tlsProfile holds the parsed form of the TLS version and cipher suite
// configuration. Zero values leave the crypto/tls defaults in place.
type tlsProfile struct {
	minVersion   uint16
	cipherSuites []uint16
}

// parseTLSProfile resolves TLS version and cipher suite names to their
// crypto/tls values. An empty version or empty suite list is not an error: it
// selects the crypto/tls default rather than a pinned value.
func parseTLSProfile(minVersion string, cipherSuites []string) (tlsProfile, error) {
	var profile tlsProfile
	if minVersion != "" {
		version, err := flag.TLSVersion(minVersion)
		if err != nil {
			return tlsProfile{}, fmt.Errorf("server: invalid tls_min_version %q: %w", minVersion, err)
		}
		profile.minVersion = version
	}
	suites, err := flag.TLSCipherSuites(cipherSuites)
	if err != nil {
		return tlsProfile{}, fmt.Errorf("server: invalid tls_cipher_suites: %w", err)
	}
	profile.cipherSuites = suites
	return profile, nil
}

// listenerTLSConfig builds the TLS configuration for the inference listener.
// With a certificate directory the key pair is loaded from tls.crt and tls.key
// and rotation is picked up without a restart. Without it, a self-signed
// certificate is generated, which is often used for testing.
func (s *Server) listenerTLSConfig(ctx context.Context) (*tls.Config, error) {
	cfg := &tls.Config{
		MinVersion:   s.tls.minVersion,
		CipherSuites: s.tls.cipherSuites,
	}

	if s.certPath == "" {
		cert, err := tlsutil.CreateSelfSignedTLSCertificate(serverLog)
		if err != nil {
			return nil, fmt.Errorf("server: create self-signed certificate: %w", err)
		}
		cfg.Certificates = []tls.Certificate{cert}
		return cfg, nil
	}

	certFile := filepath.Join(s.certPath, "tls.crt")
	keyFile := filepath.Join(s.certPath, "tls.key")
	cert, err := tls.LoadX509KeyPair(certFile, keyFile)
	if err != nil {
		return nil, fmt.Errorf("server: load key pair from cert %q and key %q: %w", certFile, keyFile, err)
	}

	reloader, err := common.NewCertReloader(ctx, s.certPath, &cert)
	if err != nil {
		return nil, fmt.Errorf("server: start certificate reloader: %w", err)
	}
	cfg.GetCertificate = func(*tls.ClientHelloInfo) (*tls.Certificate, error) {
		return reloader.Get(), nil
	}
	return cfg, nil
}
