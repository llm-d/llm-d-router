/*
Copyright 2025 The Kubernetes Authors.

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

package tls

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"math/big"
	"strings"
	"time"

	"github.com/go-logr/logr"
	cliflag "k8s.io/component-base/cli/flag"
)

// ParseTLSVersion resolves a crypto/tls version name. Empty returns 0.
// Versions below TLS 1.2 are rejected.
func ParseTLSVersion(s string) (uint16, error) {
	if s == "" {
		return 0, nil
	}
	v, err := cliflag.TLSVersion(s)
	if err != nil {
		return 0, fmt.Errorf("invalid tls-min-version %q: unknown TLS version %q; supported values: VersionTLS12, VersionTLS13", s, s)
	}
	if v < tls.VersionTLS12 {
		return 0, fmt.Errorf("tls-min-version %q is below the TLS 1.2 minimum; supported values: VersionTLS12, VersionTLS13", s)
	}
	return v, nil
}

// ParseCipherSuites resolves crypto/tls cipher suite names, skipping blank
// entries. A list with no names returns nil so tls.Config keeps its defaults.
// Insecure suites are rejected.
func ParseCipherSuites(names []string) ([]uint16, error) {
	insecure := cliflag.InsecureTLSCiphers()
	trimmed := make([]string, 0, len(names))
	for _, name := range names {
		name = strings.TrimSpace(name)
		if name == "" {
			continue
		}
		if _, ok := insecure[name]; ok {
			return nil, fmt.Errorf("invalid tls-cipher-suites: insecure cipher suite %q", name)
		}
		trimmed = append(trimmed, name)
	}
	suites, err := cliflag.TLSCipherSuites(trimmed)
	if err != nil {
		return nil, fmt.Errorf("invalid tls-cipher-suites: %w", err)
	}
	return suites, nil
}

// CreateSelfSignedTLSCertificate creates a self-signed cert the server can use to serve TLS.
// The certificate carries no subject alternative names, so clients cannot verify it. The log
// line is the only signal a caller gets.
func CreateSelfSignedTLSCertificate(logger logr.Logger) (tls.Certificate, error) {
	logger.Info("creating self-signed TLS certificate")

	serialNumberLimit := new(big.Int).Lsh(big.NewInt(1), 128)
	serialNumber, err := rand.Int(rand.Reader, serialNumberLimit)
	if err != nil {
		return tls.Certificate{}, fmt.Errorf("error creating serial number: %v", err)
	}
	now := time.Now()
	notBefore := now.UTC()
	template := x509.Certificate{
		SerialNumber: serialNumber,
		Subject: pkix.Name{
			Organization: []string{"llm-d"},
		},
		NotBefore:             notBefore,
		NotAfter:              now.Add(time.Hour * 24 * 365 * 10).UTC(), // 10 years
		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
	}

	priv, err := rsa.GenerateKey(rand.Reader, 4096)
	if err != nil {
		return tls.Certificate{}, fmt.Errorf("error generating key: %v", err)
	}

	derBytes, err := x509.CreateCertificate(rand.Reader, &template, &template, &priv.PublicKey, priv)
	if err != nil {
		return tls.Certificate{}, fmt.Errorf("error creating certificate: %v", err)
	}

	certBytes := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: derBytes})

	privBytes, err := x509.MarshalPKCS8PrivateKey(priv)
	if err != nil {
		return tls.Certificate{}, fmt.Errorf("error marshalling private key: %v", err)
	}
	keyBytes := pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: privBytes})

	return tls.X509KeyPair(certBytes, keyBytes)
}
