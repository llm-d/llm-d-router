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

package server

import (
	"crypto/tls"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestApplyTLSOverrides(t *testing.T) {
	tests := []struct {
		name             string
		minVersion       uint16
		cipherSuites     []uint16
		wantMinVersion   uint16
		wantCipherSuites []uint16
	}{
		{
			name:             "both set",
			minVersion:       tls.VersionTLS13,
			cipherSuites:     []uint16{tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
			wantMinVersion:   tls.VersionTLS13,
			wantCipherSuites: []uint16{tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
		},
		{
			name:           "only min version",
			minVersion:     tls.VersionTLS12,
			wantMinVersion: tls.VersionTLS12,
		},
		{
			name:             "only cipher suites",
			cipherSuites:     []uint16{tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384},
			wantCipherSuites: []uint16{tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384},
		},
		{
			name: "neither set",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			runner := &ExtProcServerRunner{
				TLSMinVersion:   tt.minVersion,
				TLSCipherSuites: tt.cipherSuites,
			}
			cfg := &tls.Config{}
			runner.applyTLSOverrides(cfg)
			require.Equal(t, tt.wantMinVersion, cfg.MinVersion)
			require.Equal(t, tt.wantCipherSuites, cfg.CipherSuites)
		})
	}
}

func TestApplyTLSOverridesPreservesExistingConfig(t *testing.T) {
	runner := &ExtProcServerRunner{
		TLSMinVersion:   tls.VersionTLS13,
		TLSCipherSuites: []uint16{tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
	}
	cfg := &tls.Config{
		NextProtos: []string{"h2"},
	}
	runner.applyTLSOverrides(cfg)
	require.Equal(t, []string{"h2"}, cfg.NextProtos)
	require.Equal(t, uint16(tls.VersionTLS13), cfg.MinVersion)
	require.Equal(t, []uint16{tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256}, cfg.CipherSuites)
}
