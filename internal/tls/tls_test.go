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

package tls

import (
	"crypto/tls"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestParseTLSVersion(t *testing.T) {
	tests := []struct {
		name            string
		input           string
		want            uint16
		wantErrContains string
	}{
		{name: "empty means default", input: "", want: 0},
		{name: "TLS 1.2", input: "VersionTLS12", want: tls.VersionTLS12},
		{name: "TLS 1.3", input: "VersionTLS13", want: tls.VersionTLS13},
		{
			name:            "unknown version",
			input:           "TLS1.3",
			wantErrContains: `invalid tls-min-version "TLS1.3": unknown TLS version "TLS1.3"; supported values: VersionTLS12, VersionTLS13`,
		},
		{
			name:            "TLS 1.0 below minimum",
			input:           "VersionTLS10",
			wantErrContains: `tls-min-version "VersionTLS10" is below the TLS 1.2 minimum; supported values: VersionTLS12, VersionTLS13`,
		},
		{
			name:            "TLS 1.1 below minimum",
			input:           "VersionTLS11",
			wantErrContains: `tls-min-version "VersionTLS11" is below the TLS 1.2 minimum; supported values: VersionTLS12, VersionTLS13`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseTLSVersion(tt.input)
			if tt.wantErrContains != "" {
				require.ErrorContains(t, err, tt.wantErrContains)
				require.Zero(t, got)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tt.want, got)
		})
	}
}

func TestParseCipherSuites(t *testing.T) {
	tests := []struct {
		name            string
		input           []string
		want            []uint16
		wantErrContains string
	}{
		{name: "nil means default", input: nil, want: nil},
		{name: "all blank means default", input: []string{"", "  "}, want: nil},
		{
			name:  "valid names keep order",
			input: []string{"TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384", "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256"},
			want:  []uint16{tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384, tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
		},
		{
			name:  "surrounding whitespace trimmed",
			input: []string{" TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256 "},
			want:  []uint16{tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
		},
		{
			name:  "legacy chacha alias",
			input: []string{"TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305"},
			want:  []uint16{tls.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256},
		},
		{
			name:            "insecure suite rejected",
			input:           []string{"TLS_RSA_WITH_RC4_128_SHA"},
			wantErrContains: `invalid tls-cipher-suites: insecure cipher suite "TLS_RSA_WITH_RC4_128_SHA"`,
		},
		{
			name:            "unknown suite rejected",
			input:           []string{"TLS_BOGUS"},
			wantErrContains: `invalid tls-cipher-suites: Cipher suite TLS_BOGUS not supported or doesn't exist`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseCipherSuites(tt.input)
			if tt.wantErrContains != "" {
				require.ErrorContains(t, err, tt.wantErrContains)
				require.Nil(t, got)
				return
			}
			require.NoError(t, err)
			if tt.want == nil {
				require.Nil(t, got)
				return
			}
			require.Equal(t, tt.want, got)
		})
	}
}
