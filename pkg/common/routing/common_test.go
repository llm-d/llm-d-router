// Package routing contains routing constants and utilities shared between
// the EPP/Inference-Scheduler and the Routing Sidecar.
//
//revive:disable:var-naming
package routing

import "testing"

func TestStripScheme(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "http scheme",
			input:    "http://localhost:4317",
			expected: "localhost:4317",
		},
		{
			name:     "https scheme",
			input:    "https://localhost:4317",
			expected: "localhost:4317",
		},
		{
			name:     "no scheme",
			input:    "localhost:4317",
			expected: "localhost:4317",
		},
		{
			name:     "host only",
			input:    "localhost",
			expected: "localhost",
		},
		{
			name:     "http with domain",
			input:    "http://otel-collector.monitoring.svc.cluster.local:4317",
			expected: "otel-collector.monitoring.svc.cluster.local:4317",
		},
		{
			name:     "https with domain",
			input:    "https://otel-collector.monitoring.svc.cluster.local:4317",
			expected: "otel-collector.monitoring.svc.cluster.local:4317",
		},
		{
			name:     "empty string",
			input:    "",
			expected: "",
		},
		{
			name:     "ip address with http",
			input:    "http://10.0.0.1:4317",
			expected: "10.0.0.1:4317",
		},
		{
			name:     "ip address with https",
			input:    "https://10.0.0.1:4317",
			expected: "10.0.0.1:4317",
		},
		{
			name:     "ip address without scheme",
			input:    "10.0.0.1:4317",
			expected: "10.0.0.1:4317",
		},
		{
			name:     "schemeless with double slash",
			input:    "//192.168.1.1:80",
			expected: "192.168.1.1:80",
		},
		{
			name:     "uppercase scheme",
			input:    "HTTP://localhost:4317",
			expected: "localhost:4317",
		},
		{
			name:     "port only",
			input:    ":9090",
			expected: ":9090",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := StripScheme(tt.input)
			if result != tt.expected {
				t.Errorf("StripScheme(%q) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}

func TestIsConditionalDecode(t *testing.T) {
	tests := []struct {
		name    string
		headers map[string]string
		want    bool
	}{
		{"nil headers", nil, false},
		{"empty headers", map[string]string{}, false},
		{"unrelated header", map[string]string{"x-other": "v"}, false},
		{"prefer return=minimal (not if-available)", map[string]string{PreferHeader: "return=minimal"}, false},
		{"prefer if-available", map[string]string{PreferHeader: PreferIfAvailable}, true},
		{"prefer If-Available case insensitive", map[string]string{PreferHeader: "If-Available"}, true},
		{"prefer with multiple tokens including if-available", map[string]string{PreferHeader: "return=minimal, if-available"}, true},
		{"prefer if-available with parameter", map[string]string{PreferHeader: "if-available;param=v"}, true},
		{"prefer if-available with leading whitespace", map[string]string{PreferHeader: "  if-available  "}, true},
		{"prefer with similar but distinct token", map[string]string{PreferHeader: "if-available-but-different"}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsConditionalDecode(tt.headers); got != tt.want {
				t.Errorf("IsConditionalDecode(%v) = %v, want %v", tt.headers, got, tt.want)
			}
		})
	}
}
