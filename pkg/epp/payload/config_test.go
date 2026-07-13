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

package payload

import (
	"testing"

	"github.com/go-logr/logr"
)

func TestLoadConfigFromEnv(t *testing.T) {
	tests := []struct {
		name string
		env  map[string]string
		want Config
	}{
		{
			name: "defaults",
			env:  map[string]string{},
			want: Config{Enabled: false, Backend: BackendNoop, InlineSizeThresholdBytes: DefaultInlineSizeThresholdBytes},
		},
		{
			name: "enabled inline with threshold",
			env: map[string]string{
				EnvEnabled:         "true",
				EnvBackend:         "inline",
				EnvInlineThreshold: "1024",
			},
			want: Config{Enabled: true, Backend: BackendInline, InlineSizeThresholdBytes: 1024},
		},
		{
			name: "invalid enabled flag stays disabled",
			env:  map[string]string{EnvEnabled: "yep"},
			want: Config{Enabled: false, Backend: BackendNoop, InlineSizeThresholdBytes: DefaultInlineSizeThresholdBytes},
		},
		{
			name: "phase 2 backend falls back to noop",
			env:  map[string]string{EnvEnabled: "true", EnvBackend: "gcs"},
			want: Config{Enabled: true, Backend: BackendNoop, InlineSizeThresholdBytes: DefaultInlineSizeThresholdBytes},
		},
		{
			name: "unknown backend falls back to noop",
			env:  map[string]string{EnvEnabled: "true", EnvBackend: "carrier-pigeon"},
			want: Config{Enabled: true, Backend: BackendNoop, InlineSizeThresholdBytes: DefaultInlineSizeThresholdBytes},
		},
		{
			name: "invalid threshold uses default",
			env:  map[string]string{EnvBackend: "inline", EnvInlineThreshold: "-1"},
			want: Config{Enabled: false, Backend: BackendInline, InlineSizeThresholdBytes: DefaultInlineSizeThresholdBytes},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for k, v := range tt.env {
				t.Setenv(k, v)
			}
			got := LoadConfigFromEnv(logr.Discard())
			if got != tt.want {
				t.Errorf("LoadConfigFromEnv() = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestNewCapturerGating(t *testing.T) {
	if c := NewCapturer(Config{Enabled: false, Backend: BackendInline, InlineSizeThresholdBytes: 4096}, logr.Discard()); c != nil {
		t.Error("NewCapturer with capture disabled should return nil")
	}
	if c := NewCapturer(Config{Enabled: true, Backend: BackendNoop, InlineSizeThresholdBytes: 4096}, logr.Discard()); c != nil {
		t.Error("NewCapturer with noop backend should return nil (secondary kill switch)")
	}
	if c := NewCapturer(Config{Enabled: true, Backend: BackendInline, InlineSizeThresholdBytes: 4096}, logr.Discard()); c == nil {
		t.Error("NewCapturer with inline backend enabled should return a capturer")
	}
}
