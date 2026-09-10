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

package steps

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
)

func TestParamInt(t *testing.T) {
	cases := []struct {
		name    string
		val     any
		want    int
		wantOK  bool
		wantErr bool
	}{
		{name: "absent", val: nil, wantOK: false},
		{name: "int", val: 8192, want: 8192, wantOK: true},
		{name: "int64", val: int64(8192), want: 8192, wantOK: true},
		{name: "float-integral", val: 8192.0, want: 8192, wantOK: true},
		{name: "float-fractional", val: 8192.5, wantErr: true},
		{name: "json.Number integral", val: json.Number("8192"), want: 8192, wantOK: true},
		{name: "json.Number fractional", val: json.Number("8192.5"), wantErr: true},
		{name: "float out of range", val: 1e19, wantErr: true},
		{name: "json.Number out of int64 range", val: json.Number("9223372036854775808"), wantErr: true},
		{name: "string", val: "8192", wantErr: true},
		{name: "bool", val: true, wantErr: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			params := map[string]any{}
			if tc.val != nil {
				params["k"] = tc.val
			}
			got, ok, err := paramInt(params, "k")
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, got value=%d ok=%v", got, ok)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if ok != tc.wantOK {
				t.Fatalf("ok = %v, want %v", ok, tc.wantOK)
			}
			if ok && got != tc.want {
				t.Fatalf("value = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestParamDuration(t *testing.T) {
	cases := []struct {
		name    string
		val     any
		want    time.Duration
		wantOK  bool
		wantErr bool
	}{
		{name: "absent", val: nil, wantOK: false},
		{name: "valid", val: "30s", want: 30 * time.Second, wantOK: true},
		{name: "missing unit", val: "30", wantErr: true},
		{name: "garbage", val: "abc", wantErr: true},
		{name: "non-string", val: 30, wantErr: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			params := map[string]any{}
			if tc.val != nil {
				params["k"] = tc.val
			}
			got, ok, err := paramDuration(params, "k")
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, got value=%v ok=%v", got, ok)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if ok != tc.wantOK {
				t.Fatalf("ok = %v, want %v", ok, tc.wantOK)
			}
			if ok && got != tc.want {
				t.Fatalf("value = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestParamString(t *testing.T) {
	cases := []struct {
		name    string
		val     any
		want    string
		wantErr bool
	}{
		{name: "absent", val: nil, want: ""},
		{name: "string", val: "nixl", want: "nixl"},
		{name: "empty string", val: "", want: ""},
		{name: "non-string is an error", val: 42, wantErr: true},
		{name: "bool is an error", val: true, wantErr: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			params := map[string]any{}
			if tc.val != nil {
				params["k"] = tc.val
			}
			got, err := paramString(params, "k")
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, got value=%q", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Fatalf("value = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestParamBool(t *testing.T) {
	cases := []struct {
		name    string
		val     any
		want    bool
		wantOK  bool
		wantErr bool
	}{
		{name: "absent", val: nil, wantOK: false},
		{name: "true", val: true, want: true, wantOK: true},
		{name: "false", val: false, want: false, wantOK: true},
		{name: "non-bool is an error", val: "true", wantErr: true},
		{name: "int is an error", val: 1, wantErr: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			params := map[string]any{}
			if tc.val != nil {
				params["k"] = tc.val
			}
			got, ok, err := paramBool(params, "k")
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, got value=%v ok=%v", got, ok)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if ok != tc.wantOK {
				t.Fatalf("ok = %v, want %v", ok, tc.wantOK)
			}
			if ok && got != tc.want {
				t.Fatalf("value = %v, want %v", got, tc.want)
			}
		})
	}
}

// A float-formatted limit (as a YAML decoder may produce) must still apply,
// not silently fall through to the default. This is the regression T1 covers.
func TestNewRenderStep_FloatFormattedLimit(t *testing.T) {
	step, err := NewRenderStep(nil, map[string]any{"max_total_tokens": 5.0})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	rs := step.(*RenderStep)
	if rs.maxTotalTokens != 5 {
		t.Fatalf("maxTotalTokens = %d, want 5", rs.maxTotalTokens)
	}
}

func TestNewRenderStep_UnparsableTimeout(t *testing.T) {
	if _, err := NewRenderStep(nil, map[string]any{"timeout": "30"}); err == nil {
		t.Fatal("expected error for timeout without a unit")
	}
}

func TestNewReplaceMediaURLsStep_UnparsableTimeout(t *testing.T) {
	if _, err := NewReplaceMediaURLsStep(nil, map[string]any{"download_timeout": "abc"}); err == nil {
		t.Fatal("expected error for unparsable download_timeout")
	}
}

func TestNewReplaceMediaURLsStep_FloatFormattedLimit(t *testing.T) {
	step, err := NewReplaceMediaURLsStep(nil, map[string]any{"max_download_size": 5.0})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	rs := step.(*ReplaceMediaURLsStep)
	if rs.maxDownloadSize != 5*config.BytesPerMB {
		t.Fatalf("maxDownloadSize = %d, want %d", rs.maxDownloadSize, 5*config.BytesPerMB)
	}
}

func TestNewEncodeStep_FloatFormattedLimit(t *testing.T) {
	step, err := NewEncodeStep(gateway.New(config.GatewayConfig{}), map[string]any{"max_parallel": 4.0})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	es := step.(*EncodeStep)
	if es.maxParallel != 4 {
		t.Fatalf("maxParallel = %d, want 4", es.maxParallel)
	}
}
