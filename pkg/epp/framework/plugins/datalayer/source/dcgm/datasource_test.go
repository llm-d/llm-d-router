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

package dcgm

import (
	"testing"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

func TestDCGMDataSourceFactory_Defaults(t *testing.T) {
	p, err := DCGMDataSourceFactory("test", fwkplugin.StrictDecoder(nil), nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if p == nil {
		t.Fatal("expected non-nil plugin")
	}
	if got := p.TypedName().Type; got != DCGMDataSourceType {
		t.Errorf("Type = %q, want %q", got, DCGMDataSourceType)
	}
	if got := p.TypedName().Name; got != "test" {
		t.Errorf("Name = %q, want %q", got, "test")
	}
}

func TestDCGMDataSourceFactory_CustomParams(t *testing.T) {
	raw := []byte(`{"scheme":"https","path":"/gpu","port":9500,"insecureSkipVerify":false,"useNodeAddress":true}`)
	p, err := DCGMDataSourceFactory("custom", fwkplugin.StrictDecoder(raw), nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if p == nil {
		t.Fatal("expected non-nil plugin")
	}
	if got := p.TypedName().Name; got != "custom" {
		t.Errorf("Name = %q, want %q", got, "custom")
	}
}

func TestDCGMDataSourceFactory_InvalidScheme(t *testing.T) {
	raw := []byte(`{"scheme":"grpc"}`)
	_, err := DCGMDataSourceFactory("test", fwkplugin.StrictDecoder(raw), nil)
	if err == nil {
		t.Fatal("expected error for unsupported scheme")
	}
}

func TestDCGMDataSourceFactory_InvalidJSON(t *testing.T) {
	raw := []byte(`{invalid`)
	_, err := DCGMDataSourceFactory("test", fwkplugin.StrictDecoder(raw), nil)
	if err == nil {
		t.Fatal("expected error for malformed JSON")
	}
}

func TestDCGMDataSourceFactory_UnknownField(t *testing.T) {
	raw := []byte(`{"scheme":"http","bogus":"value"}`)
	_, err := DCGMDataSourceFactory("test", fwkplugin.StrictDecoder(raw), nil)
	if err == nil {
		t.Fatal("expected error for unknown field (strict decoding)")
	}
}
