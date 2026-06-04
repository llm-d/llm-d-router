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

package handlers

import (
	"context"
	"testing"

	"github.com/go-logr/logr"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

// testParser is a simple mock parser for testing routing logic.
type testParser struct {
	name  string
	paths []string
}

func (tp *testParser) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: tp.name, Name: tp.name}
}
func (tp *testParser) Claims() fwkrh.Claims {
	return fwkrh.Claims{
		Paths:     tp.paths,
		Protocols: nil,
	}
}
func (tp *testParser) ParseRequest(ctx context.Context, body []byte, headers map[string]string) (*fwkrh.ParseResult, error) {
	return nil, nil //nolint:nilnil
}
func (tp *testParser) ParseResponse(ctx context.Context, body []byte, headers map[string]string, endofStream bool) (*fwkrh.ParsedResponse, error) {
	return nil, nil //nolint:nilnil
}

func TestParserRegistry(t *testing.T) {
	openai := &testParser{name: "openai", paths: []string{"v1/chat/completions", "v1/completions"}}
	anthropic := &testParser{name: "anthropic", paths: []string{"v1/messages"}}
	vertex := &testParser{name: "vertex", paths: []string{"PredictionService/ChatCompletions"}}
	registry := NewParserRegistry([]fwkrh.Parser{openai, anthropic, vertex}, logr.Discard())

	tests := []struct {
		name        string
		requestPath string
		wantParser  string
		expectError bool
	}{
		{
			name:        "exact match openai chat completions",
			requestPath: "/v1/chat/completions",
			wantParser:  "openai",
		},
		{
			name:        "exact match anthropic messages",
			requestPath: "/v1/messages",
			wantParser:  "anthropic",
		},
		{
			name:        "suffix match vertex chat completions",
			requestPath: "/google.cloud.aiplatform.v1beta1.PredictionService/ChatCompletions",
			wantParser:  "vertex",
		},
		{
			name:        "suffix match vertex chat completions different version",
			requestPath: "/google.cloud.aiplatform.v1.PredictionService/ChatCompletions",
			wantParser:  "vertex",
		},
		{
			name:        "path with trailing slash normalize match",
			requestPath: "/v1/messages/",
			wantParser:  "anthropic",
		},
		{
			name:        "no suffix match returns error",
			requestPath: "/unknown/path",
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser, err := registry.Resolve(tt.requestPath)
			if (err != nil) != tt.expectError {
				t.Errorf("Resolve(%q) error = %v, expectError %v", tt.requestPath, err, tt.expectError)
				return
			}
			if tt.expectError {
				return
			}
			if parser.TypedName().Name != tt.wantParser {
				t.Errorf("Resolve(%q) resolved parser = %q, want %q", tt.requestPath, parser.TypedName().Name, tt.wantParser)
			}
		})
	}
}

func TestParserRegistryPriority(t *testing.T) {
	// Both plugins claim `v1/chat/completions`
	openai := &testParser{name: "openai", paths: []string{"v1/chat/completions"}}
	custom := &testParser{name: "custom", paths: []string{"v1/chat/completions"}}

	// 1. OpenAI configured first
	registry1 := NewParserRegistry([]fwkrh.Parser{openai, custom}, logr.Discard())
	parser1, err := registry1.Resolve("/v1/chat/completions")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if parser1.TypedName().Name != "openai" {
		t.Errorf("expected openai to take precedence, got %s", parser1.TypedName().Name)
	}

	// 2. Custom configured first
	registry2 := NewParserRegistry([]fwkrh.Parser{custom, openai}, logr.Discard())
	parser2, err := registry2.Resolve("/v1/chat/completions")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if parser2.TypedName().Name != "custom" {
		t.Errorf("expected custom to take precedence, got %s", parser2.TypedName().Name)
	}
}

func TestParserRegistryWithPassthrough(t *testing.T) {
	openai := &testParser{name: "openai", paths: []string{"v1/chat/completions", "v1/completions"}}
	passthrough := &testParser{name: "passthrough", paths: nil}
	registry := NewParserRegistry([]fwkrh.Parser{openai, passthrough}, logr.Discard())

	tests := []struct {
		name        string
		requestPath string
		wantParser  string
		expectError bool
	}{
		{
			name:        "specific match wins over passthrough",
			requestPath: "/v1/completions",
			wantParser:  "openai",
		},
		{
			name:        "passthrough match",
			requestPath: "/unknown/path",
			wantParser:  "passthrough",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser, err := registry.Resolve(tt.requestPath)
			if (err != nil) != tt.expectError {
				t.Errorf("Resolve(%q) error = %v, expectError %v", tt.requestPath, err, tt.expectError)
				return
			}
			if tt.expectError {
				return
			}
			if parser.TypedName().Name != tt.wantParser {
				t.Errorf("Resolve(%q) resolved parser = %q, want %q", tt.requestPath, parser.TypedName().Name, tt.wantParser)
			}
		})
	}
}

func TestParserRegistryDuplicateTypes(t *testing.T) {
	p1 := &testParser{name: "openai", paths: []string{"v1/chat/completions"}}
	p2 := &testParser{name: "openai", paths: []string{"v1/completions"}}

	registry := NewParserRegistry([]fwkrh.Parser{p1, p2}, logr.Discard())
	parsers := registry.Parsers()
	if len(parsers) != 1 {
		t.Errorf("expected 1 parser after skipping duplicate type, got %d", len(parsers))
	}
	if parsers[0] != p1 {
		t.Errorf("expected first parser to be kept, got %v", parsers[0])
	}

	_, err := registry.Resolve("/v1/completions")
	if err == nil {
		t.Error("expected error resolving path for skipped parser type, but succeeded")
	}
}
