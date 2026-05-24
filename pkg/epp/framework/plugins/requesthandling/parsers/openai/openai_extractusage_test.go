/*
Copyright 2025 The llm-d Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package openai

import (
	"testing"
)

// Regression test for #981: extractUsage must not panic when an upstream
// response uses non-conforming types for usage fields. Before the fix,
// each vector below crashed the goroutine processing the response via an
// unchecked type assertion.
func TestExtractUsage_MalformedFields_NoPanic(t *testing.T) {
	cases := []struct {
		name string
		body string
	}{
		{
			name: "usage_field_is_string",
			body: `{"object":"chat.completion","usage":"oops"}`,
		},
		{
			name: "prompt_token_details_is_number",
			body: `{"object":"chat.completion","usage":{"prompt_token_details":0,"prompt_tokens":7}}`,
		},
		{
			name: "prompt_tokens_is_string",
			body: `{"object":"chat.completion","usage":{"prompt_tokens":"1024"}}`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("extractUsage panicked on %q: %v", tc.body, r)
				}
			}()
			_, _ = extractUsage([]byte(tc.body))
		})
	}
}

// Some non-conforming providers emit numeric usage fields as JSON strings;
// toInt coerces these rather than panicking.
func TestExtractUsage_StringCoercion(t *testing.T) {
	body := `{"object":"chat.completion","usage":{"prompt_tokens":"1024","completion_tokens":"7","total_tokens":"1031"}}`
	u, err := extractUsage([]byte(body))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if u == nil {
		t.Fatal("usage is nil")
	}
	if u.PromptTokens != 1024 {
		t.Errorf("PromptTokens = %d, want 1024", u.PromptTokens)
	}
	if u.CompletionTokens != 7 {
		t.Errorf("CompletionTokens = %d, want 7", u.CompletionTokens)
	}
	if u.TotalTokens != 1031 {
		t.Errorf("TotalTokens = %d, want 1031", u.TotalTokens)
	}
}
