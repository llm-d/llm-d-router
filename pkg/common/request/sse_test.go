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

package request

import "testing"

func TestSSEDataPayload(t *testing.T) {
	tests := []struct {
		name   string
		line   string
		want   string
		wantOK bool
	}{
		{name: "wire form", line: `data: {"a":1}`, want: `{"a":1}`, wantOK: true},
		{name: "no space after colon", line: `data:{"a":1}`, want: `{"a":1}`, wantOK: true},
		{name: "second space stays in the payload", line: `data:  x`, want: ` x`, wantOK: true},
		{name: "empty payload", line: `data: `, want: ``, wantOK: true},
		{name: "empty payload without space", line: `data:`, want: ``, wantOK: true},
		{name: "terminator is a payload", line: `data: [DONE]`, want: `[DONE]`, wantOK: true},
		{name: "another field", line: `event: message`, wantOK: false},
		{name: "comment", line: `: keep-alive`, wantOK: false},
		{name: "empty line", line: ``, wantOK: false},
		{name: "field whose name starts with data", line: `database: x`, wantOK: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := SSEDataPayload([]byte(tt.line))
			if ok != tt.wantOK || string(got) != tt.want {
				t.Fatalf("SSEDataPayload(%q) = (%q, %v), want (%q, %v)", tt.line, got, ok, tt.want, tt.wantOK)
			}
		})
	}
}

func TestSSEIsDone(t *testing.T) {
	tests := []struct {
		name    string
		payload string
		want    bool
	}{
		{name: "terminator", payload: "[DONE]", want: true},
		{name: "terminator with padding", payload: " [DONE] ", want: true},
		{name: "json", payload: `{"a":1}`, want: false},
		{name: "empty", payload: "", want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := SSEIsDone([]byte(tt.payload)); got != tt.want {
				t.Fatalf("SSEIsDone(%q) = %v, want %v", tt.payload, got, tt.want)
			}
		})
	}
}
