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

package envoy

import (
	"testing"

	corev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/testing/protocmp"
)

func TestBuildImmediateResponse(t *testing.T) {
	header := &corev3.HeaderValueOption{Header: &corev3.HeaderValue{Key: "x-answer", RawValue: []byte("yes")}}
	tests := []struct {
		name       string
		status     envoyTypePb.StatusCode
		setHeaders []*corev3.HeaderValueOption
		body       []byte
	}{
		{name: "status only", status: envoyTypePb.StatusCode_NoContent},
		{name: "empty header list", status: envoyTypePb.StatusCode_NoContent, setHeaders: []*corev3.HeaderValueOption{}},
		{name: "headers", status: envoyTypePb.StatusCode_NoContent, setHeaders: []*corev3.HeaderValueOption{header}},
		{name: "headers and body", status: envoyTypePb.StatusCode_BadRequest, setHeaders: []*corev3.HeaderValueOption{header}, body: []byte(`{"error":{}}`)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ir := BuildImmediateResponse(tt.status, tt.setHeaders, tt.body).GetImmediateResponse()
			if ir == nil {
				t.Fatal("expected ImmediateResponse, got nil")
			}
			if got := ir.GetStatus().GetCode(); got != tt.status {
				t.Errorf("status = %v, want %v", got, tt.status)
			}
			if string(ir.GetBody()) != string(tt.body) {
				t.Errorf("body = %s, want %s", ir.GetBody(), tt.body)
			}
			if len(tt.setHeaders) == 0 {
				if ir.GetHeaders() != nil {
					t.Errorf("headers = %v, want no mutation", ir.GetHeaders())
				}
				return
			}
			if diff := cmp.Diff(tt.setHeaders, ir.GetHeaders().GetSetHeaders(), protocmp.Transform()); diff != "" {
				t.Errorf("headers mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
