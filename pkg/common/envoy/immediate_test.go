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
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/testing/protocmp"
)

func TestBuildImmediateResponse(t *testing.T) {
	header := func(key, value string) *corev3.HeaderValueOption {
		return &corev3.HeaderValueOption{Header: &corev3.HeaderValue{Key: key, RawValue: []byte(value)}}
	}
	tests := []struct {
		name    string
		code    envoyTypePb.StatusCode
		headers map[string]string
		body    []byte
		want    *extProcPb.ImmediateResponse
	}{
		{
			name:    "headers and no body",
			code:    envoyTypePb.StatusCode_NoContent,
			headers: map[string]string{"x-answer": "yes", "x-other": "value"},
			want: &extProcPb.ImmediateResponse{
				Status:  &envoyTypePb.HttpStatus{Code: envoyTypePb.StatusCode_NoContent},
				Headers: &extProcPb.HeaderMutation{SetHeaders: []*corev3.HeaderValueOption{header("x-answer", "yes"), header("x-other", "value")}},
			},
		},
		{
			name: "body and no headers",
			code: envoyTypePb.StatusCode_TooManyRequests,
			body: []byte("evicted"),
			want: &extProcPb.ImmediateResponse{
				Status: &envoyTypePb.HttpStatus{Code: envoyTypePb.StatusCode_TooManyRequests},
				Body:   []byte("evicted"),
			},
		},
		{
			name:    "headers and body",
			code:    envoyTypePb.StatusCode_PreconditionFailed,
			headers: map[string]string{"x-reason": "miss"},
			body:    []byte("rejected"),
			want: &extProcPb.ImmediateResponse{
				Status:  &envoyTypePb.HttpStatus{Code: envoyTypePb.StatusCode_PreconditionFailed},
				Headers: &extProcPb.HeaderMutation{SetHeaders: []*corev3.HeaderValueOption{header("x-reason", "miss")}},
				Body:    []byte("rejected"),
			},
		},
		{
			name:    "header with an empty value is set",
			code:    envoyTypePb.StatusCode_OK,
			headers: map[string]string{"x-empty": ""},
			want: &extProcPb.ImmediateResponse{
				Status:  &envoyTypePb.HttpStatus{Code: envoyTypePb.StatusCode_OK},
				Headers: &extProcPb.HeaderMutation{SetHeaders: []*corev3.HeaderValueOption{header("x-empty", "")}},
			},
		},
		{
			name: "no headers and no body",
			code: envoyTypePb.StatusCode_OK,
			want: &extProcPb.ImmediateResponse{
				Status: &envoyTypePb.HttpStatus{Code: envoyTypePb.StatusCode_OK},
			},
		},
		{
			name:    "empty header map adds no mutation",
			code:    envoyTypePb.StatusCode_OK,
			headers: map[string]string{},
			want: &extProcPb.ImmediateResponse{
				Status: &envoyTypePb.HttpStatus{Code: envoyTypePb.StatusCode_OK},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := BuildImmediateResponse(tt.code, tt.headers, tt.body)
			sortHeadersByKey(got.GetImmediateResponse().GetHeaders().GetSetHeaders())
			sortHeadersByKey(tt.want.GetHeaders().GetSetHeaders())
			want := &extProcPb.ProcessingResponse{
				Response: &extProcPb.ProcessingResponse_ImmediateResponse{ImmediateResponse: tt.want},
			}
			if diff := cmp.Diff(want, got, protocmp.Transform()); diff != "" {
				t.Errorf("BuildImmediateResponse() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
