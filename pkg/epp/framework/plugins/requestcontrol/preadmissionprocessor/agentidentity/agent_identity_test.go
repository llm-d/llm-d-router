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

package agentidentity

import (
	"context"
	"testing"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

func TestProcessPreAdmission(t *testing.T) {
	p := &Plugin{}

	tests := []struct {
		name           string
		fairnessID     string
		headers        map[string]string
		body           *fwkrh.InferenceRequestBody
		wantFairnessID string
	}{
		{
			name:           "explicit fairness ID is preserved",
			fairnessID:     "my-explicit-id",
			headers:        map[string]string{ClaudeCodeSessionHeader: "session-abc"},
			wantFairnessID: "my-explicit-id",
		},
		{
			name:           "claude code session header used when fairness ID is default",
			fairnessID:     metadata.DefaultFairnessID,
			headers:        map[string]string{ClaudeCodeSessionHeader: "session-abc"},
			wantFairnessID: "session-abc",
		},
		{
			name:           "claude code session header used when fairness ID is empty",
			fairnessID:     "",
			headers:        map[string]string{ClaudeCodeSessionHeader: "session-abc"},
			wantFairnessID: "session-abc",
		},
		{
			name:           "opencode session header",
			fairnessID:     metadata.DefaultFairnessID,
			headers:        map[string]string{OpenCodeSessionHeader: "oc-session-1"},
			wantFairnessID: "oc-session-1",
		},
		{
			name:           "opencode parent session header",
			fairnessID:     metadata.DefaultFairnessID,
			headers:        map[string]string{OpenCodeParentSessionHeader: "parent-1"},
			wantFairnessID: "parent-1",
		},
		{
			name:       "priority order: claude code wins over opencode",
			fairnessID: metadata.DefaultFairnessID,
			headers: map[string]string{
				ClaudeCodeSessionHeader: "session-abc",
				OpenCodeSessionHeader:   "oc-session-1",
			},
			wantFairnessID: "session-abc",
		},
		{
			name:       "priority order: opencode session wins over parent",
			fairnessID: metadata.DefaultFairnessID,
			headers: map[string]string{
				OpenCodeSessionHeader:       "oc-session-1",
				OpenCodeParentSessionHeader: "parent-1",
			},
			wantFairnessID: "oc-session-1",
		},
		{
			name:       "previous_response_id in body is ignored",
			fairnessID: metadata.DefaultFairnessID,
			headers:    map[string]string{},
			body: &fwkrh.InferenceRequestBody{
				Payload: fwkrh.PayloadMap{"previous_response_id": "resp-456"},
			},
			wantFairnessID: metadata.DefaultFairnessID,
		},
		{
			name:           "nil body does not panic",
			fairnessID:     metadata.DefaultFairnessID,
			headers:        map[string]string{},
			body:           nil,
			wantFairnessID: metadata.DefaultFairnessID,
		},
		{
			name:           "no matching headers keeps default",
			fairnessID:     metadata.DefaultFairnessID,
			headers:        map[string]string{"x-unrelated": "value"},
			wantFairnessID: metadata.DefaultFairnessID,
		},
		{
			name:           "empty headers keeps default",
			fairnessID:     metadata.DefaultFairnessID,
			headers:        map[string]string{},
			wantFairnessID: metadata.DefaultFairnessID,
		},
		{
			name:           "nil headers does not panic",
			fairnessID:     metadata.DefaultFairnessID,
			headers:        nil,
			wantFairnessID: metadata.DefaultFairnessID,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &scheduling.InferenceRequest{
				FairnessID: tt.fairnessID,
				Headers:    tt.headers,
				Body:       tt.body,
			}
			err := p.ProcessPreAdmission(context.Background(), req)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if req.FairnessID != tt.wantFairnessID {
				t.Errorf("FairnessID = %q, want %q", req.FairnessID, tt.wantFairnessID)
			}
		})
	}
}