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

package agentidentity_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/requestheader/agentidentity"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/util/sessionaffinity"
)

// TestOptionalConsumersFallBackWhenSilent: plugins that read agent-identity as
// an optional session source (session affinity with an attribute source, the
// rendezvous filter) resolve the same session with publishIdentity: false as
// without agent-identity configured: the next source, here x-session-id.
func TestOptionalConsumersFallBackWhenSilent(t *testing.T) {
	sources := []sessionaffinity.SessionIDSource{
		{Attribute: "agent-identity"},
		{Header: sessionaffinity.DefaultSessionIDHeader},
	}
	headers := map[string]string{
		agentidentity.ClaudeCodeSessionHeader:  "agent-session",
		sessionaffinity.DefaultSessionIDHeader: "header-session",
	}

	resolve := func(params string) string {
		t.Helper()
		req := &scheduling.InferenceRequest{Headers: headers}
		if params != "" {
			pi, err := agentidentity.PluginFactory("agent-identity", json.NewDecoder(strings.NewReader(params)), nil)
			if err != nil {
				t.Fatalf("PluginFactory(%s): %v", params, err)
			}
			if err := pi.(*agentidentity.Plugin).RequestHeader(context.Background(), req); err != nil {
				t.Fatalf("RequestHeader: %v", err)
			}
		}
		return sessionaffinity.ResolveSessionID(context.Background(), req, sources)
	}

	notConfigured := resolve("")
	if got := resolve(`{"publishIdentity": false, "metrics": true}`); got != notConfigured || got != "header-session" {
		t.Errorf("publishIdentity false resolved %q, want %q (same as not configured)", got, notConfigured)
	}
	if got := resolve(`{}`); got != "agent-session" {
		t.Errorf("publishIdentity true resolved %q, want the agent identity", got)
	}
}
