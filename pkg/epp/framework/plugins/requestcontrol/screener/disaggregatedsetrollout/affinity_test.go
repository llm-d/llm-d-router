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

package disaggregatedsetrollout

import (
	"encoding/json"
	"fmt"
	"testing"

	"google.golang.org/protobuf/types/known/wrapperspb"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/agentidentity"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/util/sessionaffinity"
)

const testSessionHeader = "x-session-id"

// testSources reads testSessionHeader, then the agent-identity attribute.
var testSources = []sessionaffinity.SessionIDSource{
	{Header: testSessionHeader},
	{Attribute: agentidentity.PluginType},
}

func prefixAffinityConfig(sources ...sessionaffinity.SessionIDSource) Config {
	config := validConfig()
	config.RevisionGating.Affinity = &RevisionAffinity{Mode: AffinityModePrefix, Sources: sources}
	return config
}

func chatMessage(role, text string) fwkrh.Message {
	return fwkrh.Message{Role: role, Content: fwkrh.Content{Raw: text}}
}

func chatRequest(headers map[string]string, messages ...fwkrh.Message) *fwksched.InferenceRequest {
	if headers == nil {
		headers = map[string]string{}
	}
	return &fwksched.InferenceRequest{
		Headers: headers,
		Body:    &fwkrh.InferenceRequestBody{ChatCompletions: &fwkrh.ChatCompletionsRequest{Messages: messages}},
	}
}

// sessionTurns returns the requests of a three-turn chat session, each turn
// resending the conversation so far under its own x-request-id.
func sessionTurns(session int) []*fwksched.InferenceRequest {
	system := chatMessage("system", "You are a helpful assistant.")
	first := chatMessage("user", fmt.Sprintf("session %d: summarize this document", session))
	turns := []*fwksched.InferenceRequest{
		chatRequest(nil, system, first),
		chatRequest(nil, system, first, chatMessage("assistant", "summary"), chatMessage("user", "shorter")),
		chatRequest(nil, system, first, chatMessage("assistant", "summary"), chatMessage("user", "shorter"),
			chatMessage("assistant", "short"), chatMessage("user", "thanks")),
	}
	for i, turn := range turns {
		turn.Headers[reqcommon.RequestIDHeaderKey] = fmt.Sprintf("session-%d-turn-%d", session, i)
	}
	return turns
}

func mustKey(t *testing.T, screener *Screener, request *fwksched.InferenceRequest) uint64 {
	t.Helper()
	key, source := screener.affinityKey(t.Context(), request)
	if source == sourceRandom {
		t.Fatal("expected a keyed request")
	}
	return key
}

func candidateRevisions(revisions ...string) []fwksched.Endpoint {
	endpoints := make([]fwksched.Endpoint, 0, len(revisions))
	for _, revision := range revisions {
		endpoints = append(endpoints, endpoint(revision+"-prefill", revLabels(revision)))
	}
	return endpoints
}

func TestAffinityKeyStableAcrossSessionTurns(t *testing.T) {
	screener := newTestScreener(prefixAffinityConfig())
	turns := sessionTurns(1)
	want := mustKey(t, screener, turns[0])
	for i, turn := range turns[1:] {
		if got := mustKey(t, screener, turn); got != want {
			t.Fatalf("turn %d keyed %v, want %v from turn 0", i+1, got, want)
		}
	}
	structured := chatRequest(nil,
		chatMessage("system", "You are a helpful assistant."),
		fwkrh.Message{Role: "user", Content: fwkrh.Content{
			Structured: []fwkrh.ContentBlock{{Type: "text", Text: "session 1: summarize this document"}},
		}},
	)
	if got := mustKey(t, screener, structured); got != want {
		t.Fatalf("structured text block keyed %v, want %v", got, want)
	}
	if other := mustKey(t, screener, sessionTurns(2)[0]); other == want {
		t.Fatal("sessions that share only a system prompt must not share a key")
	}
}

func TestAffinityKeyAnthropicMessages(t *testing.T) {
	screener := newTestScreener(prefixAffinityConfig())
	request := func(messages ...fwkrh.AnthropicMessage) *fwksched.InferenceRequest {
		return &fwksched.InferenceRequest{
			Headers: map[string]string{},
			Body: &fwkrh.InferenceRequestBody{Messages: &fwkrh.MessagesRequest{
				System:   fwkrh.AnthropicContent{Raw: "system"},
				Messages: messages,
			}},
		}
	}
	text := func(role, text string) fwkrh.AnthropicMessage {
		return fwkrh.AnthropicMessage{Role: role, Content: fwkrh.AnthropicContent{
			Structured: []fwkrh.AnthropicContentBlock{{Type: "text", Text: text}},
		}}
	}
	first := mustKey(t, screener, request(text("user", "question")))
	next := mustKey(t, screener, request(text("user", "question"), text("assistant", "answer"), text("user", "more")))
	if first != next {
		t.Fatalf("later turn keyed %v, want %v", next, first)
	}
	if other := mustKey(t, screener, request(text("user", "different question"))); other == first {
		t.Fatal("different first user messages must not share a key")
	}
}

func TestAffinityKeyResponses(t *testing.T) {
	screener := newTestScreener(prefixAffinityConfig())
	request := func(raw string) *fwksched.InferenceRequest {
		var responses fwkrh.ResponsesRequest
		if err := json.Unmarshal([]byte(raw), &responses); err != nil {
			t.Fatal(err)
		}
		return &fwksched.InferenceRequest{
			Headers: map[string]string{},
			Body:    &fwkrh.InferenceRequestBody{Responses: &responses},
		}
	}
	first := mustKey(t, screener, request(`{"instructions":"system","input":"question"}`))
	next := mustKey(t, screener, request(`{"instructions":"system","input":[
		{"role":"user","content":"question"},
		{"role":"assistant","content":[{"type":"output_text","text":"answer"}]},
		{"role":"user","content":"more"}]}`))
	if first != next {
		t.Fatalf("later turn keyed %v, want %v", next, first)
	}
	if other := mustKey(t, screener, request(`{"instructions":"system","input":"different question"}`)); other == first {
		t.Fatal("different first user messages must not share a key")
	}
	if key, source := screener.affinityKey(t.Context(), request(`{"previous_response_id":"resp_1","input":"more"}`)); source != sourceRandom {
		t.Fatalf("stateful Responses turn: want random fallback, got key %v from %s", key, source)
	}
}

// TestAffinityKeyTokenizedRequestsUseOnlySessionID checks that pre-tokenized
// bodies, as the vLLM gRPC, vLLM HTTP generate, and SGLang parsers build them,
// key only on a session identifier and otherwise fall back to random.
func TestAffinityKeyTokenizedRequestsUseOnlySessionID(t *testing.T) {
	tokens := []uint32{1, 2, 3}
	bodies := map[string]func() *fwkrh.InferenceRequestBody{
		"grpc tokenized generate": func() *fwkrh.InferenceRequestBody {
			return &fwkrh.InferenceRequestBody{
				Completions:      &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{TokenIDs: [][]uint32{tokens}}},
				Payload:          fwkrh.PayloadProto{Message: wrapperspb.String("generate")},
				TokenizedRequest: fwkrh.NewTokenizedRequest([][]uint32{tokens}),
			}
		},
		"grpc tokenized embed": func() *fwkrh.InferenceRequestBody {
			return &fwkrh.InferenceRequestBody{
				Embeddings: &fwkrh.EmbeddingsRequest{Input: fwkrh.EmbeddingsInput{TokenIDs: [][]uint32{tokens}}},
				Payload:    fwkrh.PayloadProto{Message: wrapperspb.String("embed")},
			}
		},
		"vLLM and SGLang generate token IDs": func() *fwkrh.InferenceRequestBody {
			return &fwkrh.InferenceRequestBody{
				Generate: &fwkrh.GenerateRequest{TokenIDs: tokens},
				Payload:  fwkrh.PayloadMap{"token_ids": tokens},
			}
		},
		"completions token IDs": func() *fwkrh.InferenceRequestBody {
			return &fwkrh.InferenceRequestBody{
				Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{TokenIDs: [][]uint32{tokens}}},
			}
		},
	}
	screener := newTestScreener(prefixAffinityConfig(testSources...))
	sessionKey := mustKey(t, screener, chatRequest(map[string]string{testSessionHeader: "session-a"}, chatMessage("user", "hi")))
	for name, body := range bodies {
		t.Run(name, func(t *testing.T) {
			withSession := &fwksched.InferenceRequest{Headers: map[string]string{testSessionHeader: "session-a"}, Body: body()}
			key, source := screener.affinityKey(t.Context(), withSession)
			if source != sourceSession || key != sessionKey {
				t.Fatalf("with session header: got key %v from %s, want %v from %s", key, source, sessionKey, sourceSession)
			}
			withoutSession := &fwksched.InferenceRequest{Headers: map[string]string{}, Body: body()}
			if key, source := screener.affinityKey(t.Context(), withoutSession); source != sourceRandom {
				t.Fatalf("without session header: want random fallback, got key %v from %s", key, source)
			}
		})
	}
}

// TestAffinityKeyVertexGRPCChat checks that a chat request carrying the Vertex
// AI gRPC envelope as its payload keys on the same prefix as its HTTP form.
func TestAffinityKeyVertexGRPCChat(t *testing.T) {
	screener := newTestScreener(prefixAffinityConfig())
	http := chatRequest(nil, chatMessage("user", "question"))
	grpc := chatRequest(nil, chatMessage("user", "question"))
	grpc.Body.Payload = fwkrh.PayloadProto{Message: wrapperspb.String("vertex envelope")}
	key, source := screener.affinityKey(t.Context(), grpc)
	if source != sourcePrefix {
		t.Fatalf("want %s, got %s", sourcePrefix, source)
	}
	if want := mustKey(t, screener, http); key != want {
		t.Fatalf("gRPC chat keyed %v, want %v from the HTTP form", key, want)
	}
}

func TestAffinityKeyPrecedence(t *testing.T) {
	screener := newTestScreener(prefixAffinityConfig(testSources...))
	withAgentID := func(headers map[string]string, agentID, prompt string) *fwksched.InferenceRequest {
		req := chatRequest(headers, chatMessage("user", prompt))
		if agentID != "" {
			req.PutAttribute(agentidentity.AgentIdentityKey, agentID)
		}
		return req
	}

	// session header > agent-identity > conversation prefix.
	headerA := mustKey(t, screener, withAgentID(map[string]string{testSessionHeader: "session-a"}, "agent-1", "prompt-1"))
	headerB := mustKey(t, screener, withAgentID(map[string]string{testSessionHeader: "session-a"}, "agent-2", "prompt-2"))
	if headerA != headerB {
		t.Fatalf("same sessionHeader keyed %v and %v", headerA, headerB)
	}

	agentA := mustKey(t, screener, withAgentID(nil, "agent-1", "prompt-1"))
	agentCompacted := mustKey(t, screener, withAgentID(nil, "agent-1", "compacted summary"))
	if agentA != agentCompacted {
		t.Fatalf("same agent-identity after compaction keyed %v and %v", agentA, agentCompacted)
	}
	if agentA == headerA {
		t.Fatal("sessionHeader must take precedence over agent-identity")
	}
	if otherAgent := mustKey(t, screener, withAgentID(nil, "agent-2", "prompt-1")); otherAgent == agentA {
		t.Fatal("distinct agent-identity values with identical prompts must not share a key")
	}

	// agent-identity still keys a multimodal request whose body would otherwise fall back.
	multimodalWithAgent := chatRequest(nil, fwkrh.Message{
		Role: "user",
		Content: fwkrh.Content{Structured: []fwkrh.ContentBlock{
			{Type: "text", Text: "What is in this image?"},
			{Type: "image_url", ImageURL: fwkrh.ImageBlock{URL: "https://example.com/a.jpg"}},
		}},
	})
	multimodalWithAgent.PutAttribute(agentidentity.AgentIdentityKey, "agent-1")
	if got := mustKey(t, screener, multimodalWithAgent); got != agentA {
		t.Fatalf("multimodal request with agent-identity keyed %v, want %v", got, agentA)
	}

	promptOnly := mustKey(t, screener, withAgentID(nil, "", "prompt-1"))
	if promptOnly == agentA || promptOnly == headerA {
		t.Fatal("a request without a resolved session identifier must key on its prompt")
	}

	// Without an attribute source, agent-identity is ignored.
	headerOnly := newTestScreener(prefixAffinityConfig(sessionaffinity.SessionIDSource{Header: testSessionHeader}))
	if got := mustKey(t, headerOnly, withAgentID(nil, "agent-1", "prompt-1")); got != promptOnly {
		t.Fatalf("unconfigured agent-identity keyed %v, want prompt key %v", got, promptOnly)
	}
}

func TestScreenerConsumesConfiguredAttributes(t *testing.T) {
	if deps := newTestScreener(validConfig()).Consumes(); len(deps.Required) != 0 || len(deps.Optional) != 0 {
		t.Fatalf("affinity off: want no consumed keys, got %+v", deps)
	}
	if deps := newTestScreener(prefixAffinityConfig()).Consumes(); len(deps.Required) != 0 || len(deps.Optional) != 0 {
		t.Fatalf("no attribute sources: want no consumed keys, got %+v", deps)
	}
	deps := newTestScreener(prefixAffinityConfig(testSources...)).Consumes()
	if len(deps.Required) != 0 {
		t.Fatalf("want no required keys, got %+v", deps.Required)
	}
	if _, ok := deps.Optional[agentidentity.AgentIdentityKey]; !ok || len(deps.Optional) != 1 {
		t.Fatalf("want only optional %s, got %+v", agentidentity.AgentIdentityKey, deps.Optional)
	}
}

func TestAffinityKeyFallsBackWhenUnkeyed(t *testing.T) {
	tests := []struct {
		name    string
		config  Config
		request *fwksched.InferenceRequest
	}{
		{"affinity off", validConfig(), chatRequest(nil, chatMessage("user", "hi"))},
		{"nil request", prefixAffinityConfig(), nil},
		{"no body", prefixAffinityConfig(), &fwksched.InferenceRequest{Headers: map[string]string{}}},
		{"whitespace session header", prefixAffinityConfig(testSources...), &fwksched.InferenceRequest{
			Headers: map[string]string{testSessionHeader: "   "},
		}},
		{"no messages", prefixAffinityConfig(), chatRequest(nil)},
		{"system without user", prefixAffinityConfig(), chatRequest(nil, chatMessage("system", "preamble"))},
		{"empty user text", prefixAffinityConfig(), chatRequest(nil, chatMessage("user", ""))},
		{"empty structured blocks", prefixAffinityConfig(), chatRequest(nil, fwkrh.Message{
			Role: "user", Content: fwkrh.Content{Structured: []fwkrh.ContentBlock{}},
		})},
		{"completions", prefixAffinityConfig(), &fwksched.InferenceRequest{
			Headers: map[string]string{},
			Body:    &fwkrh.InferenceRequestBody{Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: "hi"}}},
		}},
		{"chat text and image", prefixAffinityConfig(), chatRequest(nil, fwkrh.Message{
			Role: "user",
			Content: fwkrh.Content{Structured: []fwkrh.ContentBlock{
				{Type: "text", Text: "What is in this image?"},
				{Type: "image_url", ImageURL: fwkrh.ImageBlock{URL: "https://example.com/a.jpg"}},
			}},
		})},
		{"chat encode priming body", prefixAffinityConfig(), chatRequest(nil, fwkrh.Message{
			Role: "user",
			Content: fwkrh.Content{Structured: []fwkrh.ContentBlock{
				{Type: "image_url", ImageURL: fwkrh.ImageBlock{URL: "https://example.com/a.jpg"}},
			}},
		})},
		{"chat audio block", prefixAffinityConfig(), chatRequest(nil, fwkrh.Message{
			Role: "user",
			Content: fwkrh.Content{Structured: []fwkrh.ContentBlock{
				{Type: "audio_url", AudioURL: fwkrh.AudioURLBlock{URL: "https://example.com/a.wav"}},
			}},
		})},
		{"chat tool calls before user", prefixAffinityConfig(), chatRequest(nil,
			fwkrh.Message{Role: "assistant", ToolCalls: []any{map[string]any{"id": "call_1"}}},
			chatMessage("user", "next"),
		)},
		{"anthropic text and image", prefixAffinityConfig(), &fwksched.InferenceRequest{
			Headers: map[string]string{},
			Body: &fwkrh.InferenceRequestBody{Messages: &fwkrh.MessagesRequest{
				Messages: []fwkrh.AnthropicMessage{{
					Role: "user",
					Content: fwkrh.AnthropicContent{Structured: []fwkrh.AnthropicContentBlock{
						{Type: "text", Text: "What is in this image?"},
						{Type: "image", Source: json.RawMessage(`{"type":"url","url":"https://example.com/a.jpg"}`)},
					}},
				}},
			}},
		}},
		{"anthropic non-text system block", prefixAffinityConfig(), &fwksched.InferenceRequest{
			Headers: map[string]string{},
			Body: &fwkrh.InferenceRequestBody{Messages: &fwkrh.MessagesRequest{
				System: fwkrh.AnthropicContent{Structured: []fwkrh.AnthropicContentBlock{
					{Type: "image", Source: json.RawMessage(`{"type":"url","url":"https://example.com/a.jpg"}`)},
				}},
				Messages: []fwkrh.AnthropicMessage{{
					Role:    "user",
					Content: fwkrh.AnthropicContent{Raw: "hi"},
				}},
			}},
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if key, source := newTestScreener(test.config).affinityKey(t.Context(), test.request); source != sourceRandom {
				t.Fatalf("want random fallback, got key %v from %s", key, source)
			}
		})
	}
}

// TestScreenerPrefixAffinityMultimodalFallsBackToRandom verifies that requests
// sharing a text question while varying an image (with no session header or
// agent identity) fall back to weighted random selection rather than collapsing
// onto one revision for both Chat Completions and Anthropic Messages.
func TestScreenerPrefixAffinityMultimodalFallsBackToRandom(t *testing.T) {
	config := prefixAffinityConfig()
	config.RevisionGating.Mode = GatingModeMaxRole
	screener := newTestScreener(config)
	// 80/20 split on decode.
	seedCounts(t, screener, map[string]map[string]int{
		"v1": {"prefill": 1, "decode": 8},
		"v2": {"prefill": 1, "decode": 2},
	})
	pool := candidateRevisions("v1", "v2")

	const requests = 1000
	t.Run("chat completions", func(t *testing.T) {
		onV2 := 0
		for i := range requests {
			req := chatRequest(
				map[string]string{reqcommon.RequestIDHeaderKey: fmt.Sprintf("chat-mm-%d", i)},
				fwkrh.Message{
					Role: "user",
					Content: fwkrh.Content{Structured: []fwkrh.ContentBlock{
						{Type: "text", Text: "What is in this image?"},
						{Type: "image_url", ImageURL: fwkrh.ImageBlock{URL: fmt.Sprintf("https://example.com/%d.jpg", i)}},
					}},
				},
			)
			if screenedRevision(t, screener, req, pool) == "v2" {
				onV2++
			}
		}
		if share := float64(onV2) / requests; share < 0.14 || share > 0.26 {
			t.Fatalf("want ~0.20 v2 share across multimodal chat requests, got %.3f", share)
		}
	})

	t.Run("anthropic messages", func(t *testing.T) {
		onV2 := 0
		for i := range requests {
			req := &fwksched.InferenceRequest{
				Headers: map[string]string{reqcommon.RequestIDHeaderKey: fmt.Sprintf("messages-mm-%d", i)},
				Body: &fwkrh.InferenceRequestBody{Messages: &fwkrh.MessagesRequest{
					Messages: []fwkrh.AnthropicMessage{{
						Role: "user",
						Content: fwkrh.AnthropicContent{Structured: []fwkrh.AnthropicContentBlock{
							{Type: "text", Text: "What is in this image?"},
							{Type: "image", Source: json.RawMessage(fmt.Sprintf(`{"type":"url","url":"https://example.com/%d.jpg"}`, i))},
						}},
					}},
				}},
			}
			if screenedRevision(t, screener, req, pool) == "v2" {
				onV2++
			}
		}
		if share := float64(onV2) / requests; share < 0.14 || share > 0.26 {
			t.Fatalf("want ~0.20 v2 share across multimodal Anthropic requests, got %.3f", share)
		}
	})
}

// TestScreenerPrefixAffinityPinsSessionsAndKeepsShares checks that every turn
// of a session lands on one revision while the revisions receive their
// configured shares across sessions.
func TestScreenerPrefixAffinityPinsSessionsAndKeepsShares(t *testing.T) {
	screener := newTestScreener(prefixAffinityConfig())
	// sum mode: v1 weight 10, v2 weight 3.
	seedCounts(t, screener, map[string]map[string]int{
		"v1": {"prefill": 2, "decode": 8},
		"v2": {"prefill": 1, "decode": 2},
	})
	candidates := candidatePool(2, 1)
	const sessions = 4000
	onV2 := 0
	for session := range sessions {
		revision := ""
		for i, turn := range sessionTurns(session) {
			turnRevision := screenedRevision(t, screener, turn, candidates)
			if i == 0 {
				revision = turnRevision
			} else if turnRevision != revision {
				t.Fatalf("session %d turn %d moved from %s to %s", session, i, revision, turnRevision)
			}
		}
		if revision == "v2" {
			onV2++
		}
	}
	want := 3.0 / 13.0
	if share := float64(onV2) / sessions; share < want-0.03 || share > want+0.03 {
		t.Fatalf("want v2 session share %.3f +/- .03, got %.3f", want, share)
	}
}

func TestScreenerPrefixAffinityAgentIdentityCompactionAndIdenticalPrompts(t *testing.T) {
	screener := newTestScreener(prefixAffinityConfig(testSources...))
	seedCounts(t, screener, map[string]map[string]int{
		"v1": {"prefill": 1, "decode": 5},
		"v2": {"prefill": 1, "decode": 5},
	})
	pool := candidateRevisions("v1", "v2")

	const sessions = 1000
	onV2 := 0
	for session := range sessions {
		agentID := fmt.Sprintf("agent-session-%d", session)
		// Every session starts with the exact same user prompt, then rewrites
		// its first message on turn 2 as happens after context compaction.
		turn1 := chatRequest(
			map[string]string{reqcommon.RequestIDHeaderKey: fmt.Sprintf("s-%d-t1", session)},
			chatMessage("user", "run the standard benchmark suite"),
		)
		turn1.PutAttribute(agentidentity.AgentIdentityKey, agentID)

		turn2 := chatRequest(
			map[string]string{reqcommon.RequestIDHeaderKey: fmt.Sprintf("s-%d-t2", session)},
			chatMessage("user", fmt.Sprintf("[compacted summary for %d]", session)),
			chatMessage("assistant", "done"),
			chatMessage("user", "report failures"),
		)
		turn2.PutAttribute(agentidentity.AgentIdentityKey, agentID)

		rev1 := screenedRevision(t, screener, turn1, pool)
		rev2 := screenedRevision(t, screener, turn2, pool)
		if rev1 != rev2 {
			t.Fatalf("agent session %d moved from %s to %s after compaction", session, rev1, rev2)
		}
		if rev1 == "v2" {
			onV2++
		}
	}
	if share := float64(onV2) / sessions; share < 0.44 || share > 0.56 {
		t.Fatalf("identical-prompt agent sessions: want ~0.50 v2 share, got %.3f", share)
	}
}

// TestPrefixAffinityKeysOnlyMoveTowardTheGrowingRevision follows one session
// set through a two-revision rollout via Screen. As the new revision's share
// grows, a session may move from the old revision to the new one, never back,
// regardless of revision name order.
func TestPrefixAffinityKeysOnlyMoveTowardTheGrowingRevision(t *testing.T) {
	const sessions = 1500
	for _, revisions := range [][2]string{{"old", "new"}, {"rev-b", "rev-a"}} {
		oldRevision, newRevision := revisions[0], revisions[1]
		pool := candidateRevisions(oldRevision, newRevision)
		onNew := make([]bool, sessions)
		for step := 1; step <= 9; step++ {
			config := prefixAffinityConfig()
			config.RevisionGating.Mode = GatingModeMaxRole
			screener := newTestScreener(config)
			seedCounts(t, screener, map[string]map[string]int{
				oldRevision: {"prefill": 1, "decode": 10 - step},
				newRevision: {"prefill": 1, "decode": step},
			})
			moved := 0
			for i := range sessions {
				turn := chatRequest(
					map[string]string{reqcommon.RequestIDHeaderKey: fmt.Sprintf("step-%d-session-%d", step, i)},
					chatMessage("user", fmt.Sprintf("session %d", i)),
				)
				isNew := screenedRevision(t, screener, turn, pool) == newRevision
				if onNew[i] && !isNew {
					t.Fatalf("%v: session %d moved back to %s at step %d", revisions, i, oldRevision, step)
				}
				if isNew && !onNew[i] {
					moved++
				}
				onNew[i] = isNew
			}
			if moved == 0 {
				t.Fatalf("%v: no session moved when %s grew at step %d", revisions, newRevision, step)
			}
		}
	}
}

// TestScreenerPrefixAffinityThreeRevisions verifies three-revision rollouts
// through Screen: shares hold across all three revisions, and changing one
// revision's replica count never moves sessions between the other two.
func TestScreenerPrefixAffinityThreeRevisions(t *testing.T) {
	newMaxRoleScreener := func(decodeCounts map[string]int) *Screener {
		config := prefixAffinityConfig()
		config.RevisionGating.Mode = GatingModeMaxRole
		screener := newTestScreener(config)
		counts := make(map[string]map[string]int, len(decodeCounts))
		for rev, decode := range decodeCounts {
			counts[rev] = map[string]int{"prefill": 1, "decode": decode}
		}
		seedCounts(t, screener, counts)
		return screener
	}

	pool := candidateRevisions("a1", "b2", "c3")
	const sessions = 3000

	initial := newMaxRoleScreener(map[string]int{"a1": 4, "b2": 4, "c3": 2})
	assigned := make([]string, sessions)
	counts := map[string]int{}
	for i := range sessions {
		turns := sessionTurns(i)
		rev := screenedRevision(t, initial, turns[0], pool)
		if next := screenedRevision(t, initial, turns[1], pool); next != rev {
			t.Fatalf("session %d moved from %s to %s across turns", i, rev, next)
		}
		assigned[i] = rev
		counts[rev]++
	}
	for rev, want := range map[string]float64{"a1": 0.4, "b2": 0.4, "c3": 0.2} {
		if got := float64(counts[rev]) / sessions; got < want-0.04 || got > want+0.04 {
			t.Fatalf("revision %s share = %.3f, want %.2f +/- .04", rev, got, want)
		}
	}

	// Scale c3 up while a1 and b2 keep their replica counts: sessions may move
	// only onto c3, never between a1 and b2 or off c3.
	c3Grown := newMaxRoleScreener(map[string]int{"a1": 4, "b2": 4, "c3": 6})
	afterGrow := make([]string, sessions)
	for i := range sessions {
		req := chatRequest(
			map[string]string{reqcommon.RequestIDHeaderKey: fmt.Sprintf("grow-%d", i)},
			chatMessage("system", "You are a helpful assistant."),
			chatMessage("user", fmt.Sprintf("session %d: summarize this document", i)),
		)
		next := screenedRevision(t, c3Grown, req, pool)
		if next != assigned[i] && next != "c3" {
			t.Fatalf("c3 grew: session %d moved from %s to %s", i, assigned[i], next)
		}
		afterGrow[i] = next
	}

	// Scale a1 down while b2 and c3 keep their replica counts: no session on b2
	// or c3 moves.
	a1Shrunk := newMaxRoleScreener(map[string]int{"a1": 1, "b2": 4, "c3": 6})
	for i := range sessions {
		req := chatRequest(
			map[string]string{reqcommon.RequestIDHeaderKey: fmt.Sprintf("shrink-%d", i)},
			chatMessage("system", "You are a helpful assistant."),
			chatMessage("user", fmt.Sprintf("session %d: summarize this document", i)),
		)
		next := screenedRevision(t, a1Shrunk, req, pool)
		if afterGrow[i] != "a1" && next != afterGrow[i] {
			t.Fatalf("a1 shrank: session %d on %s moved to %s", i, afterGrow[i], next)
		}
	}
}

// TestScreenerPrefixAffinityCoverageLossAndRecovery verifies through Screen
// that when one revision loses coverage (for example during a prefill restart),
// sessions on the remaining revisions do not move, and when it recovers every
// session returns to its original revision.
func TestScreenerPrefixAffinityCoverageLossAndRecovery(t *testing.T) {
	config := prefixAffinityConfig()
	config.RevisionGating.Mode = GatingModeMaxRole
	screener := newTestScreener(config)

	seedPods(t, screener,
		readyPod("a1-prefill", "a1", "prefill"),
		readyPod("a1-decode-0", "a1", "decode"),
		readyPod("a1-decode-1", "a1", "decode"),
		readyPod("b2-prefill", "b2", "prefill"),
		readyPod("b2-decode-0", "b2", "decode"),
		readyPod("b2-decode-1", "b2", "decode"),
		readyPod("c3-prefill", "c3", "prefill"),
		readyPod("c3-decode-0", "c3", "decode"),
	)
	pool := candidateRevisions("a1", "b2", "c3")

	const sessions = 1000
	initial := make([]string, sessions)
	seen := map[string]int{}
	for i := range sessions {
		turns := sessionTurns(i)
		initial[i] = screenedRevision(t, screener, turns[0], pool)
		seen[initial[i]]++
	}
	for _, rev := range []string{"a1", "b2", "c3"} {
		if seen[rev] == 0 {
			t.Fatalf("revision %s received no initial sessions", rev)
		}
	}

	// a1 loses prefill coverage.
	handler := &podNotificationHandler{screener: screener}
	if err := handler.Extract(t.Context(), podEvent(t, notReadyPod("a1-prefill", "a1", "prefill"), fwkdl.EventAddOrUpdate)); err != nil {
		t.Fatal(err)
	}
	for i := range sessions {
		turns := sessionTurns(i)
		during := screenedRevision(t, screener, turns[1], pool)
		if during == "a1" {
			t.Fatalf("session %d routed to uncovered a1", i)
		}
		if initial[i] != "a1" && during != initial[i] {
			t.Fatalf("a1 lost coverage: session %d on %s moved to %s", i, initial[i], during)
		}
	}

	// a1 regains prefill coverage.
	if err := handler.Extract(t.Context(), podEvent(t, readyPod("a1-prefill", "a1", "prefill"), fwkdl.EventAddOrUpdate)); err != nil {
		t.Fatal(err)
	}
	for i := range sessions {
		turns := sessionTurns(i)
		recovered := screenedRevision(t, screener, turns[2], pool)
		if recovered != initial[i] {
			t.Fatalf("a1 recovered: session %d on %s, want %s", i, recovered, initial[i])
		}
	}
}

func TestScreenerPrefixAffinityKeepsStrictRevisionHeader(t *testing.T) {
	screener := newTestScreener(prefixAffinityConfig())
	seedCounts(t, screener, map[string]map[string]int{
		"v1": {"prefill": 1, "decode": 1},
		"v2": {"prefill": 1, "decode": 1},
	})
	for session := range 50 {
		request := sessionTurns(session)[0]
		request.Headers[DefaultRevisionHeader] = "v2"
		got := screenCandidates(t, screener, request, candidatePool(1, 1))
		if len(got) != 1 || got[0].GetMetadata().Labels[testRevLabel] != "v2" {
			t.Fatalf("strict header must select v2, got %v", got)
		}
	}
}

func TestScreenerPrefixAffinityWithOverflow(t *testing.T) {
	config := prefixAffinityConfig()
	config.RevisionGating.Mode = GatingModeMaxRole
	config.RevisionGating.Overflow = &RevisionOverflow{GuardPluginName: testGuardName}
	screener, err := newGuardedScreener(t, config, testGuardParams)
	if err != nil {
		t.Fatal(err)
	}
	seedCounts(t, screener, map[string]map[string]int{
		"a": {"prefill": 1, "decode": 2},
		"b": {"prefill": 1, "decode": 4},
		"c": {"prefill": 1, "decode": 4},
	})
	pool := []fwksched.Endpoint{
		loadedEndpoint("a-prefill", "a", "prefill", testOverflowLimit),
		loadedEndpoint("b-prefill", "b", "prefill", 0),
		loadedEndpoint("c-prefill", "c", "prefill", 0),
	}

	const sessions = 2000
	counts := map[string]int{}
	for session := range sessions {
		turns := sessionTurns(session)
		rev := screenedRevision(t, screener, turns[0], pool)
		if next := screenedRevision(t, screener, turns[1], pool); next != rev {
			t.Fatalf("overflowed session %d moved from %s to %s across turns", session, rev, next)
		}
		counts[rev]++
	}
	if counts["a"] != 0 {
		t.Fatalf("%d sessions sent to full revision a", counts["a"])
	}
	if b, c := float64(counts["b"]), float64(counts["c"]); b/(b+c) < 0.44 || b/(b+c) > 0.56 {
		t.Fatalf("overflow split b=%v c=%v, want about even", b, c)
	}
}
