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
	"fmt"
	"testing"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const testSessionHeader = "x-session-id"

func prefixAffinityConfig(sessionHeader string) Config {
	config := validConfig()
	config.RevisionGating.Affinity = &RevisionAffinity{Mode: AffinityModePrefix, SessionHeader: sessionHeader}
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
// resending the conversation so far.
func sessionTurns(session int) []*fwksched.InferenceRequest {
	system := chatMessage("system", "You are a helpful assistant.")
	first := chatMessage("user", fmt.Sprintf("session %d: summarize this document", session))
	return []*fwksched.InferenceRequest{
		chatRequest(nil, system, first),
		chatRequest(nil, system, first, chatMessage("assistant", "summary"), chatMessage("user", "shorter")),
		chatRequest(nil, system, first, chatMessage("assistant", "summary"), chatMessage("user", "shorter"),
			chatMessage("assistant", "short"), chatMessage("user", "thanks")),
	}
}

func mustDraw(t *testing.T, screener *Screener, request *fwksched.InferenceRequest) float64 {
	t.Helper()
	draw, ok := screener.affinityDraw(request)
	if !ok {
		t.Fatal("expected a keyed draw")
	}
	if draw < 0 || draw >= 1 {
		t.Fatalf("draw %v outside [0, 1)", draw)
	}
	return draw
}

func TestAffinityDrawStableAcrossSessionTurns(t *testing.T) {
	screener := newTestScreener(prefixAffinityConfig(""))
	turns := sessionTurns(1)
	want := mustDraw(t, screener, turns[0])
	for i, turn := range turns[1:] {
		if got := mustDraw(t, screener, turn); got != want {
			t.Fatalf("turn %d drew %v, want %v from turn 0", i+1, got, want)
		}
	}
	if other := mustDraw(t, screener, sessionTurns(2)[0]); other == want {
		t.Fatal("sessions that share only a system prompt must not share a key")
	}
}

func TestAffinityDrawKeysAnthropicMessages(t *testing.T) {
	screener := newTestScreener(prefixAffinityConfig(""))
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
	first := mustDraw(t, screener, request(text("user", "question")))
	next := mustDraw(t, screener, request(text("user", "question"), text("assistant", "answer"), text("user", "more")))
	if first != next {
		t.Fatalf("later turn drew %v, want %v", next, first)
	}
	if other := mustDraw(t, screener, request(text("user", "different question"))); other == first {
		t.Fatal("different first user messages must not share a key")
	}
}

func TestAffinityDrawSessionHeaderTakesPrecedence(t *testing.T) {
	screener := newTestScreener(prefixAffinityConfig(testSessionHeader))
	headers := map[string]string{testSessionHeader: "session-a"}
	a := mustDraw(t, screener, chatRequest(headers, chatMessage("user", "one prompt")))
	b := mustDraw(t, screener, chatRequest(headers, chatMessage("user", "another prompt")))
	if a != b {
		t.Fatalf("same session header drew %v and %v", a, b)
	}
	withoutHeader := mustDraw(t, screener, chatRequest(nil, chatMessage("user", "one prompt")))
	if withoutHeader == a {
		t.Fatal("a request without the session header must key on its prompt")
	}
}

func TestAffinityDrawFallsBackWhenUnkeyed(t *testing.T) {
	tests := []struct {
		name    string
		config  Config
		request *fwksched.InferenceRequest
	}{
		{"affinity off", validConfig(), chatRequest(nil, chatMessage("user", "hi"))},
		{"nil request", prefixAffinityConfig(""), nil},
		{"no body", prefixAffinityConfig(""), &fwksched.InferenceRequest{Headers: map[string]string{}}},
		{"no messages", prefixAffinityConfig(""), chatRequest(nil)},
		{"completions", prefixAffinityConfig(""), &fwksched.InferenceRequest{
			Headers: map[string]string{},
			Body:    &fwkrh.InferenceRequestBody{Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: "hi"}}},
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if draw, ok := newTestScreener(test.config).affinityDraw(test.request); ok {
				t.Fatalf("want random fallback, got keyed draw %v", draw)
			}
		})
	}
}

// TestScreenerPrefixAffinityPinsSessionsAndKeepsShares checks the two
// properties the issue asks for together: every turn of a session lands on one
// revision, and the revisions still receive their shares across sessions.
func TestScreenerPrefixAffinityPinsSessionsAndKeepsShares(t *testing.T) {
	screener := newTestScreener(prefixAffinityConfig(""))
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
			got := screenCandidates(t, screener, turn, candidates)
			if len(got) == 0 {
				t.Fatal("empty survivor set")
			}
			turnRevision := got[0].GetMetadata().Labels[testRevLabel]
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

// TestPrefixAffinityKeysOnlyMoveTowardTheGrowingRevision follows one key set
// through a rollout. As the new revision's share grows, a key may move from the
// old revision to the new one but never back, so a key keeps its revision, and
// that revision's KV cache, for as long as the shares allow.
func TestPrefixAffinityKeysOnlyMoveTowardTheGrowingRevision(t *testing.T) {
	screener := newTestScreener(prefixAffinityConfig(""))
	const keys = 2000
	draws := make([]float64, keys)
	for i := range draws {
		draws[i] = mustDraw(t, screener, sessionTurns(i)[0])
	}
	for _, revisions := range [][2]string{{"old", "new"}, {"rev-b", "rev-a"}} {
		oldRevision, newRevision := revisions[0], revisions[1]
		onNew := make([]bool, keys)
		for step := 0; step <= 10; step++ {
			newShare := float64(step) / 10
			shares := map[string]float64{oldRevision: 1 - newShare, newRevision: newShare}
			moved := 0
			for i, draw := range draws {
				isNew := pickWeightedRevision(shares, draw) == newRevision
				if onNew[i] && !isNew {
					t.Fatalf("%v: key %d moved back to %s at new share %.1f", revisions, i, oldRevision, newShare)
				}
				if isNew && !onNew[i] {
					moved++
				}
				onNew[i] = isNew
			}
			if step > 0 && moved == 0 {
				t.Fatalf("%v: no key moved when the new share grew to %.1f", revisions, newShare)
			}
		}
	}
}

func TestScreenerPrefixAffinityKeepsStrictRevisionHeader(t *testing.T) {
	screener := newTestScreener(prefixAffinityConfig(""))
	seedCounts(t, screener, map[string]map[string]int{
		"v1": {"prefill": 1, "decode": 1},
		"v2": {"prefill": 1, "decode": 1},
	})
	for session := range 50 {
		request := sessionTurns(session)[0]
		request.Headers["x-llm-d-disagg-revision"] = "v2"
		got := screenCandidates(t, screener, request, candidatePool(1, 1))
		if len(got) != 1 || got[0].GetMetadata().Labels[testRevLabel] != "v2" {
			t.Fatalf("strict header must select v2, got %v", got)
		}
	}
}
