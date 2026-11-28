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

package threadidentity

// THE PUBLISHED KEY: the identity the plugin derives at arrival is published on the flow-key
// attribute the director turns into the queue key — the scheduler downstream reads it, and a thread
// whose key never left this hook would be a thread nothing could seat.

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/requestheader/agentidentity"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

// filler produces n content bytes: one content byte is one token, so this controls how many blocks
// a body spans at the test block size.
func filler(seed byte, n int) string { return strings.Repeat(string(seed), n) }

func user(seed byte, n int) fwkrequest.Message {
	return fwkrequest.Message{Role: "user", Content: fwkrequest.Content{Raw: filler(seed, n)}}
}

// reqSeq makes every request a distinct id: a thread's outstanding set is keyed by request id.
var reqSeq atomic.Uint64

func chatReq(msgs ...fwkrequest.Message) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{
		RequestID: fmt.Sprintf("req-%d", reqSeq.Add(1)),
		Headers:   map[string]string{},
		Body:      &fwkrequest.InferenceRequestBody{ChatCompletions: &fwkrequest.ChatCompletionsRequest{Messages: msgs}},
	}
}

// newTestPlugin builds the plugin through its Factory and replaces the render backend: the rules
// under test are about the identity, not about a chat template's exact tokens.
func newTestPlugin(t *testing.T) *Plugin {
	t.Helper()
	params := `{"tokenProducer":{"modelName":"test-model","vllm":{"url":"http://127.0.0.1:1"}}}`
	built, err := Factory("thread", json.NewDecoder(strings.NewReader(params)), testutils.NewTestHandle(t.Context()))
	require.NoError(t, err, "the plugin must build from its parameters alone")
	p := built.(*Plugin)
	p.tokenize = testRenderer
	return p
}

// testRenderer stands in for a chat template: each message's role, then its content.
func testRenderer(_ context.Context, request *fwksched.InferenceRequest) error {
	if request.Body.TokenizedRequest != nil {
		return nil
	}
	var ids []uint32
	for _, m := range request.Body.ChatCompletions.Messages {
		ids = append(ids, roleToken(m.Role))
		for i := 0; i < len(m.Content.Raw); i++ {
			ids = append(ids, 2000+uint32(m.Content.Raw[i]))
		}
	}
	if len(ids) == 0 {
		return nil
	}
	request.Body.TokenizedRequest = &fwkrequest.TokenizedRequest{
		Prompts: []fwkrequest.PromptTokens{{TokenIDs: ids}},
	}
	return nil
}

// roleToken gives each role its own token, above the byte range.
func roleToken(role string) uint32 {
	switch role {
	case "user":
		return 1001
	case "assistant":
		return 1002
	case "system":
		return 1003
	}
	return 1005 + uint32(len(role))
}

// threadOf runs the header hook and returns the derived thread and the published flow key.
func threadOf(t *testing.T, p *Plugin, req *fwksched.InferenceRequest) (threadID, flowKey string) {
	t.Helper()
	require.NoError(t, p.RequestHeader(context.Background(), req))
	flowKey, _ = fwksched.ReadRequestAttribute[string](req, agentidentity.AgentIdentityKey)
	id, ok := fwksched.ReadRequestAttribute[string](req, ThreadIdentityKey)
	if !ok {
		return "", flowKey
	}
	return id, flowKey
}

// finishTurn models a full turn: the request leaves the queue and its response completes.
func finishTurn(t *testing.T, p *Plugin, req *fwksched.InferenceRequest) {
	t.Helper()
	require.NoError(t, p.PreRequest(context.Background(), req, nil))
	p.ResponseBody(context.Background(), req, &fwkrc.Response{
		EndOfStream:      true,
		TerminationCause: fwkrc.TerminationCauseNatural,
	}, nil)
}

// TestRequestHeader_PublishesTheFlowKey: the derived id is published on the flow-key attribute
// before the request enqueues, and the plugin does NOT write the request's FairnessID field (under
// the director's scope a RequestHeader processor holds a copy, so a field write would be lost).
func TestRequestHeader_PublishesTheFlowKey(t *testing.T) {
	p := newTestPlugin(t)

	req := chatReq(user('a', 600))
	id, flowKey := threadOf(t, p, req)
	require.NotEmpty(t, id)
	assert.Equal(t, id, flowKey, "the thread id must be the flow key attribute")
	assert.Empty(t, req.FairnessID, "the fairness id field is the director's to derive, not ours to write")
	assert.Len(t, id, idHexLen)
	assert.Regexp(t, "^[0-9a-f]+$", id, "thread ids are opaque, short hex")
}

// TestLifecycle_ContinuationIsTheSameID: a finished turn's successor continues the thread — the
// identity survives the request that created it.
func TestLifecycle_ContinuationIsTheSameID(t *testing.T) {
	p := newTestPlugin(t)

	first := chatReq(user('a', 600))
	id1, _ := threadOf(t, p, first)
	finishTurn(t, p, first)

	second := chatReq(user('a', 600), fwkrequest.Message{Role: "assistant", Content: fwkrequest.Content{Raw: filler('b', 40)}}, user('c', 60))
	id2, flow2 := threadOf(t, p, second)
	assert.Equal(t, id1, id2, "an appended turn continues the thread")
	assert.Equal(t, id1, flow2)
}
