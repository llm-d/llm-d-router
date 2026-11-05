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

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/go-logr/logr/funcr"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

// filler produces n deterministic content bytes; one content byte is one token,
// so this controls how many blocks a body spans at the default block size (64).
func filler(seed byte, n int) string { return strings.Repeat(string(seed), n) }

func user(seed byte, n int) fwkrh.Message {
	return fwkrh.Message{Role: "user", Content: fwkrh.Content{Raw: filler(seed, n)}}
}

func assistant(seed byte, n int) fwkrh.Message {
	return fwkrh.Message{Role: "assistant", Content: fwkrh.Content{Raw: filler(seed, n)}}
}

// reqSeq makes every test request a distinct request id: a thread's outstanding
// set is keyed by request id.
var reqSeq atomic.Uint64

func chatReq(msgs ...fwkrh.Message) *fwksched.InferenceRequest {
	return chatReqID(fmt.Sprintf("req-%d", reqSeq.Add(1)), msgs...)
}

func chatReqID(id string, msgs ...fwkrh.Message) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{
		RequestID: id,
		Headers:   map[string]string{},
		Body:      &fwkrh.InferenceRequestBody{ChatCompletions: &fwkrh.ChatCompletionsRequest{Messages: msgs}},
	}
}

// testRenderer stands in for a chat template: the tools block first, then each
// message's role, tool calls and content, then the generation-prompt suffix
// (stub tokens). The properties the identity depends on hold: an appended message
// extends the token stream, and a mid-session change to the tools block or to an
// earlier message moves it.
func testRenderer(stub int) func(context.Context, *fwksched.InferenceRequest) error {
	return func(_ context.Context, request *fwksched.InferenceRequest) error {
		if request.Body.TokenizedRequest != nil {
			return nil // the producer's own skip path
		}
		ids := testTokenIDs(request.Body, stub)
		if len(ids) == 0 {
			return nil
		}
		request.Body.TokenizedRequest = &fwkrh.TokenizedRequest{
			Prompts:   []fwkrh.PromptTokens{{TokenIDs: ids}},
			CacheSalt: tokenizer.CacheSaltFromBody(request.Body),
		}
		return nil
	}
}

func testTokenIDs(body *fwkrh.InferenceRequestBody, stub int) []uint32 {
	var ids []uint32
	switch {
	case body.ChatCompletions != nil:
		for _, tool := range body.ChatCompletions.Tools {
			ids = append(ids, textTokens(fmt.Sprint(tool))...)
		}
		for _, m := range body.ChatCompletions.Messages {
			ids = append(ids, roleToken(m.Role))
			if len(m.ToolCalls) > 0 {
				ids = append(ids, textTokens(fmt.Sprint(m.ToolCalls))...)
			}
			ids = append(ids, textTokens(m.Content.Raw)...)
		}
	case body.Messages != nil:
		for _, m := range body.Messages.Messages {
			ids = append(ids, roleToken(m.Role))
			ids = append(ids, textTokens(m.Content.Raw)...)
		}
	}
	for i := 0; i < stub && len(ids) > 0; i++ {
		ids = append(ids, 900_000+uint32(i))
	}
	return ids
}

// textTokens renders text as one token per byte, above the role-token range.
func textTokens(s string) []uint32 {
	out := make([]uint32, len(s))
	for i := 0; i < len(s); i++ {
		out[i] = 2000 + uint32(s[i])
	}
	return out
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
	case "tool":
		return 1004
	}
	return 1005 + uint32(len(role))
}

// newTestPluginWith builds the plugin through the Factory and then replaces the
// render backend with the test renderer. The rules under test are about the
// identity, not about a chat template's exact tokens.
//
// Every test config carries a tokenProducer block, because the plugin refuses to
// start without one (F3); the URL is never dialled — the render is replaced below
// — and the warmup it starts gives up against the closed port.
func newTestPluginWith(t *testing.T, params string) *Plugin {
	t.Helper()
	built, err := Factory("thread", json.NewDecoder(strings.NewReader(withTestProducer(t, params))),
		testutils.NewTestHandle(t.Context()))
	require.NoError(t, err)
	p := built.(*Plugin)
	p.tokenize = testRenderer(0)
	return p
}

// testProducerConfig is the render backend every test config carries: a vllm
// block with no reachable server (the render is always substituted).
const testProducerConfig = `{"modelName":"test-model","vllm":{"url":"http://127.0.0.1:1"}}`

// withTestProducer merges that block into a test's parameter object, keeping the
// test's own keys (a test that sets its own tokenProducer keeps it).
func withTestProducer(t *testing.T, params string) string {
	t.Helper()
	merged := map[string]json.RawMessage{"tokenProducer": json.RawMessage(testProducerConfig)}
	if params != "" {
		require.NoError(t, json.Unmarshal([]byte(params), &merged))
		if _, ok := merged["tokenProducer"]; !ok {
			merged["tokenProducer"] = json.RawMessage(testProducerConfig)
		}
	}
	raw, err := json.Marshal(merged)
	require.NoError(t, err)
	return string(raw)
}

func newTestPlugin(t *testing.T) *Plugin {
	t.Helper()
	return newTestPluginWith(t, "")
}

// threadOf runs the header hook and returns the derived id and the fairness id.
func threadOf(t *testing.T, p *Plugin, req *fwksched.InferenceRequest) (threadID, fairnessID string) {
	t.Helper()
	require.NoError(t, p.RequestHeader(context.Background(), req))
	id, ok := fwksched.ReadRequestAttribute[string](req, ThreadIdentityKey)
	if !ok {
		return "", req.FairnessID
	}
	return id, req.FairnessID
}

// finishTurn models a full turn: the request leaves the queue and its response
// completes (the director forces the EndOfStream call even on abort).
func finishTurn(t *testing.T, p *Plugin, req *fwksched.InferenceRequest) {
	t.Helper()
	require.NoError(t, p.PreRequest(context.Background(), req, nil))
	p.ResponseBody(context.Background(), req, &fwkrc.Response{
		EndOfStream:      true,
		TerminationCause: fwkrc.TerminationCauseNatural,
	}, nil)
}

// TestRequestHeader_SetsTheQueueKey: the derived thread id is the request's
// fairness id, so it exists before the request enqueues.
func TestRequestHeader_SetsTheQueueKey(t *testing.T) {
	p := newTestPlugin(t)

	id, fairness := threadOf(t, p, chatReq(user('a', 600)))
	require.NotEmpty(t, id)
	assert.Equal(t, id, fairness, "the thread id must be the flow-control queue key")
	assert.Len(t, id, idHexLen)
	assert.Regexp(t, "^[0-9a-f]+$", id, "thread ids are opaque, short hex")
}

// TestRequestHeader_ExplicitFairnessHeaderWins: an operator override still wins.
func TestRequestHeader_ExplicitFairnessHeaderWins(t *testing.T) {
	p := newTestPlugin(t)
	req := chatReq(user('a', 600))
	req.FairnessID = "client-thread-1"

	id, fairness := threadOf(t, p, req)
	require.NotEmpty(t, id)
	assert.Equal(t, "client-thread-1", fairness, "the explicit id is not overwritten")
}

// TestLifecycle_ContinuationIsTheSameID: a finished turn's successor continues it.
func TestLifecycle_ContinuationIsTheSameID(t *testing.T) {
	p := newTestPlugin(t)

	first := chatReq(user('a', 600))
	id1, _ := threadOf(t, p, first)
	finishTurn(t, p, first)

	second := chatReq(user('a', 600), assistant('b', 40), user('c', 60))
	id2, fairness2 := threadOf(t, p, second)
	assert.Equal(t, id1, id2, "an appended turn continues the thread")
	assert.Equal(t, id1, fairness2)
}

// TestArrivalRecord_IsPublishedForTheScheduler pins the arrival record the
// scheduler consumes: the outcome (continued / forked / new) and the arrival's
// turn number on its thread, as one data key. It is an OUTPUT of the derivation.
func TestArrivalRecord_IsPublishedForTheScheduler(t *testing.T) {
	p := newTestPlugin(t)

	// Turn 1: a new thread.
	first := chatReq(user('a', 600))
	_, _ = threadOf(t, p, first)
	assert.Equal(t, ThreadArrival{Outcome: OutcomeNew, Turn: 1}, arrivalOf(t, first))
	finishTurn(t, p, first)

	// Turn 2 and 3: continuations of the same thread.
	for wantTurn := 2; wantTurn <= 3; wantTurn++ {
		next := chatReq(user('a', 600), assistant('b', 40), user('c', 60))
		_, _ = threadOf(t, p, next)
		assert.Equal(t, ThreadArrival{Outcome: OutcomeMatched, Turn: wantTurn}, arrivalOf(t, next))
		finishTurn(t, p, next)
	}

	// A sibling of a busy tip is a fork, and a fork's turn restarts at 1.
	busy := chatReq(user('z', 600))
	forkedID, _ := threadOf(t, p, busy)
	require.NoError(t, p.PreRequest(context.Background(), busy, nil))
	sibling := chatReq(user('z', 600), user('y', 60))
	siblingID, _ := threadOf(t, p, sibling)
	require.NotEqual(t, forkedID, siblingID)
	assert.Equal(t, ThreadArrival{Outcome: OutcomeForked, Turn: 1}, arrivalOf(t, sibling))

	// The record is declared in Produces(), so a consumer may depend on the key.
	_, declared := p.Produces()[ThreadArrivalKey]
	assert.True(t, declared, "the arrival record must be a declared output")
}

// TestArrivalRecord_DoesNotChangeTheDerivation: the record is published AFTER the
// resolve, so reading it from the request cannot alter any id.
func TestArrivalRecord_DoesNotChangeTheDerivation(t *testing.T) {
	replay := func(drop bool) []string {
		p := newTestPlugin(t)
		var ids []string
		for _, msgs := range [][]fwkrh.Message{
			{user('a', 300)},
			{user('a', 300), assistant('b', 40), user('c', 50)},
			{user('a', 300), assistant('b', 40), user('c', 50), assistant('d', 30), user('e', 20)},
		} {
			req := chatReq(msgs...)
			id, _ := threadOf(t, p, req)
			ids = append(ids, id)
			if drop {
				req.PutAttribute(ThreadArrivalKey, nil)
			}
			finishTurn(t, p, req)
		}
		return ids
	}
	assert.Equal(t, replay(false), replay(true))
}

// arrivalOf reads the published arrival record.
func arrivalOf(t *testing.T, request *fwksched.InferenceRequest) ThreadArrival {
	t.Helper()
	record, ok := fwksched.ReadRequestAttribute[ThreadArrival](request, ThreadArrivalKey)
	require.True(t, ok, "every derived request publishes its arrival record")
	return record
}

// TestLifecycle_ForkWhileBusy: extending a tip that is still in flight forks.
func TestLifecycle_ForkWhileBusy(t *testing.T) {
	p := newTestPlugin(t)

	first := chatReq(user('a', 600))
	id1, _ := threadOf(t, p, first)
	require.NoError(t, p.PreRequest(context.Background(), first, nil))

	// No assistant message after the tip, so H1 cannot prove completion.
	second := chatReq(user('a', 600), user('c', 60))
	id2, _ := threadOf(t, p, second)
	assert.NotEqual(t, id1, id2, "a busy tip's extension is a sibling")
}

// TestH1_ReplyGuard: the request carries the tip's reply, so a busy tip is proven
// complete and the request continues (H1).
func TestH1_ReplyGuard(t *testing.T) {
	p := newTestPlugin(t)

	first := chatReq(user('a', 600))
	id1, _ := threadOf(t, p, first)
	require.NoError(t, p.PreRequest(context.Background(), first, nil))

	second := chatReq(user('a', 600), assistant('b', 40), user('c', 60))
	id2, _ := threadOf(t, p, second)
	assert.Equal(t, id1, id2)
	require.NotNil(t, p.metrics.sticky)
	assert.EqualValues(t, 1, testutil.ToFloat64(p.metrics.sticky))
}

// TestSessionIsAnAttributeNotAQueueKey: the session header is published on its own
// key and never overwrites the thread id.
func TestSessionIsAnAttributeNotAQueueKey(t *testing.T) {
	p := newTestPluginWith(t, `{"sessionHeaders":["x-session-scope"]}`)

	req := chatReq(user('a', 600))
	req.Headers["x-session-scope"] = "session-7"
	id, fairness := threadOf(t, p, req)

	session, ok := fwksched.ReadRequestAttribute[string](req, ThreadSessionKey)
	require.True(t, ok)
	assert.Equal(t, "session-7", session)
	assert.Equal(t, id, fairness, "the thread, not the session, is the queue key")
	assert.NotEqual(t, "session-7", id)

	// A request under another session is a different thread.
	other := chatReq(user('a', 600), assistant('b', 40), user('c', 60))
	other.Headers["x-session-scope"] = "session-8"
	id2, _ := threadOf(t, p, other)
	assert.NotEqual(t, id, id2)
}

// TestToolSetChangeSplitsTheThread is declared class (i): the identity follows the
// engine's tokens, so a mid-session tool-set change (rendered at the head by every
// common template) splits the thread — exactly where the KV cache splits.
func TestToolSetChangeSplitsTheThread(t *testing.T) {
	p := newTestPlugin(t)

	first := chatReq(user('a', 600))
	first.Body.ChatCompletions.Tools = []any{map[string]any{"name": "search"}}
	id1, _ := threadOf(t, p, first)
	finishTurn(t, p, first)

	second := chatReq(user('a', 600), assistant('b', 40), user('c', 60))
	second.Body.ChatCompletions.Tools = []any{map[string]any{"name": "search"}, map[string]any{"name": "task"}}
	id2, _ := threadOf(t, p, second)
	assert.NotEqual(t, id1, id2, "a new tool set is a new token prefix")

	// The same body with the SAME tools continues.
	third := chatReq(user('a', 600), assistant('b', 40), user('c', 60), assistant('d', 30), user('e', 20))
	third.Body.ChatCompletions.Tools = second.Body.ChatCompletions.Tools
	id3, _ := threadOf(t, p, third)
	assert.Equal(t, id2, id3, "with the tool set unchanged the thread continues")
}

// TestRerenderedHistorySplitsTheThread is the other half of class (i): a client
// that drops an earlier turn's reasoning re-renders history, so the tokens before
// the new message move and the thread splits.
func TestRerenderedHistorySplitsTheThread(t *testing.T) {
	p := newTestPlugin(t)

	first := chatReq(user('a', 600), assistant('r', 200))
	id1, _ := threadOf(t, p, first)
	finishTurn(t, p, first)

	// The same conversation with that assistant turn's reasoning stripped.
	stripped := chatReq(user('a', 600), fwkrh.Message{Role: "assistant", Content: fwkrh.Content{Raw: ""}}, user('c', 60))
	id2, _ := threadOf(t, p, stripped)
	assert.NotEqual(t, id1, id2, "re-rendered history is a different token prefix")
}

// TestGenerationPromptStub_InsideThePartialBlockContinues: the tip's END renders
// differently in history (a generation-prompt stub the history drops). With the
// stub inside the tip's partial block the request continues.
func TestGenerationPromptStub_InsideThePartialBlockContinues(t *testing.T) {
	// 1 role token + 140 content tokens + 20 stub tokens = 161: two full blocks,
	// the stub inside the partial block (128..160).
	plantGenerationStub(t, 140, 20, true)
}

// TestGenerationPromptStub_StraddlingABlockEdgeContinues: the same stub planted so
// that it completes the tip's last FULL block (the divergence crosses a block
// edge). A partial-block-only tolerance calls this a false new thread; the
// one-full-block tolerance absorbs it.
func TestGenerationPromptStub_StraddlingABlockEdgeContinues(t *testing.T) {
	// 1 role token + 188 content tokens + 20 stub tokens = 209: three full blocks,
	// the stub's first 4 tokens inside the last full block (128..191).
	plantGenerationStub(t, 188, 20, false)
}

// plantGenerationStub drives the stub fixture end to end: a tip whose render ends
// with a generation-prompt stub, then the next turn of the same conversation,
// whose history does not contain that stub.
func plantGenerationStub(t *testing.T, contentLen, stub int, expectInPartialBlock bool) {
	t.Helper()
	p := newTestPlugin(t)
	p.tokenize = testRenderer(stub)

	tip := chatReq(user('a', contentLen))
	tipIDs := testTokenIDs(tip.Body, stub)
	require.GreaterOrEqual(t, len(tipIDs)/testBlockSize, 2, "the fixture needs a multi-block tip")
	lastFullEnd := (len(tipIDs) / testBlockSize) * testBlockSize

	nextIDs := testTokenIDs(chatReq(user('a', contentLen), assistant('b', 40), user('c', 60)).Body, stub)
	divergence := firstDiff(tipIDs, nextIDs)
	require.Equal(t, 1+contentLen, divergence, "the tip diverges exactly where its stub begins")
	if expectInPartialBlock {
		require.GreaterOrEqual(t, divergence, lastFullEnd, "the stub must lie inside the tip's partial block")
	} else {
		require.GreaterOrEqual(t, divergence, lastFullEnd-testBlockSize, "the stub must lie inside the tip's last full block")
		require.Less(t, divergence, lastFullEnd)
	}

	tipID, _ := threadOf(t, p, tip)
	finishTurn(t, p, tip)

	nextID, _ := threadOf(t, p, chatReq(user('a', contentLen), assistant('b', 40), user('c', 60)))
	assert.Equal(t, tipID, nextID, "an end-of-tip-only difference continues the thread")
}

// TestRenderFailure_StartsItsOwnThread is declared class (iv): a request the
// render cannot describe gets its own thread, and the failure is counted.
func TestRenderFailure_StartsItsOwnThread(t *testing.T) {
	p := newTestPlugin(t)
	p.tokenize = func(context.Context, *fwksched.InferenceRequest) error {
		return fmt.Errorf("render endpoint unavailable")
	}

	first := chatReq(user('a', 600))
	id1, fairness1 := threadOf(t, p, first)
	require.NotEmpty(t, id1)
	assert.Equal(t, id1, fairness1, "a failed render still yields a queue key")
	require.NotNil(t, p.metrics.renderFailure)
	assert.EqualValues(t, 1, testutil.ToFloat64(p.metrics.renderFailure))

	// Two failing requests never share a thread, and neither is continuable.
	second := chatReq(user('a', 600))
	id2, _ := threadOf(t, p, second)
	assert.NotEqual(t, id1, id2)
}

// TestEmptyRender_StartsItsOwnThread: a body that renders to no tokens is a
// render failure too (nothing to key on).
func TestEmptyRender_StartsItsOwnThread(t *testing.T) {
	p := newTestPlugin(t)
	req := &fwksched.InferenceRequest{RequestID: "r-empty", Headers: map[string]string{},
		Body: &fwkrh.InferenceRequestBody{ChatCompletions: &fwkrh.ChatCompletionsRequest{}}}

	id, _ := threadOf(t, p, req)
	require.NotEmpty(t, id)
	assert.EqualValues(t, 1, testutil.ToFloat64(p.metrics.renderFailure))
}

// TestArrivalRendersOnceAndDispatchSkips: the render is RELOCATED from dispatch to
// arrival, not added — exactly one render per request, and the token producer's
// existing skip path makes the dispatch-time call a no-op.
func TestArrivalRendersOnceAndDispatchSkips(t *testing.T) {
	// Count only THIS request's renders: the backend's warmup probes the same
	// server with a "warmup" body.
	var renders atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if strings.Contains(string(body), "aaaa") {
			renders.Add(1)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"token_ids": tokens(1, 3*testBlockSize)})
	}))
	defer srv.Close()

	params := fmt.Sprintf(`{"tokenProducer":{"modelName":"m","vllm":{"url":%q}}}`, srv.URL)
	built, err := Factory("thread", json.NewDecoder(strings.NewReader(params)), testutils.NewTestHandle(t.Context()))
	require.NoError(t, err)
	p := built.(*Plugin)

	req := chatReq(user('a', 600))
	// The renderer receives the wire payload, which a hand-built body carries only
	// once Payload is set.
	req.Body.Payload = fwkrh.PayloadMap{"model": "m", "messages": []any{
		map[string]any{"role": "user", "content": strings.Repeat("a", 600)},
	}}
	_, _ = threadOf(t, p, req)
	require.Equal(t, int64(1), renders.Load(), "the arrival hook renders once")
	require.NotNil(t, req.Body.TokenizedRequest)

	// The dispatch-time token producer finds the tokens and returns early.
	require.NoError(t, p.producer.Tokenize(context.Background(), req))
	assert.Equal(t, int64(1), renders.Load(), "no second render")
}

// TestMultiPromptConcatenatesChains: every prompt counts, so a multi-prompt
// request's identity is the concatenation of the per-prompt chains, and the
// request is counted.
func TestMultiPromptConcatenatesChains(t *testing.T) {
	p := newTestPlugin(t)
	p.tokenize = func(_ context.Context, request *fwksched.InferenceRequest) error {
		request.Body.TokenizedRequest = &fwkrh.TokenizedRequest{Prompts: []fwkrh.PromptTokens{
			{TokenIDs: tokens(1, testBlockSize)},
			{TokenIDs: tokens(2, testBlockSize)},
		}}
		return nil
	}

	req := chatReq(user('a', 10))
	id1, _ := threadOf(t, p, req)
	require.NotNil(t, p.metrics.multiPrompt)
	assert.EqualValues(t, 1, testutil.ToFloat64(p.metrics.multiPrompt))
	assert.Equal(t, 2, p.idx.threads[id1].fullBlocks, "both prompts contribute blocks")

	// A repeat continues it.
	req2 := chatReq(user('a', 10))
	id2, _ := threadOf(t, p, req2)
	assert.Equal(t, id1, id2)
}

// TestDeterminism_TwoReplaysAgree: the same requests in the same order produce the
// same thread ids (the ids are minted from content, not from a clock).
func TestDeterminism_TwoReplaysAgree(t *testing.T) {
	replay := func() []string {
		p := newTestPlugin(t)
		var ids []string
		for i := 0; i < 20; i++ {
			req := chatReq(user('a', 300), assistant('b', 50), user(byte('c'+i%3), 40+i))
			id, _ := threadOf(t, p, req)
			ids = append(ids, id)
			finishTurn(t, p, req)
		}
		return ids
	}
	assert.Equal(t, replay(), replay())
}

// TestDumpState: the snapshot reports the table.
func TestDumpState(t *testing.T) {
	p := newTestPlugin(t)
	_, _ = threadOf(t, p, chatReq(user('a', 600)))

	raw, err := p.DumpState()
	require.NoError(t, err)
	var snap threadTable
	require.NoError(t, json.Unmarshal(raw, &snap))
	assert.Equal(t, 1, snap.Threads)
	assert.Equal(t, 1, snap.Busy, "the arrival marked the thread busy")
	require.Len(t, snap.Rows, 1)
	assert.Equal(t, testBlockSize, snap.BlockSize)
	assert.Greater(t, snap.Rows[0].Tokens, 0)
}

// TestFactory_Config: defaults resolve, invalid values fail closed.
func TestFactory_Config(t *testing.T) {
	p := newTestPlugin(t)
	assert.Equal(t, defaultBlockSizeTokens, p.idx.blockSize)
	assert.True(t, p.forkOnBusy)
	assert.True(t, p.sticky)
	assert.True(t, p.retry)
	assert.Equal(t, defaultMaxThreads, p.idx.maxThreads)
	assert.Equal(t, defaultMarkerTTLSeconds*time.Second, p.markerTTL,
		"the marker TTL is the client request timeout, not the flow queue's 5-minute GC")
	// and that timeout is the campaign harness's own (9000 s) plus headroom: a marker that
	// expires under a LIVE request would let its thread take the next request as a second
	// chain, which is the one thing the identity may not do.
	assert.Greater(t, defaultMarkerTTLSeconds, 9000,
		"the default marker TTL must sit above the harness request timeout")

	// The deprecated arm name still parses.
	built, err := Factory("legacy", json.NewDecoder(strings.NewReader(
		withTestProducer(t, `{"queuedTtlSeconds":900}`))), nil)
	require.NoError(t, err)
	assert.Equal(t, 900*time.Second, built.(*Plugin).markerTTL)

	for name, params := range map[string]string{
		"negative block size": `{"blockSizeTokens":-1}`,
		"negative sweep":      `{"sweepSeconds":-5}`,
		"negative cap":        `{"maxThreads":-1}`,
		"bad json":            `{`,
		"bad token producer":  `{"tokenProducer":{"estimate":{},"modelName":"m"}}`,
	} {
		_, err := Factory("bad", json.NewDecoder(strings.NewReader(params)), nil)
		assert.Error(t, err, name)
	}
}

// TestFactory_FailsClosedWithoutARender: F3 — a plugin whose identity would live in
// byte space (a missing tokenProducer block, or one that selects the estimate
// backend) REFUSES TO START rather than warning and deriving the retired v1 key.
func TestFactory_FailsClosedWithoutARender(t *testing.T) {
	cases := map[string]string{
		"missing block": `{"blockSizeTokens":64}`,
		"empty object":  `{}`,
	}
	for name, params := range cases {
		_, err := Factory("norender", json.NewDecoder(strings.NewReader(params)), nil)
		require.Error(t, err, name)
		assert.Contains(t, err.Error(), "tokenProducer", name)
	}
	// No parameters at all is the same refusal.
	_, err := Factory("norender", nil, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "tokenProducer")

	// A block that parses but selects no renderer: refused after construction.
	for _, params := range []string{
		`{"tokenProducer":{}}`,
		`{"tokenProducer":{"estimate":{}}}`,
	} {
		_, err := Factory("norender", json.NewDecoder(strings.NewReader(params)), nil)
		require.Error(t, err, params)
		assert.Contains(t, err.Error(), "tokenProducer", params)
	}

	// An estimate backend beside a model name is already an invalid producer block.
	_, err = Factory("norender", json.NewDecoder(strings.NewReader(
		`{"tokenProducer":{"estimate":{},"modelName":"m"}}`)), nil)
	require.Error(t, err)
}

// TestFactory_EchoesTheEffectiveConfig: the arm is verifiable from its artifacts —
// the render backend included, since a wrong one silently changes the identity.
func TestFactory_EchoesTheEffectiveConfig(t *testing.T) {
	c := &logCapture{}
	// verbosity 2 IS the default level (logutil.DEFAULT): the echo must appear there.
	ctx := log.IntoContext(t.Context(), funcr.New(func(_, args string) {
		c.mu.Lock()
		defer c.mu.Unlock()
		c.lines = append(c.lines, args)
	}, funcr.Options{Verbosity: logutil.DEFAULT}))
	built, err := Factory("echo", json.NewDecoder(strings.NewReader(withTestProducer(t, `{"logRequests":true}`))),
		testutils.NewTestHandle(ctx))
	require.NoError(t, err)
	require.NotNil(t, built)
	lines := c.snapshot()
	require.NotEmpty(t, lines, "the effective config must be echoed at the default level")
	assert.Contains(t, lines[0], "effective config")
	assert.Contains(t, lines[0], "tokenProducer")
}

// TestFactory_MetricsRegistered: the derivation's counters and the table gauge are
// exported.
func TestFactory_MetricsRegistered(t *testing.T) {
	handle := testutils.NewTestHandle(t.Context())
	built, err := Factory("thread", json.NewDecoder(strings.NewReader(withTestProducer(t, ""))), handle)
	require.NoError(t, err)
	p := built.(*Plugin)
	p.tokenize = testRenderer(0)

	_, _ = threadOf(t, p, chatReq(user('a', 600)))
	_, _ = p.DumpState()

	reg, ok := handle.Metrics().(*prometheus.Registry)
	require.True(t, ok)
	families, err := reg.Gather()
	require.NoError(t, err)
	names := make([]string, 0, len(families))
	for _, f := range families {
		names = append(names, f.GetName())
	}
	for _, want := range []string{
		"llm_d_epp_thread_new_total",
		"llm_d_epp_thread_forks_total",
		"llm_d_epp_thread_records",
		"llm_d_epp_thread_render_failure_total",
		"llm_d_epp_thread_multi_prompt_requests_total",
		"llm_d_epp_thread_render_ns",
		"llm_d_epp_thread_chain_ns",
		"llm_d_epp_thread_resolve_ns",
		"llm_d_epp_thread_sticky_busy_total",
		"llm_d_epp_thread_retry_total",
		"llm_d_epp_thread_explicit_override_total",
	} {
		assert.Contains(t, names, want)
	}
}

// TestExplicitFairnessOverrideIsCounted: a harness that sets the fairness id turns
// the derivation into a no-op, so the bypass is counted.
func TestExplicitFairnessOverrideIsCounted(t *testing.T) {
	p := newTestPlugin(t)
	req := chatReq(user('a', 600))
	req.FairnessID = "client-thread-1"
	_, _ = threadOf(t, p, req)
	require.NotNil(t, p.metrics.overridden)
	assert.EqualValues(t, 1, testutil.ToFloat64(p.metrics.overridden))

	_, _ = threadOf(t, p, chatReq(user('b', 600)))
	assert.EqualValues(t, 1, testutil.ToFloat64(p.metrics.overridden), "no override, no count")
}

// TestRetryIsContainedAndCounted: S2 — an identical duplicate of a still-queued
// turn continues its thread, and the counter names how much of the fork rate was
// retries.
func TestRetryIsContainedAndCounted(t *testing.T) {
	p := newTestPlugin(t)

	id1, _ := threadOf(t, p, chatReqID("dup", user('a', 600))) // arrival only: queued
	id2, _ := threadOf(t, p, chatReqID("dup-2", user('a', 600)))
	assert.Equal(t, id1, id2, "an identical duplicate of a queued turn is a retry")
	require.NotNil(t, p.metrics.retried)
	assert.EqualValues(t, 1, testutil.ToFloat64(p.metrics.retried))
}

// TestLogRequests: with the instrument on, one JSON line per request carries the
// scope, the derived id and the outcome; with it off nothing is printed.
func TestLogRequests(t *testing.T) {
	on := newTestPluginWith(t, `{"sessionHeaders":["x-session-scope"],"logRequests":true}`)
	req := chatReq(user('a', 600))
	req.Headers["x-session-scope"] = "session-3"
	var id string
	lines := captureLogs(t, on, func() { id, _ = threadOf(t, on, req) })
	require.Len(t, lines, 1, "one line per request, and only when the instrument is on")
	out := lines[0]
	assert.Contains(t, out, `"scope"="session-3"`)
	assert.Contains(t, out, `"derived"="`+id+`"`)
	assert.Contains(t, out, `"outcome"="new"`)

	off := newTestPlugin(t)
	assert.Empty(t, captureLogs(t, off, func() { _, _ = threadOf(t, off, chatReq(user('a', 600))) }),
		"the instrument is off by default")
}

// TestLogHeadersKeepResolveOutcomeIdentical: the header copies are a pure observer.
func TestLogHeadersKeepResolveOutcomeIdentical(t *testing.T) {
	run := func(params string) []string {
		p := newTestPluginWith(t, params)
		var ids []string
		for _, msgs := range [][]fwkrh.Message{
			{user('a', 300)},
			{user('a', 300), assistant('b', 40), user('c', 50)},
			{user('z', 300)},
		} {
			req := chatReq(msgs...)
			req.Headers["x-correlation-id"] = "thread-9"
			id, _ := threadOf(t, p, req)
			ids = append(ids, id)
			finishTurn(t, p, req)
		}
		return ids
	}
	plain := run(`{}`)
	logged := run(`{"logRequests":true,"logHeaders":true}`)
	assert.Equal(t, plain, logged)
}

// TestLogHeaderNamesConfigured: the whitelist takes configured extras.
func TestLogHeaderNamesConfigured(t *testing.T) {
	p := newTestPluginWith(t, `{"logRequests":true,"logHeaders":true,"logHeaderNames":["X-Thread-Id"]}`)
	req := chatReq(user('a', 600))
	req.Headers["x-thread-id"] = "t-1"

	out := captureLogs(t, p, func() { _, _ = threadOf(t, p, req) })
	require.Len(t, out, 1)
	assert.Contains(t, out[0], `"header_x-thread-id"="t-1"`)
}

// logCapture collects what a test's plugin LOGS (the startup echo at the default level,
// the per-request instrument at V(DEBUG)) — the shipped code logs, it does not print.
type logCapture struct {
	mu    sync.Mutex
	lines []string
}

func (c *logCapture) sink() *logCapture { return c }

// attach points the plugin's logger at the capture, at V(DEBUG) so a test can assert on
// the per-request lines too (the DEFAULT level would drop them, which is the point).
func (c *logCapture) attach(p *Plugin) {
	p.logger = funcr.New(func(_, args string) {
		c.mu.Lock()
		defer c.mu.Unlock()
		c.lines = append(c.lines, args)
	}, funcr.Options{Verbosity: 10}).WithName(PluginType)
}

func (c *logCapture) snapshot() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]string(nil), c.lines...)
}

// captureLogs runs fn with a capturing logger attached to p.
func captureLogs(t *testing.T, p *Plugin, fn func()) []string {
	t.Helper()
	c := &logCapture{}
	c.attach(p)
	fn()
	return c.snapshot()
}

// firstDiff is the index of the first differing token (or the shorter length).
func firstDiff(a, b []uint32) int {
	for i := 0; i < len(a) && i < len(b); i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return min(len(a), len(b))
}
