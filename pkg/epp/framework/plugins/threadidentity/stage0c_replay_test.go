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

// STAGE 0C (v2) — replay a recorded campaign window through the arrival hook with
// the ENGINE's own tokens.
//
// The v2 identity is a token-space key, so an offline replay needs the render that
// the engine would perform. STAGE0C_TOKENS is a JSONL corpus of {i, n, in_tokens,
// tokens} produced offline with the model's chat template (transformers
// apply_chat_template, the same Jinja template vLLM applies); the replay injects
// those tokens in place of the render and VALIDATES them against the recorded
// in_tokens, so a template that drifted cannot silently pass.
//
// Each request is parsed by the PRODUCTION parser (parsers/openai) and pushed
// through the plugin's real hooks at the recorded arrival / dispatch / settle
// times, with the clock injected.
//
//	STAGE0C_WINDOW=/tmp/stage0c/a0.jsonl \
//	STAGE0C_TOKENS=/tmp/biv2/a0.qwen3.tokens.jsonl \
//	STAGE0C_OUT=/tmp/biv2/v2-a0.jsonl \
//	  go test ./pkg/epp/framework/plugins/threadidentity/ -run TestStage0CReplay -v -timeout 3h
//
// Env: STAGE0C_UNSCOPED=1 (send no session header), STAGE0C_RETRY_RULE=0 (the
// campaign setting), STAGE0C_SCOPE=session (scope by session id instead of root; "thread" is still accepted).
//
// Nothing here touches the cluster, a GPU, or the network.

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"testing"
	"time"

	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"

	testutils "github.com/llm-d/llm-d-router/test/utils"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
)

// stage0cScopeHeader is the header the replay uses as the client session id. It is
// NOT one of the built-in provider headers, so the plugin reads it explicitly.
const stage0cScopeHeader = "x-session-scope"

// stage0cProducer is the render backend the replayed arm must declare: the plugin
// refuses to start without one (F3). It is never dialled — the replay substitutes
// the recorded token corpus for the render.
const stage0cProducer = `{"modelName":"stage0c","vllm":{"url":"http://127.0.0.1:1"}}`

// stage0cMaxTokenDelta is how far the offline render may sit from the recorded
// in_tokens. The corpus renders with a constant offset (the template's
// generation-prompt suffix); anything larger means the template is not the engine's.
const stage0cMaxTokenDelta = 2

// stage0cLine is one prepared request (stage0c_prep.py).
type stage0cLine struct {
	I       int             `json:"i"`
	Req     string          `json:"req"`
	Conv    string          `json:"conv"`
	Turn    int             `json:"turn"`
	GT      string          `json:"gt"`
	GTRoot  string          `json:"gt_root"`
	Kind    string          `json:"kind"`
	StartNs int64           `json:"start_ns"`
	AckNs   int64           `json:"ack_ns"`
	EndNs   int64           `json:"end_ns"`
	InTok   int             `json:"in_tokens"`
	Body    json.RawMessage `json:"body"`
	Off     int64           `json:"-"`
	Len     int             `json:"-"`
}

// stage0cResult is one request's outcome, keyed by the window index.
type stage0cResult struct {
	I       int    `json:"i"`
	Req     string `json:"req"`
	ID      string `json:"id"`
	Parse   bool   `json:"parseError"`
	Outcome string `json:"outcome,omitempty"`
	Turn    int    `json:"turn"`
	Tokens  int    `json:"tokens"`
	Blocks  int    `json:"blocks"`
}

// tokenCorpus injects the offline render's tokens, reading each request's line on
// demand so a 900 MB corpus never has to fit in memory.
type tokenCorpus struct {
	f       *os.File
	offsets map[string]tokenOffset
}

type tokenOffset struct {
	off int64
	len int
}

func openTokenCorpus(t *testing.T, path string) *tokenCorpus {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open token corpus: %v", err)
	}
	t.Cleanup(func() { _ = f.Close() })

	c := &tokenCorpus{f: f, offsets: make(map[string]tokenOffset)}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 512<<20)
	var offset int64
	for sc.Scan() {
		raw := sc.Bytes()
		var head struct {
			I int `json:"i"`
		}
		if err := json.Unmarshal(raw, &head); err != nil {
			t.Fatalf("bad token corpus line at offset %d: %v", offset, err)
		}
		c.offsets[fmt.Sprintf("r%d", head.I)] = tokenOffset{off: offset, len: len(raw)}
		offset += int64(len(raw)) + 1
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("scan token corpus: %v", err)
	}
	return c
}

// tokenize is the arrival hook's render, served from the corpus.
func (c *tokenCorpus) tokenize(_ context.Context, request *fwksched.InferenceRequest) error {
	if request.Body.TokenizedRequest != nil {
		return nil
	}
	at, ok := c.offsets[request.RequestID]
	if !ok {
		return fmt.Errorf("no recorded tokens for request %q", request.RequestID)
	}
	buf := make([]byte, at.len)
	if _, err := c.f.ReadAt(buf, at.off); err != nil {
		return fmt.Errorf("read tokens for %q: %w", request.RequestID, err)
	}
	var rec struct {
		Tokens   []uint32 `json:"tokens"`
		InTokens int      `json:"in_tokens"`
	}
	if err := json.Unmarshal(buf, &rec); err != nil {
		return fmt.Errorf("decode tokens for %q: %w", request.RequestID, err)
	}
	// The cross-check needs a recorded count. Some prepared windows carry no
	// in_tokens (the 0B prep did not record one), so the check is skipped there
	// rather than passing trivially — the render is still the engine's template
	// applied to the real body.
	if delta := len(rec.Tokens) - rec.InTokens; rec.InTokens > 0 && (delta > stage0cMaxTokenDelta || delta < -stage0cMaxTokenDelta) {
		return fmt.Errorf("render drifted from the engine for %q: %d rendered, %d recorded",
			request.RequestID, len(rec.Tokens), rec.InTokens)
	}
	request.Body.TokenizedRequest = &fwkrequest.TokenizedRequest{
		Prompts: []fwkrequest.PromptTokens{{TokenIDs: rec.Tokens}},
	}
	return nil
}

func TestStage0CReplay(t *testing.T) {
	window := os.Getenv("STAGE0C_WINDOW")
	if window == "" {
		t.Skip("STAGE0C_WINDOW not set: stage-0c replay is an offline validation harness")
	}
	corpusPath := os.Getenv("STAGE0C_TOKENS")
	if corpusPath == "" {
		t.Skip("STAGE0C_TOKENS not set: the v2 identity needs the engine's tokens")
	}
	out := os.Getenv("STAGE0C_OUT")
	if out == "" {
		out = window + ".v2.jsonl"
	}
	unscoped := os.Getenv("STAGE0C_UNSCOPED") == "1"

	ctx := context.Background()
	f, err := os.Open(window)
	if err != nil {
		t.Fatalf("open window: %v", err)
	}
	defer f.Close()

	// Pass 1 — metadata + byte offsets only (a WEKA window is ~850 MB of bodies).
	start := time.Now()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 256<<20)
	var (
		lines  []*stage0cLine
		offset int64
	)
	for sc.Scan() {
		raw := sc.Bytes()
		l := &stage0cLine{Off: offset, Len: len(raw)}
		if err := json.Unmarshal(raw, l); err != nil {
			t.Fatalf("bad window line at offset %d: %v", offset, err)
		}
		lines = append(lines, l)
		offset += int64(len(raw)) + 1
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("scan window: %v", err)
	}
	t.Logf("[stage0c] window %s: %d requests indexed in %s (unscoped=%v)",
		window, len(lines), time.Since(start).Round(time.Millisecond), unscoped)

	// Events in wall-clock order: a settle precedes an arrival at the same instant.
	const (
		evSettle = iota
		evArrive
		evDispatch
	)
	type event struct {
		ts   int64
		kind int
		idx  int
	}
	events := make([]event, 0, 3*len(lines))
	for idx, l := range lines {
		events = append(events,
			event{l.StartNs, evArrive, idx},
			event{l.AckNs, evDispatch, idx},
			event{l.EndNs, evSettle, idx})
	}
	sort.Slice(events, func(i, j int) bool {
		if events[i].ts != events[j].ts {
			return events[i].ts < events[j].ts
		}
		if events[i].kind != events[j].kind {
			return events[i].kind < events[j].kind
		}
		return events[i].idx < events[j].idx
	})

	// The plugin, configured as the stage-1 arm is, except that the session header
	// is the replay's own scope header and the render is the recorded corpus.
	retryRule := "true"
	if os.Getenv("STAGE0C_RETRY_RULE") == "0" {
		retryRule = "false"
	}
	handle := testutils.NewTestHandle(ctx)
	built, err := Factory("stage0c", json.NewDecoder(bytes.NewReader(
		[]byte(`{"sessionHeaders":["`+stage0cScopeHeader+`"],"retryRule":`+retryRule+`,"tokenProducer":`+stage0cProducer+`}`))), handle)
	if err != nil {
		t.Fatalf("build plugin: %v", err)
	}
	p := built.(*Plugin)
	var clock time.Time
	p.now = func() time.Time { return clock }
	p.tokenize = openTokenCorpus(t, corpusPath).tokenize

	parser := openai.NewOpenAIParser()
	reqs := make([]*fwksched.InferenceRequest, len(lines))
	results := make([]stage0cResult, len(lines))
	for i, l := range lines {
		results[i] = stage0cResult{I: l.I, Req: l.Req}
	}

	var parseErrors, noID int
	for _, ev := range events {
		l := lines[ev.idx]
		clock = time.Unix(0, ev.ts)

		switch ev.kind {
		case evArrive:
			buf := make([]byte, l.Len)
			if _, err := f.ReadAt(buf, l.Off); err != nil {
				t.Fatalf("read body %d: %v", l.I, err)
			}
			var wire struct {
				Body json.RawMessage `json:"body"`
			}
			if err := json.Unmarshal(buf, &wire); err != nil {
				t.Fatalf("decode line %d: %v", l.I, err)
			}
			headers := map[string]string{":path": "/v1/chat/completions"}
			if !unscoped {
				if s := os.Getenv("STAGE0C_SCOPE"); s == "session" || s == "thread" { // "thread" is the pre-rename value
					headers[stage0cScopeHeader] = l.GT
				} else {
					headers[stage0cScopeHeader] = l.GTRoot
				}
			}
			req := &fwksched.InferenceRequest{RequestID: l.Req, Headers: headers}
			parsed, err := parser.ParseRequest(ctx, wire.Body, headers)
			if err != nil {
				parseErrors++
				results[ev.idx].Parse = true
			} else {
				req.Body = parsed.Body
			}
			reqs[ev.idx] = req

			matched0, forked0, new0 := counterValue(p.metrics.matched), counterValue(p.metrics.forked), counterValue(p.metrics.fresh)
			if err := p.RequestHeader(ctx, req); err != nil {
				t.Fatalf("RequestHeader %d: %v", l.I, err)
			}
			switch {
			case counterValue(p.metrics.matched) > matched0:
				results[ev.idx].Outcome = "matched"
			case counterValue(p.metrics.forked) > forked0:
				results[ev.idx].Outcome = "forked"
			case counterValue(p.metrics.fresh) > new0:
				results[ev.idx].Outcome = "new"
			default:
				results[ev.idx].Outcome = "none"
			}
			id, _ := fwksched.ReadRequestAttribute[string](req, ThreadIdentityKey)
			if id == "" {
				noID++
			}
			results[ev.idx].ID = id
			if arrival, ok := fwksched.ReadRequestAttribute[ThreadArrival](req, ThreadArrivalKey); ok {
				results[ev.idx].Turn = arrival.Turn
			}
			results[ev.idx].Tokens = tokenizedTokens(req)
			results[ev.idx].Blocks = tokenizedBlocks(p, req)

		case evDispatch:
			if req := reqs[ev.idx]; req != nil {
				if err := p.PreRequest(ctx, req, nil); err != nil {
					t.Fatalf("PreRequest %d: %v", l.I, err)
				}
			}
		case evSettle:
			if req := reqs[ev.idx]; req != nil {
				p.ResponseBody(ctx, req, &fwkrc.Response{
					EndOfStream:      true,
					TerminationCause: fwkrc.TerminationCauseNatural,
				}, nil)
				req.Body = nil // keep memory flat on a 13k-request window
				reqs[ev.idx] = nil
			}
		}
	}

	t.Logf("[stage0c] %d requests, %d threads, %d parse errors, %d no-id",
		len(lines), p.idx.Len(), parseErrors, noID)
	t.Logf("[stage0c] counters: matched=%v forked=%v new=%v sticky=%v retry=%v renderFailure=%v multiPrompt=%v overridden=%v",
		counterValue(p.metrics.matched), counterValue(p.metrics.forked),
		counterValue(p.metrics.fresh), counterValue(p.metrics.sticky),
		counterValue(p.metrics.retried), counterValue(p.metrics.renderFailure),
		counterValue(p.metrics.multiPrompt), counterValue(p.metrics.overridden))

	w, err := os.Create(out)
	if err != nil {
		t.Fatalf("create output: %v", err)
	}
	defer w.Close()
	bw := bufio.NewWriter(w)
	enc := json.NewEncoder(bw)
	for i := range results {
		if err := enc.Encode(results[i]); err != nil {
			t.Fatalf("write result: %v", err)
		}
	}
	if err := bw.Flush(); err != nil {
		t.Fatalf("flush: %v", err)
	}
	t.Logf("[stage0c] wrote %s", out)
}

// tokenizedTokens is the request's total token count as the identity sees it.
func tokenizedTokens(request *fwksched.InferenceRequest) int {
	total := 0
	if request.Body == nil || request.Body.TokenizedRequest == nil {
		return 0
	}
	for _, p := range request.Body.TokenizedRequest.Prompts {
		total += len(p.TokenIDs)
	}
	return total
}

// tokenizedBlocks is the request's full-block count.
func tokenizedBlocks(p *Plugin, request *fwksched.InferenceRequest) int {
	if request.Body == nil {
		return 0
	}
	return identityChain(promptBlocks(context.Background(), request, p.blockSize), p.blockSize).fullBlocks
}

// counterValue reads one counter (nil-safe: a plugin built without a handle has
// unregistered metrics).
func counterValue(c prometheus.Counter) float64 {
	if c == nil {
		return -1
	}
	return testutil.ToFloat64(c)
}
