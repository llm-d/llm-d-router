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

// THE DIFFERENTIAL IDENTITY DUMP.
//
// A corpus of real multi-turn requests is replayed through RequestHeader, in arrival order,
// against ONE deterministic fake /render server. For every request the dump records:
//
//	the exact /render request BODY the plugin sent,
//	the outcome (continue / fork / new),
//	the derived thread id,
//	the published thread blocks.
//
// Two trees run the same harness over the same corpus and the two dumps must be IDENTICAL. That
// is a stronger claim than "the tests pass": it pins the render INPUT (so routing the render
// through upstream's token producer changed nothing the engine would see) and the identity
// (so removing session scoping left the unscoped derivation byte-identical).
//
// The fake server answers the body it RECEIVED, not the request it expected: its token ids are
// derived from the body's own bytes, so any difference in what is posted shows up as a different
// chain and therefore a different thread id. The dump would catch it.
//
//	THREADID_DIFF_CORPUS=corpus.jsonl THREADID_DIFF_OUT=a.jsonl \
//	  go test ./pkg/epp/framework/plugins/threadidentity/ -run TestDifferentialIdentityDump -v
//
// With no env set it replays the committed fixture (testdata/differential/corpus.jsonl).

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

// diffCorpusLine is one replayed request: a real chat body from the corpus.
type diffCorpusLine struct {
	Index int             `json:"i"`
	Conv  string          `json:"conv"`
	Turn  int             `json:"turn"`
	Body  json.RawMessage `json:"body"`
}

// diffDumpLine is one request's observable behaviour. Every field is compared across trees.
type diffDumpLine struct {
	Index    int      `json:"i"`
	Conv     string   `json:"conv"`
	Turn     int      `json:"turn"`
	Renders  []string `json:"renders"` // the exact bodies posted to /render, in order
	Outcome  string   `json:"outcome"`
	ThreadID string   `json:"id"`
	Blocks   []uint64 `json:"blocks"`
	Tokens   int      `json:"tokens"`
	Err      string   `json:"err,omitempty"`
}

// deterministicRender is the fake /render backend. It renders the SHAPE a chat template
// would: one role token per message, then the message's content as one token per four bytes.
// So an appended message EXTENDS the previous turn's stream (a continuation), and a changed
// earlier message moves it — the two properties the identity depends on. The tokens are a pure
// function of the body the plugin posted, so the dump fingerprints what was sent.
func deterministicRender(t *testing.T) (*httptest.Server, *[]string) {
	t.Helper()
	var seen []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw := new(bytes.Buffer)
		_, _ = raw.ReadFrom(r.Body)
		// The backend's own WARMUP (a constant "warmup" turn, issued once at construction and
		// racing with the first request) is answered but not recorded: it carries no request's
		// tokens, and its timing is a scheduler artifact, not a behavioural difference.
		if !strings.Contains(raw.String(), `"warmup"`) {
			seen = append(seen, raw.String())
		}

		var body struct {
			Messages []struct {
				Role    string          `json:"role"`
				Content json.RawMessage `json:"content"`
			} `json:"messages"`
		}
		ids := []uint32{}
		if err := json.Unmarshal(raw.Bytes(), &body); err == nil {
			for _, m := range body.Messages {
				ids = append(ids, renderRoleToken(m.Role))
				var text string
				_ = json.Unmarshal(m.Content, &text)
				for i := 0; i < len(text); i += 4 {
					h := fnv.New32a()
					_, _ = h.Write([]byte(m.Role))
					_, _ = h.Write([]byte{0})
					_, _ = h.Write([]byte(text[i:min(i+4, len(text))]))
					ids = append(ids, 1000+h.Sum32()%900000)
				}
			}
		}
		if len(ids) == 0 {
			ids = []uint32{1}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"token_ids": ids})
	}))
	t.Cleanup(srv.Close)
	return srv, &seen
}

// renderRoleToken gives each role its own token, above the content range.
func renderRoleToken(role string) uint32 {
	switch role {
	case "system":
		return 900001
	case "assistant":
		return 900002
	case "user":
		return 900003
	default:
		return 900004
	}
}

// TestDifferentialIdentityDump replays the corpus and writes the dump.
func TestDifferentialIdentityDump(t *testing.T) {
	corpusPath := os.Getenv("THREADID_DIFF_CORPUS")
	if corpusPath == "" {
		corpusPath = filepath.Join("testdata", "differential", "corpus.jsonl.gz")
	}
	outPath := os.Getenv("THREADID_DIFF_OUT")
	if outPath == "" {
		t.Skip("THREADID_DIFF_OUT not set: the differential dump is a two-tree comparison harness")
	}

	corpus, err := os.Open(corpusPath)
	require.NoError(t, err, "corpus %s", corpusPath)
	defer corpus.Close()
	var src io.Reader = corpus
	if strings.HasSuffix(corpusPath, ".gz") {
		zr, zerr := gzip.NewReader(corpus)
		require.NoError(t, zerr, "corpus %s is not gzip", corpusPath)
		defer zr.Close()
		src = zr
	}

	srv, posted := deterministicRender(t)
	params := fmt.Sprintf(`{"blockSizeTokens":64,"stickyBusyGuard":false,"tokenProducer":{"modelName":"mock-model","vllm":{"url":%q,"timeout":"30s"}}}`,
		srv.URL)
	built, err := Factory("thread", json.NewDecoder(strings.NewReader(params)), testutils.NewTestHandle(context.Background()))
	require.NoError(t, err)
	p := built.(*Plugin)
	// a fixed clock: the same request must land at the same time on both trees
	at := time.Unix(1793221981, 0).UTC()
	p.now = func() time.Time { return at }

	parser := openai.NewOpenAIParser()
	out, err := os.Create(outPath)
	require.NoError(t, err)
	defer out.Close()
	w := bufio.NewWriter(out)
	defer w.Flush()
	enc := json.NewEncoder(w)

	sc := bufio.NewScanner(src)
	sc.Buffer(make([]byte, 1<<20), 64<<20)
	requests := 0
	for sc.Scan() {
		line := sc.Bytes()
		if len(bytes.TrimSpace(line)) == 0 {
			continue
		}
		var c diffCorpusLine
		require.NoError(t, json.Unmarshal(line, &c))

		postedBefore := len(*posted)
		headers := map[string]string{":path": "/v1/chat/completions"}
		req := &fwksched.InferenceRequest{RequestID: fmt.Sprintf("r%d", c.Index), Headers: headers}
		parsed, perr := parser.ParseRequest(context.Background(), c.Body, headers)
		dump := diffDumpLine{Index: c.Index, Conv: c.Conv, Turn: c.Turn}
		if perr != nil {
			dump.Err = "parse: " + perr.Error()
		} else {
			req.Body = parsed.Body
			if err := p.RequestHeader(context.Background(), req); err != nil {
				dump.Err = "requestheader: " + err.Error()
			}
			if id, ok := fwksched.ReadRequestAttribute[string](req, ThreadIdentityKey); ok {
				dump.ThreadID = id
			}
			if arrival, ok := fwksched.ReadRequestAttribute[ThreadArrival](req, ThreadArrivalKey); ok {
				dump.Outcome = string(arrival.Outcome)
			}
			if blocks, ok := fwksched.ReadRequestAttribute[[]PromptBlocks](req, ThreadBlocksKey); ok {
				for _, b := range blocks {
					for _, k := range b.Keys {
						dump.Blocks = append(dump.Blocks, uint64(k))
					}
				}
			}
			if req.Body != nil && req.Body.TokenizedRequest != nil {
				for _, pr := range req.Body.TokenizedRequest.Prompts {
					dump.Tokens += len(pr.TokenIDs)
				}
			}
			_ = p.PreRequest(context.Background(), req, nil)
			// HALF the turns stay in flight, so the next arrival on the same conversation meets
			// a BUSY tip (a fork); the other half settle first (a continuation). Which half is a
			// function of the turn number, so both trees replay the same sequence.
			if c.Turn%2 == 0 {
				p.ResponseBody(context.Background(), req, &fwkrc.Response{EndOfStream: true, TerminationCause: fwkrc.TerminationCauseNatural}, nil)
			}
		}
		dump.Renders = append([]string(nil), (*posted)[postedBefore:]...)
		require.NoError(t, enc.Encode(dump))
		requests++
	}
	require.NoError(t, sc.Err())
	t.Logf("[differential] %d requests replayed from %s -> %s", requests, corpusPath, outPath)
}
