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

package steps

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/kv"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

const (
	pdTestRequestID = "req-1"
	pdTestPrefill   = "10.0.3.7:8000"
	// pdTestTimeout bounds every wait so a wrong implementation fails instead
	// of hanging the test run.
	pdTestTimeout = 5 * time.Second
)

// Roles of the requests the fake gateway receives.
const (
	roleReserve = "reserve"
	rolePrefill = "prefill"
	roleDecode  = "decode"
)

// pdRequest is one request the fake gateway received.
type pdRequest struct {
	header http.Header
	path   string
	body   map[string]any
}

// pdGateway fakes the gateway in front of an SGLang deployment. It sorts each
// request by role (reserve, prefill, decode) and hands it to that role's
// handler. The default handlers behave like SGLang: the ask is answered with
// pdTestPrefill, prefill completes only after decode has arrived, and decode
// completes only after prefill has arrived.
type pdGateway struct {
	t        *testing.T
	mu       sync.Mutex
	requests map[string][]pdRequest

	prefillArrived chan struct{}
	decodeArrived  chan struct{}
	prefillOnce    sync.Once
	decodeOnce     sync.Once

	reserve http.HandlerFunc
	prefill http.HandlerFunc
	decode  http.HandlerFunc
}

func newPDGateway(t *testing.T) *pdGateway {
	g := &pdGateway{
		t:              t,
		requests:       map[string][]pdRequest{},
		prefillArrived: make(chan struct{}),
		decodeArrived:  make(chan struct{}),
	}
	g.reserve = func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set(routing.ReservedEndpointHeader, pdTestPrefill)
	}
	g.prefill = func(w http.ResponseWriter, r *http.Request) {
		if !g.wait(r, g.decodeArrived) {
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []any{}})
	}
	g.decode = func(w http.ResponseWriter, r *http.Request) {
		if !g.wait(r, g.prefillArrived) {
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{"message": map[string]any{"role": "assistant", "content": "done"}}},
		})
	}
	return g
}

// wait blocks until ch closes. It returns false when the request is cancelled
// or the wait times out.
func (g *pdGateway) wait(r *http.Request, ch chan struct{}) bool {
	select {
	case <-ch:
		return true
	case <-r.Context().Done():
		return false
	case <-time.After(pdTestTimeout):
		g.t.Errorf("%s request waited %s for its peer: the requests were not in flight together", role(r), pdTestTimeout)
		return false
	}
}

func role(r *http.Request) string {
	switch {
	case routing.HasPreference(map[string]string{routing.PreferHeader: r.Header.Get("Prefer")}, routing.PreferReserveEndpoint):
		return roleReserve
	case r.Header.Get(routing.PrefillPinHeader) != "":
		return rolePrefill
	case r.Header.Get(gateway.EPPProfileHeader) == gateway.PhaseDecode:
		return roleDecode
	default:
		return "unknown"
	}
}

func (g *pdGateway) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	raw, _ := io.ReadAll(r.Body)
	var body map[string]any
	_ = json.Unmarshal(raw, &body)
	name := role(r)
	g.mu.Lock()
	g.requests[name] = append(g.requests[name], pdRequest{header: r.Header.Clone(), path: r.URL.Path, body: body})
	g.mu.Unlock()

	switch name {
	case roleReserve:
		g.reserve(w, r)
	case rolePrefill:
		g.prefillOnce.Do(func() { close(g.prefillArrived) })
		g.prefill(w, r)
	case roleDecode:
		g.decodeOnce.Do(func() { close(g.decodeArrived) })
		g.decode(w, r)
	default:
		g.t.Errorf("unexpected request %s %v", r.URL.Path, r.Header)
		w.WriteHeader(http.StatusInternalServerError)
	}
}

// only returns the single request of role, failing when there is not exactly one.
func (g *pdGateway) only(role string) pdRequest {
	g.t.Helper()
	g.mu.Lock()
	defer g.mu.Unlock()
	if len(g.requests[role]) != 1 {
		g.t.Fatalf("%s requests = %d, want 1", role, len(g.requests[role]))
	}
	return g.requests[role][0]
}

func (g *pdGateway) count(role string) int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return len(g.requests[role])
}

// waitClosed fails the test when ch is not closed within pdTestTimeout.
func waitClosed(t *testing.T, ch <-chan struct{}, msg string) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(pdTestTimeout):
		t.Fatal(msg)
	}
}

// headerRecorder records whether the step wrote anything to the client.
type headerRecorder struct {
	*httptest.ResponseRecorder
	wroteHeader bool
}

func (r *headerRecorder) WriteHeader(code int) {
	r.wroteHeader = true
	r.ResponseRecorder.WriteHeader(code)
}

func (r *headerRecorder) Write(b []byte) (int, error) {
	r.wroteHeader = true
	return r.ResponseRecorder.Write(b)
}

func newPDStep(t *testing.T, url string, params map[string]any) pipeline.Step {
	t.Helper()
	merged := map[string]any{ParamKVConnector: kv.SGLang}
	for k, v := range params {
		merged[k] = v
	}
	step, err := NewPrefillDecodeStep(gateway.New(config.GatewayConfig{Address: url}), merged)
	if err != nil {
		t.Fatalf("NewPrefillDecodeStep: %v", err)
	}
	return step
}

func newPDRequest(path string, body map[string]any) (*pipeline.RequestContext, *headerRecorder) {
	rec := &headerRecorder{ResponseRecorder: httptest.NewRecorder()}
	return &pipeline.RequestContext{
		RequestID:    pdTestRequestID,
		OriginalPath: path,
		OriginalHeaders: http.Header{
			"Prefer":               {"if-available"},
			"X-Prefill-Pin":        {"10.9.9.9:8000"},
			"X-Data-Parallel-Rank": {"3"},
			"Authorization":        {"Bearer t"},
		},
		Model:          "m",
		Stream:         body["stream"] == true,
		Body:           body,
		ResponseWriter: rec,
	}, rec
}

func chatBody() map[string]any {
	return map[string]any{
		"model":                  "m",
		"messages":               []any{map[string]any{"role": "user", "content": "hi"}},
		"max_tokens":             float64(64),
		"routed_dp_rank":         float64(1),
		"data_parallel_rank":     float64(1),
		"disagg_prefill_dp_rank": float64(1),
	}
}

func TestPrefillDecodeStep_SendsBothRequestsTogether(t *testing.T) {
	g := newPDGateway(t)
	srv := httptest.NewServer(g)
	defer srv.Close()

	reqCtx, rec := newPDRequest(reqcommon.PathChatCompletions, chatBody())
	if err := newPDStep(t, srv.URL, nil).Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("Execute: %v", err)
	}

	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "done") {
		t.Fatalf("client got %d %q, want the decode answer", rec.Code, rec.Body.String())
	}

	reserve, prefill, decode := g.only(roleReserve), g.only(rolePrefill), g.only(roleDecode)

	// Each request has its own id, so EPP per-request state does not collide.
	for _, tc := range []struct {
		name   string
		req    pdRequest
		id     string
		prof   string
		prefer string
		pin    string
	}{
		{"reserve", reserve, pdTestRequestID + reserveRequestIDSuffix, gateway.PhasePrefill, routing.PreferReserveEndpoint, ""},
		{"prefill", prefill, pdTestRequestID + prefillRequestIDSuffix, gateway.PhasePrefill, "", pdTestPrefill},
		{"decode", decode, pdTestRequestID, gateway.PhaseDecode, "", ""},
	} {
		if got := tc.req.header.Get(reqcommon.RequestIDHeaderKey); got != tc.id {
			t.Errorf("%s: x-request-id = %q, want %q", tc.name, got, tc.id)
		}
		if got := tc.req.header.Get(gateway.EPPProfileHeader); got != tc.prof {
			t.Errorf("%s: EPP-Profile = %q, want %q", tc.name, got, tc.prof)
		}
		if got := tc.req.header.Get("Prefer"); got != tc.prefer {
			t.Errorf("%s: Prefer = %q, want %q (the client's copy is dropped)", tc.name, got, tc.prefer)
		}
		if got := tc.req.header.Get(routing.PrefillPinHeader); got != tc.pin {
			t.Errorf("%s: %s = %q, want %q (the client's copy is dropped)", tc.name, routing.PrefillPinHeader, got, tc.pin)
		}
		if got := tc.req.header.Get("X-Data-Parallel-Rank"); got != "" {
			t.Errorf("%s: client X-Data-Parallel-Rank was forwarded: %q", tc.name, got)
		}
		if got := tc.req.header.Get("Authorization"); got != "Bearer t" {
			t.Errorf("%s: Authorization = %q, want the client's", tc.name, got)
		}
		if tc.req.path != reqcommon.PathChatCompletions {
			t.Errorf("%s: path = %q, want %q", tc.name, tc.req.path, reqcommon.PathChatCompletions)
		}
		for _, field := range []string{"routed_dp_rank", "data_parallel_rank", "disagg_prefill_dp_rank", reqcommon.FieldKVTransferParams} {
			if _, present := tc.req.body[field]; present && tc.name != "reserve" {
				t.Errorf("%s: body field %q must not be sent", tc.name, field)
			}
		}
	}

	// The ask carries the prefill body before the bootstrap fields exist.
	if _, present := reserve.body["bootstrap_room"]; present {
		t.Errorf("reserve: body carries bootstrap fields: %v", reserve.body)
	}
	if reserve.body["max_tokens"] != float64(1) {
		t.Errorf("reserve: max_tokens = %v, want the prefill body's 1", reserve.body["max_tokens"])
	}

	// Both bodies name the reserved prefill pod and share one integer room.
	room := prefill.body["bootstrap_room"]
	if _, ok := room.(float64); !ok {
		t.Fatalf("prefill: bootstrap_room = %v (%T), want a number", room, room)
	}
	for name, body := range map[string]map[string]any{"prefill": prefill.body, "decode": decode.body} {
		if body["bootstrap_host"] != "10.0.3.7" {
			t.Errorf("%s: bootstrap_host = %v, want 10.0.3.7", name, body["bootstrap_host"])
		}
		if body["bootstrap_port"] != float64(8998) {
			t.Errorf("%s: bootstrap_port = %v, want 8998", name, body["bootstrap_port"])
		}
		if body["bootstrap_room"] != room {
			t.Errorf("%s: bootstrap_room = %v, want %v", name, body["bootstrap_room"], room)
		}
	}
	if prefill.body["max_tokens"] != float64(1) || prefill.body["stream"] != false {
		t.Errorf("prefill: max_tokens = %v stream = %v, want 1 and false", prefill.body["max_tokens"], prefill.body["stream"])
	}
	if decode.body["max_tokens"] != float64(64) {
		t.Errorf("decode: max_tokens = %v, want the client's 64", decode.body["max_tokens"])
	}
}

func TestPrefillDecodeStep_Formats(t *testing.T) {
	tests := []struct {
		name      string
		useOpenAI bool
		path      string
		wantPath  string
		body      map[string]any
		tokenIDs  []int
	}{
		{
			name:      "completions",
			useOpenAI: true,
			path:      reqcommon.PathCompletions,
			wantPath:  reqcommon.PathCompletions,
			body:      map[string]any{"model": "m", "prompt": "hi", "routed_dp_rank": float64(1)},
			tokenIDs:  []int{1, 2, 3},
		},
		{
			name:      "tokens-in generate",
			useOpenAI: false,
			path:      reqcommon.PathChatCompletions,
			wantPath:  reqcommon.APITypeVLLMGenerate.Path(),
			body:      chatBody(),
			tokenIDs:  []int{1, 2, 3},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := newPDGateway(t)
			srv := httptest.NewServer(g)
			defer srv.Close()

			reqCtx, _ := newPDRequest(tt.path, tt.body)
			reqCtx.TokenIDs = tt.tokenIDs
			if err := newPDStep(t, srv.URL, map[string]any{"use_openai_format": tt.useOpenAI}).Execute(context.Background(), reqCtx); err != nil {
				t.Fatalf("Execute: %v", err)
			}
			reserve, prefill, decode := g.only(roleReserve), g.only(rolePrefill), g.only(roleDecode)
			if reserve.path != tt.wantPath || prefill.path != tt.wantPath {
				t.Errorf("reserve path %q, prefill path %q, want %q", reserve.path, prefill.path, tt.wantPath)
			}
			if decode.path != tt.path {
				t.Errorf("decode path = %q, want the client's %q", decode.path, tt.path)
			}
			for name, body := range map[string]map[string]any{"prefill": prefill.body, "decode": decode.body} {
				if body["bootstrap_host"] != "10.0.3.7" || body["bootstrap_room"] == nil {
					t.Errorf("%s: missing bootstrap fields: %v", name, body)
				}
				if _, present := body["routed_dp_rank"]; present {
					t.Errorf("%s: client rank field was sent", name)
				}
			}
			if prefill.body["bootstrap_room"] != decode.body["bootstrap_room"] {
				t.Errorf("rooms differ: prefill %v, decode %v", prefill.body["bootstrap_room"], decode.body["bootstrap_room"])
			}
		})
	}
}

func TestPrefillDecodeStep_ReserveFailures(t *testing.T) {
	tests := []struct {
		name       string
		reserve    http.HandlerFunc
		wantStatus int
	}{
		{
			name: "EPP refuses the ask",
			reserve: func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(http.StatusInternalServerError)
			},
			wantStatus: http.StatusInternalServerError,
		},
		{
			name: "no endpoint available",
			reserve: func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(http.StatusServiceUnavailable)
			},
			wantStatus: http.StatusServiceUnavailable,
		},
		{
			name:    "answer without the endpoint header",
			reserve: func(http.ResponseWriter, *http.Request) {},
		},
		{
			name: "answer with an endpoint that has no port",
			reserve: func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set(routing.ReservedEndpointHeader, "10.0.3.7")
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := newPDGateway(t)
			g.reserve = tt.reserve
			srv := httptest.NewServer(g)
			defer srv.Close()

			reqCtx, rec := newPDRequest(reqcommon.PathChatCompletions, chatBody())
			err := newPDStep(t, srv.URL, nil).Execute(context.Background(), reqCtx)
			if err == nil {
				t.Fatal("expected an error")
			}
			var upstream *pipeline.UpstreamError
			if tt.wantStatus != 0 && (!errors.As(err, &upstream) || upstream.StatusCode != tt.wantStatus) {
				t.Errorf("err = %v, want an UpstreamError with status %d", err, tt.wantStatus)
			}
			if g.count(rolePrefill) != 0 || g.count(roleDecode) != 0 {
				t.Errorf("prefill %d and decode %d requests were sent after a failed ask", g.count(rolePrefill), g.count(roleDecode))
			}
			if rec.wroteHeader {
				t.Error("the step wrote to the client; the server must answer with the error")
			}
		})
	}
}

func TestPrefillDecodeStep_PrefillFailsBeforeDecodeAnswers(t *testing.T) {
	g := newPDGateway(t)
	decodeCancelled := make(chan struct{})
	g.prefill = func(w http.ResponseWriter, r *http.Request) {
		if !g.wait(r, g.decodeArrived) {
			return
		}
		// The pinned endpoint is gone: EPP answers 503.
		w.WriteHeader(http.StatusServiceUnavailable)
	}
	g.decode = func(_ http.ResponseWriter, r *http.Request) {
		// Decode waits for KV that never comes.
		<-r.Context().Done()
		close(decodeCancelled)
	}
	srv := httptest.NewServer(g)
	defer srv.Close()

	reqCtx, rec := newPDRequest(reqcommon.PathChatCompletions, chatBody())
	err := newPDStep(t, srv.URL, nil).Execute(context.Background(), reqCtx)

	var upstream *pipeline.UpstreamError
	if !errors.As(err, &upstream) || upstream.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("err = %v, want the prefill UpstreamError 503", err)
	}
	if rec.wroteHeader {
		t.Errorf("the step wrote %d to the client; the server must answer with the prefill error", rec.Code)
	}
	waitClosed(t, decodeCancelled, "the decode request was not cancelled")
}

func TestPrefillDecodeStep_PrefillFailsAfterDecodeStartedStreaming(t *testing.T) {
	g := newPDGateway(t)
	decodeStarted := make(chan struct{})
	g.decode = func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush()
		close(decodeStarted)
		<-r.Context().Done()
	}
	g.prefill = func(w http.ResponseWriter, _ *http.Request) {
		<-decodeStarted
		w.WriteHeader(http.StatusInternalServerError)
	}
	srv := httptest.NewServer(g)
	defer srv.Close()

	body := chatBody()
	body["stream"] = true
	reqCtx, rec := newPDRequest(reqcommon.PathChatCompletions, body)
	err := newPDStep(t, srv.URL, nil).Execute(context.Background(), reqCtx)

	var streamed *pipeline.UpstreamStreamedError
	if !errors.As(err, &streamed) || streamed.StatusCode != http.StatusInternalServerError {
		t.Fatalf("err = %v, want an UpstreamStreamedError with the prefill status 500", err)
	}
	if rec.Code != http.StatusOK {
		t.Errorf("client status = %d, want the decode 200 already sent", rec.Code)
	}
}

func TestPrefillDecodeStep_PrefillFailsAfterDecodeCompleted(t *testing.T) {
	g := newPDGateway(t)
	decodeDone := make(chan struct{})
	g.decode = func(w http.ResponseWriter, _ *http.Request) {
		defer close(decodeDone)
		_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"done"}}]}`)
	}
	g.prefill = func(w http.ResponseWriter, _ *http.Request) {
		<-decodeDone
		// Let the proxy finish copying the decode answer to the client.
		time.Sleep(200 * time.Millisecond)
		w.WriteHeader(http.StatusInternalServerError)
	}
	srv := httptest.NewServer(g)
	defer srv.Close()

	reqCtx, rec := newPDRequest(reqcommon.PathChatCompletions, chatBody())
	if err := newPDStep(t, srv.URL, nil).Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("Execute: %v, want nil: the client already has the full answer", err)
	}
	if !strings.Contains(rec.Body.String(), "done") {
		t.Errorf("client body = %q, want the decode answer", rec.Body.String())
	}
}

func TestPrefillDecodeStep_DecodeFailsCancelsPrefill(t *testing.T) {
	g := newPDGateway(t)
	prefillCancelled := make(chan struct{})
	g.prefill = func(_ http.ResponseWriter, r *http.Request) {
		// Prefill waits for a decoder that never joins.
		<-r.Context().Done()
		close(prefillCancelled)
	}
	g.decode = func(w http.ResponseWriter, r *http.Request) {
		// Fail only once prefill is in flight, so the cancel has a request to stop.
		if !g.wait(r, g.prefillArrived) {
			return
		}
		w.WriteHeader(http.StatusTooManyRequests)
	}
	srv := httptest.NewServer(g)
	defer srv.Close()

	reqCtx, rec := newPDRequest(reqcommon.PathChatCompletions, chatBody())
	err := newPDStep(t, srv.URL, nil).Execute(context.Background(), reqCtx)

	var streamed *pipeline.UpstreamStreamedError
	if !errors.As(err, &streamed) || streamed.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("err = %v, want the decode UpstreamStreamedError 429", err)
	}
	if rec.Code != http.StatusTooManyRequests {
		t.Errorf("client status = %d, want the decode 429", rec.Code)
	}
	waitClosed(t, prefillCancelled, "the prefill request was not cancelled")
}

func TestPrefillDecodeStep_ClientCancelStopsBoth(t *testing.T) {
	g := newPDGateway(t)
	var cancelled sync.WaitGroup
	cancelled.Add(2)
	both := make(chan struct{}, 2)
	hold := func(_ http.ResponseWriter, r *http.Request) {
		both <- struct{}{}
		<-r.Context().Done()
		cancelled.Done()
	}
	g.prefill, g.decode = hold, hold
	srv := httptest.NewServer(g)
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		<-both
		<-both
		cancel()
	}()
	reqCtx, _ := newPDRequest(reqcommon.PathChatCompletions, chatBody())
	if err := newPDStep(t, srv.URL, nil).Execute(ctx, reqCtx); err != nil {
		t.Fatalf("Execute: %v, want nil as for a decode step whose client went away", err)
	}

	done := make(chan struct{})
	go func() { cancelled.Wait(); close(done) }()
	waitClosed(t, done, "prefill and decode were not both cancelled")
}

func TestNewPrefillDecodeStep(t *testing.T) {
	gw := gateway.New(config.GatewayConfig{Address: "http://unused"})
	tests := []struct {
		name    string
		gw      *gateway.Client
		params  map[string]any
		wantErr string
	}{
		{name: "kv-sglang", gw: gw, params: map[string]any{ParamKVConnector: kv.SGLang}},
		{name: "nil gateway", params: map[string]any{ParamKVConnector: kv.SGLang}, wantErr: "gateway client is required"},
		{name: "default connector is serial", gw: gw, params: map[string]any{}, wantErr: kv.SharedStorage},
		{name: "kv-nixl is serial", gw: gw, params: map[string]any{ParamKVConnector: kv.NIXL}, wantErr: kv.NIXL},
		{name: "unknown kv connector", gw: gw, params: map[string]any{ParamKVConnector: "nope"}, wantErr: "unknown kv_connector"},
		{name: "unknown ec connector", gw: gw, params: map[string]any{ParamKVConnector: kv.SGLang, ParamECConnector: "nope"}, wantErr: "nope"},
		{name: "bad use_openai_format", gw: gw, params: map[string]any{ParamKVConnector: kv.SGLang, "use_openai_format": "yes"}, wantErr: "use_openai_format"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			step, err := NewPrefillDecodeStep(tt.gw, tt.params)
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if step.Name() != PrefillDecodeStepName {
					t.Errorf("Name() = %q, want %q", step.Name(), PrefillDecodeStepName)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("err = %v, want it to mention %q", err, tt.wantErr)
			}
		})
	}
}

func TestSerialStepsRejectConcurrentConnector(t *testing.T) {
	gw := gateway.New(config.GatewayConfig{Address: "http://unused"})
	params := map[string]any{ParamKVConnector: kv.SGLang}
	for name, factory := range map[string]pipeline.StepFactory{
		PrefillStepName: NewPrefillStep,
		DecodeStepName:  NewDecodeStep,
	} {
		if _, err := factory(gw, params); err == nil || !strings.Contains(err.Error(), PrefillDecodeStepName) {
			t.Errorf("%s: err = %v, want a startup error pointing at %q", name, err, PrefillDecodeStepName)
		}
	}
}
