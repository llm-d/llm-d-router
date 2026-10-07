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
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/go-logr/logr/funcr"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
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
		w.WriteHeader(http.StatusNoContent)
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
	case r.Header.Get(routing.EndpointPinHeader) != "":
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

// writeNotifier closes done once n bytes have been written to the client.
type writeNotifier struct {
	http.ResponseWriter
	n       int
	written int
	done    chan struct{}
}

func (w *writeNotifier) Write(b []byte) (int, error) {
	n, err := w.ResponseWriter.Write(b)
	w.written += n
	if w.written >= w.n && w.done != nil {
		close(w.done)
		w.done = nil
	}
	return n, err
}

// executeAborting runs step.Execute and reports whether it aborted the client
// connection with http.ErrAbortHandler instead of returning.
func executeAborting(ctx context.Context, step pipeline.Step, reqCtx *pipeline.RequestContext) (aborted bool, err error) {
	defer func() {
		if r := recover(); r != nil {
			if e, ok := r.(error); !ok || !errors.Is(e, http.ErrAbortHandler) {
				panic(r)
			}
			aborted = true
		}
	}()
	return false, step.Execute(ctx, reqCtx)
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
			"Prefer":                {"if-available"},
			"X-Llm-D-Pin-Host-Port": {"10.9.9.9:8000"},
			"X-Data-Parallel-Rank":  {"3"},
			"Authorization":         {"Bearer t"},
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
		if got := tc.req.header.Get(routing.EndpointPinHeader); got != tc.pin {
			t.Errorf("%s: %s = %q, want %q (the client's copy is dropped)", tc.name, routing.EndpointPinHeader, got, tc.pin)
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
			name: "answer without the endpoint header",
			reserve: func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(http.StatusNoContent)
			},
		},
		{
			name: "answer with an endpoint that has no port",
			reserve: func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set(routing.ReservedEndpointHeader, "10.0.3.7")
				w.WriteHeader(http.StatusNoContent)
			},
		},
		{
			// An EPP that answers the ask with an error status, with the
			// endpoint header.
			name: "412 with the endpoint header",
			reserve: func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set(routing.ReservedEndpointHeader, pdTestPrefill)
				w.WriteHeader(http.StatusPreconditionFailed)
			},
			wantStatus: http.StatusPreconditionFailed,
		},
		{
			// An EPP that does not know the preference routes the ask to a
			// model server, which answers 200.
			name:    "ask forwarded to a model server",
			reserve: func(http.ResponseWriter, *http.Request) {},
		},
		{
			name: "200 with the endpoint header",
			reserve: func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set(routing.ReservedEndpointHeader, pdTestPrefill)
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
			isUpstream := errors.As(err, &upstream)
			if tt.wantStatus != 0 && (!isUpstream || upstream.StatusCode != tt.wantStatus) {
				t.Errorf("err = %v, want an UpstreamError with status %d", err, tt.wantStatus)
			}
			if tt.wantStatus == 0 && isUpstream {
				t.Errorf("err = %v, want an error that carries no upstream status", err)
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

func TestPrefillDecodeStep_LogsTheAsk(t *testing.T) {
	const (
		reservingMsg = `"msg"="reserving prefill endpoint" "path"="` + reqcommon.PathChatCompletions + `"`
		reservedMsg  = `"msg"="reserved prefill endpoint" "status"=204 "header"="` + routing.ReservedEndpointHeader + `" "endpoint"="` + pdTestPrefill + `"`
	)
	refuse := func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}
	tests := []struct {
		name      string
		verbosity int
		reserve   http.HandlerFunc
		wantErr   bool
		// want is the ask's log lines, in order.
		want []string
	}{
		{
			name:      "debug logs the ask before its answer",
			verbosity: logutil.DEBUG,
			want:      []string{reservingMsg, reservedMsg},
		},
		{
			name:      "debug logs the ask that EPP refuses, and no answer",
			verbosity: logutil.DEBUG,
			reserve:   refuse,
			wantErr:   true,
			want:      []string{reservingMsg},
		},
		{
			name:      "trace logs the ask too",
			verbosity: logutil.TRACE,
			want:      []string{reservingMsg, reservedMsg},
		},
		{
			name:      "verbose, one level below debug, does not log the ask",
			verbosity: logutil.VERBOSE,
		},
		{
			name:      "default does not log the ask",
			verbosity: logutil.DEFAULT,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := newPDGateway(t)
			if tt.reserve != nil {
				g.reserve = tt.reserve
			}
			srv := httptest.NewServer(g)
			defer srv.Close()

			var mu sync.Mutex
			var got []string
			logger := funcr.New(func(_, args string) {
				if !strings.Contains(args, "prefill endpoint") {
					return
				}
				mu.Lock()
				got = append(got, args)
				mu.Unlock()
			}, funcr.Options{Verbosity: tt.verbosity})

			reqCtx, _ := newPDRequest(reqcommon.PathChatCompletions, chatBody())
			err := newPDStep(t, srv.URL, nil).Execute(log.IntoContext(context.Background(), logger), reqCtx)
			if (err != nil) != tt.wantErr {
				t.Fatalf("Execute error = %v, wantErr %v", err, tt.wantErr)
			}

			mu.Lock()
			defer mu.Unlock()
			if len(got) != len(tt.want) {
				t.Fatalf("ask log lines = %q, want %q", got, tt.want)
			}
			for i, want := range tt.want {
				if !strings.Contains(got[i], want) {
					t.Errorf("ask log line %d = %q, want it to contain %q", i, got[i], want)
				}
			}
		})
	}
}

func TestPrefillDecodeStep_LogsThePrefillProfileRequests(t *testing.T) {
	const (
		reserveMsg = `"msg"="reserve-endpoint request" "method"="POST" "path"="` + reqcommon.PathChatCompletions + `"`
		prefillMsg = `"msg"="prefill request" "method"="POST" "path"="` + reqcommon.PathChatCompletions + `"`
		reserveID  = `"x-request-id"="` + pdTestRequestID + reserveRequestIDSuffix + `"`
		prefillID  = `"x-request-id"="` + pdTestRequestID + prefillRequestIDSuffix + `"`
		profile    = `"epp-profile"="prefill"`
		prefer     = `"prefer"="reserve-endpoint"`
		pin        = `"x-llm-d-pin-host-port"="` + pdTestPrefill + `"`
	)
	tests := []struct {
		name      string
		verbosity int
		reserve   http.HandlerFunc
		wantErr   bool
		// want is, for each log line in order, the texts it contains.
		want [][]string
	}{
		{
			name:      "debug logs the ask and the pinned prefill request",
			verbosity: logutil.DEBUG,
			want:      [][]string{{reserveMsg, reserveID, profile, prefer}, {prefillMsg, prefillID, profile, pin}},
		},
		{
			name:      "debug logs only the ask when EPP refuses it",
			verbosity: logutil.DEBUG,
			reserve: func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(http.StatusServiceUnavailable)
			},
			wantErr: true,
			want:    [][]string{{reserveMsg, reserveID, profile, prefer}},
		},
		{
			name:      "verbose, one level below debug, does not log the requests",
			verbosity: logutil.VERBOSE,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := newPDGateway(t)
			if tt.reserve != nil {
				g.reserve = tt.reserve
			}
			srv := httptest.NewServer(g)
			defer srv.Close()

			var mu sync.Mutex
			var got []string
			logger := funcr.New(func(_, args string) {
				if !strings.Contains(args, ` request" "method"="POST"`) {
					return
				}
				mu.Lock()
				got = append(got, args)
				mu.Unlock()
			}, funcr.Options{Verbosity: tt.verbosity})

			reqCtx, _ := newPDRequest(reqcommon.PathChatCompletions, chatBody())
			err := newPDStep(t, srv.URL, nil).Execute(log.IntoContext(context.Background(), logger), reqCtx)
			if (err != nil) != tt.wantErr {
				t.Fatalf("Execute error = %v, wantErr %v", err, tt.wantErr)
			}

			mu.Lock()
			defer mu.Unlock()
			if len(got) != len(tt.want) {
				t.Fatalf("request log lines = %q, want %d lines", got, len(tt.want))
			}
			for i, line := range got {
				for _, want := range tt.want[i] {
					if !strings.Contains(line, want) {
						t.Errorf("request log line %d = %q, want it to contain %q", i, line, want)
					}
				}
				// The client values of the dropped and the sensitive headers.
				for _, absent := range []string{"if-available", "10.9.9.9", "x-data-parallel-rank", "Bearer t"} {
					if strings.Contains(line, absent) {
						t.Errorf("request log line %d = %q, want it without %q", i, line, absent)
					}
				}
			}
		})
	}
}

func TestPrefillDecodeStep_LogsTheDecodeRequestWithoutTheDroppedHeaders(t *testing.T) {
	g := newPDGateway(t)
	srv := httptest.NewServer(g)
	defer srv.Close()

	logger, records := captureLogger(logutil.DEBUG)
	reqCtx, _ := newPDRequest(reqcommon.PathChatCompletions, chatBody())
	if err := newPDStep(t, srv.URL, nil).Execute(log.IntoContext(context.Background(), logger), reqCtx); err != nil {
		t.Fatalf("Execute: %v", err)
	}

	var got []string
	for _, record := range records() {
		if strings.Contains(record, `"msg"="request body"`) {
			got = append(got, record)
		}
	}
	if len(got) != 1 {
		t.Fatalf("decode request log lines = %q, want 1", got)
	}
	line := strings.ToLower(got[0])
	for _, want := range []string{`"epp-profile"="decode"`, `"x-request-id"="` + pdTestRequestID + `"`} {
		if !strings.Contains(line, want) {
			t.Errorf("decode request log line = %q, want it to contain %q", got[0], want)
		}
	}
	// The dropped client headers and their values, and the sensitive header's value.
	for _, absent := range []string{"prefer", "if-available", "x-data-parallel-rank", "10.9.9.9", "bearer t"} {
		if strings.Contains(line, absent) {
			t.Errorf("decode request log line = %q, want it without %q", got[0], absent)
		}
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
	tests := []struct {
		name string
		ctx  context.Context
	}{
		{
			// httputil.ReverseProxy aborts the connection with http.ErrAbortHandler
			// only when the request context names the server.
			name: "under an HTTP server, the proxy aborts and the step logs the cause",
			ctx:  context.WithValue(context.Background(), http.ServerContextKey, &http.Server{}),
		},
		{
			name: "outside a server, the step aborts",
			ctx:  context.Background(),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := newPDGateway(t)
			decodeStarted := make(chan struct{})
			decodeCancelled := make(chan struct{})
			g.decode = func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				w.WriteHeader(http.StatusOK)
				w.(http.Flusher).Flush()
				close(decodeStarted)
				<-r.Context().Done()
				close(decodeCancelled)
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
			logger, records := captureLogger(logutil.DEFAULT)
			aborted, err := executeAborting(log.IntoContext(tt.ctx, logger), newPDStep(t, srv.URL, nil), reqCtx)
			if !aborted {
				t.Fatalf("Execute returned %v, want the client connection aborted", err)
			}
			if rec.Code != http.StatusOK {
				t.Errorf("client status = %d, want the decode 200 already sent", rec.Code)
			}
			waitClosed(t, decodeCancelled, "the decode request was not cancelled")
			if got := countRecords(records(), `"msg"="prefill failed after decode started streaming"`, "HTTP 500"); got != 1 {
				t.Errorf("%d records log the prefill failure with its status, want 1: %q", got, records())
			}
		})
	}
}

func TestPrefillDecodeStep_PrefillFailsAfterDecodeCompleted(t *testing.T) {
	const answer = `{"choices":[{"message":{"content":"done"}}]}`
	g := newPDGateway(t)
	answered := make(chan struct{})
	g.decode = func(w http.ResponseWriter, _ *http.Request) {
		// With Content-Length, the proxy reads the end of the body together with
		// its last bytes, so the client has the whole answer once it is written.
		w.Header().Set("Content-Length", strconv.Itoa(len(answer)))
		_, _ = io.WriteString(w, answer)
	}
	g.prefill = func(w http.ResponseWriter, _ *http.Request) {
		<-answered
		w.WriteHeader(http.StatusInternalServerError)
	}
	srv := httptest.NewServer(g)
	defer srv.Close()

	reqCtx, rec := newPDRequest(reqcommon.PathChatCompletions, chatBody())
	reqCtx.ResponseWriter = &writeNotifier{ResponseWriter: rec, n: len(answer), done: answered}
	if err := newPDStep(t, srv.URL, nil).Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("Execute: %v, want nil: the client already has the full answer", err)
	}
	if rec.Body.String() != answer {
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
