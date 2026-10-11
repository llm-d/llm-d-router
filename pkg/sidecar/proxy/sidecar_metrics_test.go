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

package proxy

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/utils/set"
	"sigs.k8s.io/controller-runtime/pkg/log"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	"github.com/llm-d/llm-d-router/pkg/common/observability/toolcalling"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
	"github.com/llm-d/llm-d-router/pkg/sidecar/metrics"
)

// stageMetrics is a point-in-time copy of the per-stage metrics, compared
// before and after a request because the collectors are process-global.
type stageMetrics struct {
	encodeCount, prefillCount, decodeCount    uint64
	encodeErrors, prefillErrors, decodeErrors float64
}

func snapshotStageMetrics(t *testing.T) stageMetrics {
	t.Helper()
	return stageMetrics{
		encodeCount:   histogramCount(t, metricEncodeDuration),
		prefillCount:  histogramCount(t, metricPrefillDuration),
		decodeCount:   histogramCount(t, metricDecodeDuration),
		encodeErrors:  stageErrors(t, metrics.StageEncode),
		prefillErrors: stageErrors(t, metrics.StagePrefill),
		decodeErrors:  stageErrors(t, metrics.StageDecode),
	}
}

// delta returns after minus before for each field.
func (after stageMetrics) delta(before stageMetrics) stageMetrics {
	return stageMetrics{
		encodeCount:   after.encodeCount - before.encodeCount,
		prefillCount:  after.prefillCount - before.prefillCount,
		decodeCount:   after.decodeCount - before.decodeCount,
		encodeErrors:  after.encodeErrors - before.encodeErrors,
		prefillErrors: after.prefillErrors - before.prefillErrors,
		decodeErrors:  after.decodeErrors - before.decodeErrors,
	}
}

func chatRequest(t *testing.T, body map[string]any) *http.Request {
	t.Helper()
	raw, err := json.Marshal(body)
	require.NoError(t, err)
	return httptest.NewRequest(http.MethodPost, reqcommon.PathChatCompletions, bytes.NewReader(raw))
}

func textChatBody() map[string]any {
	return map[string]any{
		"model":    "test-model",
		"messages": []any{map[string]any{"role": "user", "content": "hello"}},
	}
}

// statusHandler replies with the given status and a JSON body.
func statusHandler(status int, body string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	})
}

func TestDisaggTypeMetricSelection(t *testing.T) {
	encoderHeader := http.CanonicalHeaderKey(routing.EncoderEndpointsHeader)
	prefillHeader := http.CanonicalHeaderKey(routing.PrefillEndpointHeader)
	allTypes := []string{metricsutil.DisaggPathPrefillDecode, metricsutil.DisaggPathEncodePrefillDecode, metricsutil.DisaggPathEncodeDecode}

	tests := []struct {
		name   string
		header http.Header
		// allowed enables SSRF protection with only these hosts permitted; nil
		// leaves it disabled.
		allowed []string
		// want is the disagg_type expected to increment; empty means none.
		want string
	}{
		{
			name:   "no headers records no disagg type",
			header: http.Header{},
		},
		{
			name: "encoders filtered out by SSRF protection fall back to prefill-decode",
			header: http.Header{
				encoderHeader: []string{"enc1:8000"},
				prefillHeader: []string{"prefill1:8000"},
			},
			allowed: []string{"prefill1:8000"},
			want:    metricsutil.DisaggPathPrefillDecode,
		},
		{
			name:   "prefill only records prefill-decode",
			header: http.Header{prefillHeader: []string{"prefill1:8000"}},
			want:   metricsutil.DisaggPathPrefillDecode,
		},
		{
			name: "encoder and prefill records encode-prefill-decode",
			header: http.Header{
				encoderHeader: []string{"enc1:8000"},
				prefillHeader: []string{"prefill1:8000"},
			},
			want: metricsutil.DisaggPathEncodePrefillDecode,
		},
		{
			name:   "encoder only records encode-decode",
			header: http.Header{encoderHeader: []string{"enc1:8000"}},
			want:   metricsutil.DisaggPathEncodeDecode,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := NewProxy(Config{Port: "8000"})
			s.allowlistValidator = &AllowlistValidator{}
			if tt.allowed != nil {
				s.allowlistValidator = &AllowlistValidator{
					logger:         log.Log,
					enabled:        true,
					allowedTargets: set.New(tt.allowed...),
				}
			}
			s.handleECConnector = func(http.ResponseWriter, *http.Request, string, []string, reqcommon.APIType) {}
			s.handlePDConnector = func(http.ResponseWriter, *http.Request, string, string, reqcommon.APIType) {}
			s.decoderProxy = http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
				if tt.want != "" {
					t.Error("decoder passthrough must not be reached for a disaggregated request")
				}
			})

			before := map[string]float64{}
			for _, dt := range allTypes {
				before[dt] = disaggCount(t, dt)
			}

			s.disaggregatedPrefillHandler(reqcommon.APITypeChatCompletions)(
				httptest.NewRecorder(), &http.Request{Header: tt.header, URL: &url.URL{Path: reqcommon.PathChatCompletions}})

			for _, dt := range allTypes {
				want := 0.0
				if dt == tt.want {
					want = 1
				}
				assert.Equalf(t, want, disaggCount(t, dt)-before[dt], "disagg_type=%q delta", dt)
			}
		})
	}
}

func TestHandleECEncodeMetrics(t *testing.T) {
	connectors := []struct {
		name   string
		handle func(s *Server) func(http.ResponseWriter, *http.Request, string, []string, reqcommon.APIType)
	}{
		{name: "ec-nixl", handle: func(s *Server) func(http.ResponseWriter, *http.Request, string, []string, reqcommon.APIType) {
			return s.handleECNIXL
		}},
		{name: "ec-shared-storage", handle: func(s *Server) func(http.ResponseWriter, *http.Request, string, []string, reqcommon.APIType) {
			return s.handleECSharedStorage
		}},
	}
	imageBody := func() map[string]any {
		return userMessageRequest(imageURLItem("https://example.com/img.jpg"))
	}
	outcomes := []struct {
		name          string
		body          func() map[string]any
		encoderStatus int
		decoderStatus int
		wantCode      int
		wantDecoder   bool
		wantDelta     stageMetrics
	}{
		{
			name:          "encoder 5xx records encode error without duration",
			body:          imageBody,
			encoderStatus: http.StatusInternalServerError,
			decoderStatus: http.StatusOK,
			wantCode:      http.StatusBadGateway,
			wantDelta:     stageMetrics{encodeErrors: 1},
		},
		{
			name:          "encoder success records encode and decode duration",
			body:          imageBody,
			encoderStatus: http.StatusOK,
			decoderStatus: http.StatusOK,
			wantCode:      http.StatusOK,
			wantDecoder:   true,
			wantDelta:     stageMetrics{encodeCount: 1, decodeCount: 1},
		},
		{
			name:          "text-only request samples no encode duration",
			body:          textChatBody,
			encoderStatus: http.StatusOK,
			decoderStatus: http.StatusOK,
			wantCode:      http.StatusOK,
			wantDecoder:   true,
			wantDelta:     stageMetrics{decodeCount: 1},
		},
		{
			name:          "decoder 503 records decode error",
			body:          imageBody,
			encoderStatus: http.StatusOK,
			decoderStatus: http.StatusServiceUnavailable,
			wantCode:      http.StatusServiceUnavailable,
			wantDecoder:   true,
			wantDelta:     stageMetrics{encodeCount: 1, decodeCount: 1, decodeErrors: 1},
		},
	}

	for _, c := range connectors {
		for _, o := range outcomes {
			t.Run(c.name+"/"+o.name, func(t *testing.T) {
				encoder := httptest.NewServer(statusHandler(o.encoderStatus,
					`{"choices":[{"message":{"content":""}}],"ec_transfer_params":{"hash-0":{"peer_host":"10.0.0.1"}}}`))
				defer encoder.Close()
				encoderURL, err := url.Parse(encoder.URL)
				require.NoError(t, err)

				s := NewProxy(Config{Port: "0", DecoderURL: encoderURL})
				s.logger = log.Log
				decoderCalled := false
				s.decoderProxy = http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					decoderCalled = true
					w.WriteHeader(o.decoderStatus)
				})

				before := snapshotStageMetrics(t)
				rw := httptest.NewRecorder()
				req := chatRequest(t, o.body())
				// Empty prefill endpoint routes straight to decoderProxy after encode.
				c.handle(s)(rw, req, "", []string{encoderURL.Host}, reqcommon.APITypeChatCompletions)

				assert.Equal(t, o.wantCode, rw.Code)
				assert.Equal(t, o.wantDecoder, decoderCalled)
				assert.Equal(t, o.wantDelta, snapshotStageMetrics(t).delta(before))
			})
		}
	}
}

func TestRunConcurrentPDMetrics(t *testing.T) {
	tests := []struct {
		name      string
		decoder   http.Handler
		client    func() http.ResponseWriter
		wantDelta stageMetrics
	}{
		{
			name:      "decode success records duration only",
			decoder:   statusHandler(http.StatusOK, `{"choices":[]}`),
			client:    func() http.ResponseWriter { return httptest.NewRecorder() },
			wantDelta: stageMetrics{decodeCount: 1},
		},
		{
			name:      "decode 500 records decode error and duration",
			decoder:   statusHandler(http.StatusInternalServerError, `{"error":"boom"}`),
			client:    func() http.ResponseWriter { return httptest.NewRecorder() },
			wantDelta: stageMetrics{decodeCount: 1, decodeErrors: 1},
		},
		{
			name:      "client write failure records decode error",
			decoder:   statusHandler(http.StatusOK, `{"choices":[]}`),
			client:    func() http.ResponseWriter { return errWriter{} },
			wantDelta: stageMetrics{decodeCount: 1, decodeErrors: 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			prefill := httptest.NewServer(statusHandler(http.StatusOK, `{}`))
			defer prefill.Close()
			prefillURL, err := url.Parse(prefill.URL)
			require.NoError(t, err)

			s := NewProxy(Config{Port: "0", DecoderURL: prefillURL})
			s.logger = log.Log
			s.decoderProxy = tt.decoder

			before := snapshotStageMetrics(t)
			body := []byte(`{"model":"m","messages":[]}`)
			req := httptest.NewRequest(http.MethodPost, reqcommon.PathChatCompletions, bytes.NewReader(body))
			s.runConcurrentPD(tt.client(), req, body, body, prefillURL.Host, constants.KVConnectorSGLang, nil)

			// Prefill runs in a goroutine; wait for its sample so it cannot leak
			// into a later test's delta.
			require.Eventually(t, func() bool {
				return snapshotStageMetrics(t).delta(before).prefillCount == 1
			}, 5*time.Second, 10*time.Millisecond)

			want := tt.wantDelta
			want.prefillCount = 1
			assert.Equal(t, want, snapshotStageMetrics(t).delta(before))
		})
	}
}

func TestRunConcurrentPDDecodeAbortMetrics(t *testing.T) {
	prefill := httptest.NewServer(statusHandler(http.StatusOK, `{}`))
	defer prefill.Close()
	prefillURL, err := url.Parse(prefill.URL)
	require.NoError(t, err)

	s := NewProxy(Config{Port: "0", DecoderURL: prefillURL})
	s.logger = log.Log
	// The reverse proxy panics with ErrAbortHandler when a stream breaks after
	// headers were sent.
	s.decoderProxy = http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		panic(http.ErrAbortHandler)
	})

	before := snapshotStageMetrics(t)
	body := []byte(`{"model":"m","messages":[]}`)
	req := httptest.NewRequest(http.MethodPost, reqcommon.PathChatCompletions, bytes.NewReader(body))

	var recovered any
	func() {
		defer func() { recovered = recover() }()
		s.runConcurrentPD(httptest.NewRecorder(), req, body, body, prefillURL.Host, constants.KVConnectorSGLang, nil)
	}()
	assert.Equal(t, http.ErrAbortHandler, recovered, "abort panic must propagate")

	require.Eventually(t, func() bool {
		return snapshotStageMetrics(t).delta(before).prefillCount == 1
	}, 5*time.Second, 10*time.Millisecond)
	assert.Equal(t, stageMetrics{prefillCount: 1, decodeCount: 1, decodeErrors: 1},
		snapshotStageMetrics(t).delta(before))
}

func TestHandleNIXLV2SerialMetrics(t *testing.T) {
	tests := []struct {
		name          string
		prefillStatus int
		decoder       http.Handler
		wantCode      int
		wantDelta     stageMetrics
	}{
		{
			name:          "prefill 500 records prefill error and skips decode",
			prefillStatus: http.StatusInternalServerError,
			decoder: http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
				t.Error("decode must not run after a prefill failure")
			}),
			wantCode:  http.StatusInternalServerError,
			wantDelta: stageMetrics{prefillCount: 1, prefillErrors: 1},
		},
		{
			name:          "decode 500 records decode error and duration",
			prefillStatus: http.StatusOK,
			decoder:       statusHandler(http.StatusInternalServerError, `{"error":"boom"}`),
			wantCode:      http.StatusInternalServerError,
			wantDelta:     stageMetrics{prefillCount: 1, decodeCount: 1, decodeErrors: 1},
		},
		{
			name:          "success records durations only",
			prefillStatus: http.StatusOK,
			decoder:       statusHandler(http.StatusOK, `{"choices":[]}`),
			wantCode:      http.StatusOK,
			wantDelta:     stageMetrics{prefillCount: 1, decodeCount: 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			prefill := httptest.NewServer(statusHandler(tt.prefillStatus, `{"kv_transfer_params":{}}`))
			defer prefill.Close()
			prefillURL, err := url.Parse(prefill.URL)
			require.NoError(t, err)

			s := NewProxy(Config{Port: "0", DecoderURL: prefillURL})
			s.logger = log.Log
			s.decoderProxy = tt.decoder

			before := snapshotStageMetrics(t)
			rw := httptest.NewRecorder()
			s.handleNIXLV2(rw, chatRequest(t, textChatBody()), prefillURL.Host, "", reqcommon.APITypeChatCompletions)

			assert.Equal(t, tt.wantCode, rw.Code)
			assert.Equal(t, tt.wantDelta, snapshotStageMetrics(t).delta(before))
		})
	}
}

func TestDecoderOnlyToolCallingMetrics(t *testing.T) {
	const chatBody = ` {"tools":[{"function":{"parameters":{"z":9007199254740993,"a":1}}}],"tool_choice":"auto","parallel_tool_calls":false,"response_format":null} `
	chatFields := []toolcalling.Field{toolcalling.FieldTools, toolcalling.FieldToolChoice, toolcalling.FieldParallelToolCalls, toolcalling.FieldResponseFormat}
	for _, tt := range []struct {
		name          string
		api           reqcommon.APIType
		body          string
		unknownLength bool
		status        int
		fields        []toolcalling.Field
	}{
		{name: "Chat fields", body: chatBody, fields: chatFields},
		{name: "Messages fields", api: reqcommon.APITypeMessages, body: chatBody, fields: []toolcalling.Field{toolcalling.FieldTools, toolcalling.FieldToolChoice}},
		{name: "unknown length", body: chatBody, unknownLength: true, fields: chatFields},
		{name: "upstream error", body: chatBody, status: 400, fields: chatFields},
		{name: "non-tool", body: `{"messages":[{"content":"tools \\u1234","tools":[]}]}`},
		{name: "structured output", body: `{"response_format":{}}`, fields: []toolcalling.Field{toolcalling.FieldResponseFormat}},
		{name: "malformed", body: `{"tools":[],"broken":`},
		{name: "invalid suffix", body: `{"tools":[]} invalid`},
		{name: "Responses excluded", api: reqcommon.APITypeResponses, body: chatBody},
		{name: "Completions excluded", api: reqcommon.APITypeCompletions, body: chatBody},
		{name: "large prompt", body: `{"messages":[{"content":"` + strings.Repeat("x", 1<<20) + `"}],"tools":[]}`, fields: []toolcalling.Field{toolcalling.FieldTools}},
	} {
		for _, dataParallel := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/dataParallel=%t", tt.name, dataParallel), func(t *testing.T) {
				bodies := make(chan []byte, 1)
				upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					raw, err := io.ReadAll(r.Body)
					assert.NoError(t, err)
					bodies <- raw
					code := tt.status
					if code == 0 {
						code = http.StatusOK
					}
					statusHandler(code, `{"result":"unchanged"}`).ServeHTTP(w, r)
				}))
				defer upstream.Close()
				upstreamURL, err := url.Parse(upstream.URL)
				require.NoError(t, err)
				s := NewProxy(Config{DecoderURL: upstreamURL})
				s.logger = log.Log
				s.allowlistValidator = &AllowlistValidator{}
				routes := s.createRoutes()
				r := httptest.NewRequest(http.MethodPost, tt.api.Path(), strings.NewReader(tt.body))
				if tt.unknownLength {
					r.ContentLength = -1
				}
				if dataParallel {
					s.dataParallelProxies[upstreamURL.Host] = s.decoderProxy
					s.decoderProxy = http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
						t.Error("local decode must not run after data-parallel selection")
					})
					r.Header.Set(routing.DataParallelEndpointHeader, upstreamURL.Host)
				}
				before := toolFieldCounters(t)
				rw := httptest.NewRecorder()
				routes.ServeHTTP(rw, r)
				code := tt.status
				if code == 0 {
					code = http.StatusOK
				}
				require.Equal(t, code, rw.Code)
				require.Equal(t, `{"result":"unchanged"}`, rw.Body.String())
				require.Len(t, bodies, 1)
				require.Equal(t, tt.body, string(<-bodies))
				want, delta := map[[2]string]float64{}, map[[2]string]float64{}
				for _, field := range tt.fields {
					want[[2]string{string(field), string(toolcalling.FieldStatusPreserved)}] = 1
				}
				for key, value := range toolFieldCounters(t) {
					if change := value - before[key]; change != 0 {
						delta[key] = change
					}
				}
				require.Equal(t, want, delta)
			})
		}
	}
}

func TestHandleNIXLV2SerialToolCallingMetrics(t *testing.T) {
	chatBody := `{"model":"test-model","messages":[],"max_tokens":11,"tools":[{"type":"function","function":{"name":"example","parameters":{"z":9007199254740993,"a":1}}}],"tool_choice":"auto","parallel_tool_calls":false,"response_format":{"type":"json_object"}}`
	chatFields := []toolcalling.Field{toolcalling.FieldTools, toolcalling.FieldToolChoice, toolcalling.FieldParallelToolCalls, toolcalling.FieldResponseFormat}
	messagesFields := []toolcalling.Field{toolcalling.FieldTools, toolcalling.FieldToolChoice}
	tests := []struct {
		name            string
		api             reqcommon.APIType
		body            string
		failPrefill     int
		prefillStatus   int
		prefillResponse string
		decodeStatus    int
		requestIDError  bool
		denyTarget      bool
		invalidTarget   bool
		dataParallel    bool
		wantCode        int
		wantPrefill     int
		wantDecode      int
		wantFields      []toolcalling.Field
	}{
		{name: "chat fields preserved across both stages", body: chatBody, wantCode: 200, wantPrefill: 1, wantDecode: 1, wantFields: chatFields},
		{name: "messages ignores unsupported fields", api: reqcommon.APITypeMessages, body: chatBody, wantCode: 200, wantPrefill: 1, wantDecode: 1, wantFields: messagesFields},
		{name: "data-parallel decode preserves fields", body: chatBody, dataParallel: true, wantCode: 200, wantPrefill: 1, wantDecode: 1, wantFields: chatFields},
		{name: "null and empty fields are observed", body: `{"tools":[],"tool_choice":null}`, wantCode: 200, wantPrefill: 1, wantDecode: 1, wantFields: messagesFields},
		{name: "non-tool request", body: `{"messages":[]}`, wantCode: 200, wantPrefill: 1, wantDecode: 1},
		{name: "nested tools are not observed", body: `{"messages":[{"tools":[]}]}`, wantCode: 200, wantPrefill: 1, wantDecode: 1},
		{name: "malformed body keeps existing rejection", body: `{"tools":`, wantCode: 400},
		{name: "retry success counts fields once", body: chatBody, failPrefill: 2, prefillStatus: 503, wantCode: 200, wantPrefill: 3, wantDecode: 1, wantFields: chatFields},
		{name: "exhausted retries do not imply field rejection", body: chatBody, failPrefill: 3, prefillStatus: 503, wantCode: 503, wantPrefill: 3, wantFields: chatFields},
		{name: "generic upstream 400 does not imply field rejection", body: chatBody, failPrefill: 1, prefillStatus: 400, wantCode: 400, wantPrefill: 1, wantFields: chatFields},
		{name: "invalid prefill response skips decode", body: chatBody, prefillResponse: `not-json`, wantCode: 400, wantPrefill: 1, wantFields: chatFields},
		{name: "decode failure does not imply field rejection", body: chatBody, decodeStatus: 500, wantCode: 500, wantPrefill: 1, wantDecode: 1, wantFields: chatFields},
		{name: "request ID failure sends no request", body: chatBody, requestIDError: true, wantCode: 502},
		{name: "SSRF denial sends no request", body: chatBody, denyTarget: true, wantCode: 403},
		{name: "handler initialization failure sends no request", body: chatBody, invalidTarget: true, wantCode: 502},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			prefillBodies := make(chan []byte, 4)
			decodeBodies := make(chan []byte, 1)
			var attempts atomic.Int32
			prefill := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				raw, readErr := io.ReadAll(r.Body)
				assert.NoError(t, readErr)
				prefillBodies <- raw
				if int(attempts.Add(1)) <= tt.failPrefill {
					statusHandler(tt.prefillStatus, `{"error":"upstream failure"}`).ServeHTTP(w, r)
					return
				}
				response := tt.prefillResponse
				if response == "" {
					response = `{"kv_transfer_params":{}}`
				}
				statusHandler(http.StatusOK, response).ServeHTTP(w, r)
			}))
			defer prefill.Close()
			decode := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				raw, readErr := io.ReadAll(r.Body)
				assert.NoError(t, readErr)
				decodeBodies <- raw
				code := tt.decodeStatus
				if code == 0 {
					code = http.StatusOK
				}
				statusHandler(code, `{"choices":[]}`).ServeHTTP(w, r)
			}))
			defer decode.Close()
			decodeURL, parseErr := url.Parse(decode.URL)
			require.NoError(t, parseErr)
			prefillURL, parseErr := url.Parse(prefill.URL)
			require.NoError(t, parseErr)
			s := NewProxy(Config{DecoderURL: decodeURL, KVConnector: constants.KVConnectorNIXLV2, PrefillMaxRetries: 2})
			s.logger = log.Log
			s.allowlistValidator = &AllowlistValidator{enabled: tt.denyTarget}
			if tt.requestIDError {
				s.nixlRequestIDFn = func() (string, error) { return "", errors.New("request ID failure") }
			}
			target := prefillURL.Host
			if tt.invalidTarget {
				target = "%invalid"
			}
			before := toolFieldCounters(t)
			r := httptest.NewRequest(http.MethodPost, tt.api.Path(), strings.NewReader(tt.body))
			r.Header.Set(routing.PrefillEndpointHeader, target)
			routes := s.createRoutes()
			if tt.dataParallel {
				s.forwardDataParallel = true
				s.dataParallelProxies[decodeURL.Host] = s.decoderProxy
				s.decoderProxy = http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
					t.Error("local decode must not run after data-parallel dispatch")
				})
				r.Header.Set(routing.DataParallelEndpointHeader, decodeURL.Host)
			}
			rw := httptest.NewRecorder()
			routes.ServeHTTP(rw, r)
			require.Equal(t, tt.wantCode, rw.Code, rw.Body.String())
			require.Len(t, prefillBodies, tt.wantPrefill)
			require.Len(t, decodeBodies, tt.wantDecode)

			delta := make(map[[2]string]float64)
			for key, value := range toolFieldCounters(t) {
				if change := value - before[key]; change != 0 {
					delta[key] = change
				}
			}
			want := make(map[[2]string]float64)
			for _, field := range tt.wantFields {
				want[[2]string{string(field), string(toolcalling.FieldStatusPreserved)}] = 1
			}
			require.Equal(t, want, delta)

			// Check the bytes received upstream, including nested field key order
			// and numbers that cannot be represented exactly as float64.
			if tt.wantPrefill+tt.wantDecode > 0 {
				var original map[string]json.RawMessage
				require.NoError(t, json.Unmarshal([]byte(tt.body), &original))
				for _, bodies := range []chan []byte{prefillBodies, decodeBodies} {
					for len(bodies) > 0 {
						var actual map[string]json.RawMessage
						require.NoError(t, json.Unmarshal(<-bodies, &actual))
						for _, field := range chatFields {
							require.Equal(t, original[string(field)], actual[string(field)], "upstream field %s", field)
						}
					}
				}
			}
		})
	}
}

func TestP2PToolCallingMetrics(t *testing.T) {
	chatBody := `{"model":"test-model","messages":[],"max_tokens":11,"tools":[{"type":"function","function":{"name":"example","parameters":{"z":9007199254740993,"a":1}}}],"tool_choice":"auto","parallel_tool_calls":false,"response_format":{"type":"json_object"}}`
	chatFields := []toolcalling.Field{toolcalling.FieldTools, toolcalling.FieldToolChoice, toolcalling.FieldParallelToolCalls, toolcalling.FieldResponseFormat}
	messagesFields := []toolcalling.Field{toolcalling.FieldTools, toolcalling.FieldToolChoice}
	for _, path := range []string{"prefill and decode", "peer source", "self source"} {
		t.Run(path, func(t *testing.T) {
			type testCase struct {
				name          string
				api           reqcommon.APIType
				body          string
				prefillStatus int
				decodeStatus  int
				invalidTarget bool
				wantCode      int
				wantFields    []toolcalling.Field
			}
			tests := []testCase{
				{name: "chat fields", body: chatBody, wantCode: 200, wantFields: chatFields},
				{name: "messages fields", api: reqcommon.APITypeMessages, body: chatBody, wantCode: 200, wantFields: messagesFields},
				{name: "null and empty fields", body: `{"tools":[],"tool_choice":null}`, wantCode: 200, wantFields: messagesFields},
				{name: "non-tool body", body: `{"messages":[]}`, wantCode: 200},
				{name: "nested tools", body: `{"messages":[{"tools":[]}]}`, wantCode: 200},
				{name: "malformed body", body: `{"tools":`, wantCode: 400},
				{name: "unsupported API", api: reqcommon.APITypeResponses, body: chatBody, wantCode: 200},
				{name: "decode error", body: chatBody, decodeStatus: 500, wantCode: 500, wantFields: chatFields},
			}
			if path == "prefill and decode" {
				tests = append(tests,
					testCase{name: "prefill error", body: chatBody, prefillStatus: 400, wantCode: 400, wantFields: chatFields},
					testCase{name: "handler initialization error", body: chatBody, invalidTarget: true, wantCode: 502},
				)
			}
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					t.Setenv("POD_IP", "127.0.0.1")
					prefillBodies := make(chan []byte, 1)
					decodeBodies := make(chan []byte, 1)
					upstream := func(bodies chan []byte, code int) http.Handler {
						return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
							raw, err := io.ReadAll(r.Body)
							assert.NoError(t, err)
							bodies <- raw
							if code == 0 {
								code = http.StatusOK
							}
							statusHandler(code, `{"choices":[]}`).ServeHTTP(w, r)
						})
					}
					prefill := httptest.NewServer(upstream(prefillBodies, tt.prefillStatus))
					defer prefill.Close()
					decode := httptest.NewServer(upstream(decodeBodies, tt.decodeStatus))
					defer decode.Close()
					prefillURL, err := url.Parse(prefill.URL)
					require.NoError(t, err)
					decodeURL, err := url.Parse(decode.URL)
					require.NoError(t, err)
					s := NewProxy(Config{Port: "8000", DecoderURL: decodeURL, KVConnector: constants.KVConnectorOffloading})
					s.logger = log.Log
					s.createRoutes()
					r := httptest.NewRequest(http.MethodPost, tt.api.Path(), strings.NewReader(tt.body))
					rw := httptest.NewRecorder()
					before := toolFieldCounters(t)
					if path == "prefill and decode" {
						target := prefillURL.Host
						if tt.invalidTarget {
							target = "%invalid"
						}
						s.handleP2P(rw, r, target, "10.0.0.2:8000", tt.api)
					} else {
						source := "10.0.0.2:8000"
						if path == "self source" {
							source = "127.0.0.1:8000"
						}
						s.decodeWithP2PSource(rw, r, source)
					}
					require.Equal(t, tt.wantCode, rw.Code, rw.Body.String())
					wantPrefill, wantDecode := 0, 0
					if tt.wantCode != 400 || tt.prefillStatus != 0 {
						if !tt.invalidTarget {
							wantDecode = 1
							if path == "prefill and decode" {
								wantPrefill = 1
								if tt.prefillStatus != 0 {
									wantDecode = 0
								}
							}
						}
					}
					require.Len(t, prefillBodies, wantPrefill)
					require.Len(t, decodeBodies, wantDecode)
					delta := make(map[[2]string]float64)
					for key, value := range toolFieldCounters(t) {
						if change := value - before[key]; change != 0 {
							delta[key] = change
						}
					}
					want := make(map[[2]string]float64)
					for _, field := range tt.wantFields {
						want[[2]string{string(field), string(toolcalling.FieldStatusPreserved)}] = 1
					}
					require.Equal(t, want, delta)
					if wantPrefill+wantDecode > 0 {
						var original map[string]json.RawMessage
						require.NoError(t, json.Unmarshal([]byte(tt.body), &original))
						for _, bodies := range []chan []byte{prefillBodies, decodeBodies} {
							for len(bodies) > 0 {
								raw := <-bodies
								var actual map[string]json.RawMessage
								require.NoError(t, json.Unmarshal(raw, &actual))
								for _, field := range chatFields {
									require.Equal(t, original[string(field)], actual[string(field)], "upstream field %s", field)
								}
								if path == "self source" {
									require.Equal(t, tt.body, string(raw))
								}
							}
						}
					}
				})
			}
		})
	}
}

func TestP2PDispatchedToolCallingMetrics(t *testing.T) {
	original := `{"tools":[],"tool_choice":"auto"}`
	for _, api := range []reqcommon.APIType{reqcommon.APITypeChatCompletions, reqcommon.APITypeMessages} {
		t.Run(api.String(), func(t *testing.T) {
			for _, tt := range []struct {
				name          string
				prefill       string
				decode        string
				prefillStatus int
				captureError  bool
				wantTools     toolcalling.FieldStatusValue
				wantChoice    toolcalling.FieldStatusValue
			}{
				{name: "prefill changed", prefill: `{"tools":[],"tool_choice":"none"}`, decode: original, wantTools: toolcalling.FieldStatusPreserved, wantChoice: toolcalling.FieldStatusChanged},
				{name: "decode dropped", prefill: original, decode: `{"tool_choice":"auto"}`, wantTools: toolcalling.FieldStatusDropped, wantChoice: toolcalling.FieldStatusPreserved},
				{name: "worst outcome across stages", prefill: `{"tools":[],"tool_choice":"none"}`, decode: `{"tools":[]}`, wantTools: toolcalling.FieldStatusPreserved, wantChoice: toolcalling.FieldStatusDropped},
				{name: "undispatched decode excluded", prefill: original, decode: `{}`, prefillStatus: 400, wantTools: toolcalling.FieldStatusPreserved, wantChoice: toolcalling.FieldStatusPreserved},
				{name: "failed outbound capture", prefill: `{`, decode: original, wantTools: toolcalling.FieldStatusPreserved, wantChoice: toolcalling.FieldStatusPreserved},
				{name: "failed inbound capture", prefill: original, decode: original, captureError: true},
			} {
				t.Run(tt.name, func(t *testing.T) {
					code := tt.prefillStatus
					if code == 0 {
						code = http.StatusOK
					}
					prefill := httptest.NewServer(statusHandler(code, `{}`))
					defer prefill.Close()
					prefillURL, err := url.Parse(prefill.URL)
					require.NoError(t, err)
					s := NewProxy(Config{DecoderURL: prefillURL})
					s.logger = log.Log
					decodeCount := 0
					s.decoderProxy = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						decodeCount++
						body, readErr := io.ReadAll(r.Body)
						require.NoError(t, readErr)
						require.Equal(t, tt.decode, string(body))
						statusHandler(http.StatusOK, `{}`).ServeHTTP(w, r)
					})
					captured := original
					if tt.captureError {
						captured = `{`
					}
					toolFields, captureErr := captureToolCallingRequest(api, []byte(captured))
					require.Equal(t, tt.captureError, captureErr != nil)
					before := toolFieldCounters(t)
					r := httptest.NewRequest(http.MethodPost, api.Path(), strings.NewReader(original))
					rw := httptest.NewRecorder()
					s.handleP2PSequentialRequests(rw, r, []byte(tt.prefill), []byte(tt.decode), prefillURL.Host, toolFields)
					require.Equal(t, code, rw.Code)
					require.Equal(t, tt.prefillStatus == 0, decodeCount == 1)
					delta := make(map[[2]string]float64)
					for key, value := range toolFieldCounters(t) {
						if change := value - before[key]; change != 0 {
							delta[key] = change
						}
					}
					want := make(map[[2]string]float64)
					if !tt.captureError {
						want[[2]string{string(toolcalling.FieldTools), string(tt.wantTools)}] = 1
						want[[2]string{string(toolcalling.FieldToolChoice), string(tt.wantChoice)}] = 1
					}
					require.Equal(t, want, delta)
				})
			}
		})
	}
}

func TestHandleP2PMetrics(t *testing.T) {
	tests := []struct {
		name          string
		prefillStatus int
		decoder       http.Handler
		client        func() http.ResponseWriter
		wantDelta     stageMetrics
	}{
		{
			name:          "prefill 500 records prefill error and skips decode",
			prefillStatus: http.StatusInternalServerError,
			decoder: http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
				t.Error("decode must not run after a prefill failure")
			}),
			client:    func() http.ResponseWriter { return httptest.NewRecorder() },
			wantDelta: stageMetrics{prefillCount: 1, prefillErrors: 1},
		},
		{
			name:          "decode 500 records decode error and duration",
			prefillStatus: http.StatusOK,
			decoder:       statusHandler(http.StatusInternalServerError, `{"error":"boom"}`),
			client:        func() http.ResponseWriter { return httptest.NewRecorder() },
			wantDelta:     stageMetrics{prefillCount: 1, decodeCount: 1, decodeErrors: 1},
		},
		{
			name:          "client write failure records decode error",
			prefillStatus: http.StatusOK,
			decoder:       statusHandler(http.StatusOK, `{"choices":[]}`),
			client:        func() http.ResponseWriter { return errWriter{} },
			wantDelta:     stageMetrics{prefillCount: 1, decodeCount: 1, decodeErrors: 1},
		},
		{
			name:          "success records durations only",
			prefillStatus: http.StatusOK,
			decoder:       statusHandler(http.StatusOK, `{"choices":[]}`),
			client:        func() http.ResponseWriter { return httptest.NewRecorder() },
			wantDelta:     stageMetrics{prefillCount: 1, decodeCount: 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			prefill := httptest.NewServer(statusHandler(tt.prefillStatus, `{}`))
			defer prefill.Close()
			prefillURL, err := url.Parse(prefill.URL)
			require.NoError(t, err)

			s := NewProxy(Config{Port: "0", DecoderURL: prefillURL})
			s.logger = log.Log
			s.decoderProxy = tt.decoder

			before := snapshotStageMetrics(t)
			s.handleP2P(tt.client(), chatRequest(t, textChatBody()), prefillURL.Host, "", reqcommon.APITypeChatCompletions)

			assert.Equal(t, tt.wantDelta, snapshotStageMetrics(t).delta(before))
		})
	}
}

func TestHandleP2PDecodeAbortMetrics(t *testing.T) {
	prefill := httptest.NewServer(statusHandler(http.StatusOK, `{}`))
	defer prefill.Close()
	prefillURL, err := url.Parse(prefill.URL)
	require.NoError(t, err)

	s := NewProxy(Config{Port: "0", DecoderURL: prefillURL})
	s.logger = log.Log
	s.decoderProxy = http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		panic(http.ErrAbortHandler)
	})

	before := snapshotStageMetrics(t)
	beforeFields := toolFieldCounters(t)
	body := textChatBody()
	body[string(toolcalling.FieldTools)] = []any{}
	var recovered any
	func() {
		defer func() { recovered = recover() }()
		s.handleP2P(httptest.NewRecorder(), chatRequest(t, body), prefillURL.Host, "", reqcommon.APITypeChatCompletions)
	}()
	assert.Equal(t, http.ErrAbortHandler, recovered, "abort panic must propagate")
	assert.Equal(t, stageMetrics{prefillCount: 1, decodeCount: 1, decodeErrors: 1},
		snapshotStageMetrics(t).delta(before))
	for key, value := range toolFieldCounters(t) {
		want := float64(0)
		if key == [2]string{string(toolcalling.FieldTools), string(toolcalling.FieldStatusPreserved)} {
			want = 1
		}
		assert.Equal(t, want, value-beforeFields[key], "field outcome %v", key)
	}
}

func TestRecordDecodeAbort(t *testing.T) {
	t.Run("not returned records decode error and duration", func(t *testing.T) {
		before := snapshotStageMetrics(t)
		returned := false
		recordDecodeAbort(&returned, time.Now())
		assert.Equal(t, stageMetrics{decodeCount: 1, decodeErrors: 1}, snapshotStageMetrics(t).delta(before))
	})
	t.Run("returned records nothing", func(t *testing.T) {
		before := snapshotStageMetrics(t)
		returned := true
		recordDecodeAbort(&returned, time.Now())
		assert.Equal(t, stageMetrics{}, snapshotStageMetrics(t).delta(before))
	})
}

// flushRecorder counts Flush calls on top of httptest.ResponseRecorder.
type flushRecorder struct {
	*httptest.ResponseRecorder
	flushes int
}

func (f *flushRecorder) Flush() { f.flushes++ }

// failingWriter records the status it is given and fails every body write.
type failingWriter struct {
	header http.Header
	status int
}

func (f *failingWriter) Header() http.Header        { return f.header }
func (f *failingWriter) WriteHeader(statusCode int) { f.status = statusCode }
func (f *failingWriter) Write([]byte) (int, error)  { return 0, errors.New("client gone") }

// readerFromHijacker is a response writer exposing io.ReaderFrom and
// http.Hijacker, as net/http's response writer does.
type readerFromHijacker struct {
	*httptest.ResponseRecorder
}

func (readerFromHijacker) ReadFrom(io.Reader) (int64, error) { return 0, nil }
func (readerFromHijacker) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	return nil, nil, nil
}

func TestCaptureResponseStatus(t *testing.T) {
	t.Run("write without header records implicit 200", func(t *testing.T) {
		w, status := captureResponseStatus(httptest.NewRecorder())
		_, err := w.Write([]byte("ok"))
		require.NoError(t, err)
		assert.Equal(t, http.StatusOK, status.statusCode)
		assert.False(t, status.failed())
	})
	t.Run("explicit error status is recorded and passed through", func(t *testing.T) {
		rec := httptest.NewRecorder()
		w, status := captureResponseStatus(rec)
		w.WriteHeader(http.StatusServiceUnavailable)
		assert.Equal(t, http.StatusServiceUnavailable, status.statusCode)
		assert.Equal(t, http.StatusServiceUnavailable, rec.Code)
		assert.True(t, status.failed())
	})
	t.Run("first final status wins", func(t *testing.T) {
		w, status := captureResponseStatus(httptest.NewRecorder())
		w.WriteHeader(http.StatusBadGateway)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("x"))
		assert.Equal(t, http.StatusBadGateway, status.statusCode)
		assert.True(t, status.failed())
	})
	t.Run("informational status does not latch", func(t *testing.T) {
		w, status := captureResponseStatus(httptest.NewRecorder())
		w.WriteHeader(http.StatusEarlyHints)
		w.WriteHeader(http.StatusOK)
		assert.Equal(t, http.StatusOK, status.statusCode)
		assert.False(t, status.failed())
	})
	t.Run("no status written counts as failed", func(t *testing.T) {
		_, status := captureResponseStatus(httptest.NewRecorder())
		assert.True(t, status.failed())
	})
	t.Run("flush passes through to a flusher", func(t *testing.T) {
		fr := &flushRecorder{ResponseRecorder: httptest.NewRecorder()}
		w, _ := captureResponseStatus(fr)
		http.NewResponseController(w).Flush()
		assert.Equal(t, 1, fr.flushes)
	})
	t.Run("underlying write error marks failed despite 200", func(t *testing.T) {
		w, status := captureResponseStatus(&failingWriter{header: http.Header{}})
		w.WriteHeader(http.StatusOK)
		_, err := w.Write([]byte("x"))
		require.Error(t, err)
		assert.Equal(t, http.StatusOK, status.statusCode)
		assert.True(t, status.failed())
	})
	t.Run("optional interfaces survive the cached-tokens wrapper", func(t *testing.T) {
		inner, _ := captureResponseStatus(readerFromHijacker{httptest.NewRecorder()})
		w, _ := newCachedTokensResponseWriterWithFinalize(inner, 1, false)
		_, isReaderFrom := w.(io.ReaderFrom)
		assert.True(t, isReaderFrom)
		_, _, err := http.NewResponseController(w).Hijack()
		assert.NoError(t, err)
	})
}

func TestHandleNIXLV2ParallelToolCallingMetrics(t *testing.T) {
	const chatBody = `{"model":"test-model","messages":[],"tools":[{"type":"function","function":{"name":"example","parameters":{"z":9007199254740993,"a":1}}}],"tool_choice":"auto","parallel_tool_calls":false,"response_format":{"type":"json_object"}}`
	chatFields := []toolcalling.Field{toolcalling.FieldTools, toolcalling.FieldToolChoice, toolcalling.FieldParallelToolCalls, toolcalling.FieldResponseFormat}
	messagesFields := []toolcalling.Field{toolcalling.FieldTools, toolcalling.FieldToolChoice}
	for _, tt := range []struct {
		name           string
		api            reqcommon.APIType
		body           string
		prefillCode    int
		decodeCode     int
		wantCode       int
		wantFields     []toolcalling.Field
		waitForDecode  bool
		requestIDError bool
		invalidTarget  bool
		dataParallel   bool
		missingReplica bool
		prefillChoice  string
		decodeChoice   string
	}{
		{name: "chat fields counted once across both stages", body: chatBody, wantCode: 200, wantFields: chatFields},
		{name: "Messages ignores unsupported fields", api: reqcommon.APITypeMessages, body: chatBody, wantCode: 200, wantFields: messagesFields},
		{name: "null and empty fields", body: `{"tools":[],"tool_choice":null}`, wantCode: 200, wantFields: messagesFields},
		{name: "non-tool request", body: `{"messages":[]}`, wantCode: 200},
		{name: "nested fields", body: `{"messages":[{"tools":[]}]}`, wantCode: 200},
		{name: "structured output only", body: `{"response_format":null}`, wantCode: 200, wantFields: []toolcalling.Field{toolcalling.FieldResponseFormat}},
		{name: "Responses is outside sidecar coverage", api: reqcommon.APITypeResponses, body: chatBody, wantCode: 200},
		{name: "decode starts before prefill completes", body: chatBody, waitForDecode: true, wantCode: 200, wantFields: chatFields},
		{name: "upstream rejection does not imply tool rejection", body: chatBody, prefillCode: 400, wantCode: 400, wantFields: chatFields},
		{name: "decode failure retains field outcomes", body: chatBody, decodeCode: 500, wantCode: 500, wantFields: chatFields},
		{name: "malformed input", body: `{"tools":`, wantCode: 400},
		{name: "request ID failure", body: chatBody, requestIDError: true, wantCode: 502},
		{name: "handler initialization failure", body: chatBody, invalidTarget: true, wantCode: 502},
		{name: "only decode changes choice", body: chatBody, prefillChoice: "auto", decodeChoice: "none", wantCode: 200, wantFields: chatFields},
		{name: "only prefill changes choice", body: chatBody, prefillChoice: "none", decodeChoice: "auto", wantCode: 200, wantFields: chatFields},
		{name: "data-parallel decode changes choice", body: chatBody, dataParallel: true, prefillChoice: "auto", decodeChoice: "none", wantCode: 200, wantFields: chatFields},
		{name: "Messages data-parallel decode changes choice", api: reqcommon.APITypeMessages, body: chatBody, dataParallel: true, prefillChoice: "auto", decodeChoice: "none", wantCode: 200, wantFields: messagesFields},
		{name: "non-tool data-parallel request", body: `{"messages":[]}`, dataParallel: true, wantCode: 200},
		{name: "missing replica supplies no decode evidence", body: chatBody, dataParallel: true, missingReplica: true, prefillChoice: "auto", decodeChoice: "none", wantCode: 400, wantFields: chatFields},
	} {
		t.Run(tt.name, func(t *testing.T) {
			prefillBodies, decodeBodies := make(chan []byte, 1), make(chan []byte, 1)
			decodeStarted := make(chan struct{})
			prefill := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				raw, err := io.ReadAll(r.Body)
				assert.NoError(t, err)
				prefillBodies <- raw
				if tt.waitForDecode {
					select {
					case <-decodeStarted:
					case <-r.Context().Done():
						return
					}
				}
				code := tt.prefillCode
				if code == 0 {
					code = http.StatusOK
				}
				statusHandler(code, `{}`).ServeHTTP(w, r)
			}))
			defer prefill.Close()
			prefillURL, err := url.Parse(prefill.URL)
			require.NoError(t, err)
			s := NewProxy(Config{
				DecoderURL: prefillURL, KVConnector: constants.KVConnectorNIXLV2,
				MoRIIOWriteMode: true, MoRIIOParallelDispatch: true,
				MoRIIODecodePodIP: "127.0.0.1", MoRIIOParallelDecodeWaitTimeout: 2 * time.Second,
			})
			s.logger = log.Log
			s.allowlistValidator = &AllowlistValidator{}
			if tt.requestIDError {
				s.nixlRequestIDFn = func() (string, error) { return "", errors.New("request ID failure") }
			}
			s.decoderProxy = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				raw, err := io.ReadAll(r.Body)
				assert.NoError(t, err)
				decodeBodies <- raw
				close(decodeStarted)
				code := tt.decodeCode
				if code == 0 {
					code = http.StatusOK
				}
				statusHandler(code, `{"choices":[]}`).ServeHTTP(w, r)
			})
			target := prefillURL.Host
			if tt.invalidTarget {
				target = "%invalid"
			}
			before := toolFieldCounters(t)
			rw := httptest.NewRecorder()
			r := httptest.NewRequest(http.MethodPost, tt.api.Path(), strings.NewReader(tt.body))
			if tt.dataParallel {
				s.forwardDataParallel = true
				if !tt.missingReplica {
					s.dataParallelProxies[prefillURL.Host] = s.decoderProxy
				}
				s.decoderProxy = http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
					t.Error("local decode must not run after data-parallel selection")
				})
				r.Header.Set(routing.DataParallelEndpointHeader, prefillURL.Host)
			}
			if tt.prefillChoice != "" {
				// Supply different serialization results to verify each dispatch independently.
				body, err := decodeRequestBody([]byte(tt.body))
				require.NoError(t, err)
				body[string(toolcalling.FieldToolChoice)] = &parallelToolChoice{prefill: tt.prefillChoice, decode: tt.decodeChoice}
				s.runNIXLProtocolV2WriteParallel(rw, r, []byte(tt.body), body, "request-id", "txrequest-id", target, "", tt.api)
			} else {
				s.handleNIXLV2(rw, r, target, "", tt.api)
			}
			require.Equal(t, tt.wantCode, rw.Code)
			if tt.wantCode == http.StatusOK {
				require.Equal(t, `{"choices":[]}`, rw.Body.String())
			}
			delta := make(map[[2]string]float64)
			for key, value := range toolFieldCounters(t) {
				if change := value - before[key]; change != 0 {
					delta[key] = change
				}
			}
			want := make(map[[2]string]float64)
			for _, field := range tt.wantFields {
				status := toolcalling.FieldStatusPreserved
				if field == toolcalling.FieldToolChoice && tt.prefillChoice != "" && !tt.missingReplica {
					status = toolcalling.FieldStatusChanged
				}
				want[[2]string{string(field), string(status)}] = 1
			}
			require.Equal(t, want, delta)
			if tt.requestIDError || tt.invalidTarget || tt.wantCode == http.StatusBadRequest && tt.prefillCode == 0 && !tt.missingReplica {
				require.Empty(t, prefillBodies)
				require.Empty(t, decodeBodies)
				return
			}
			var original map[string]json.RawMessage
			require.NoError(t, json.Unmarshal([]byte(tt.body), &original))
			for i, bodies := range []chan []byte{prefillBodies, decodeBodies} {
				if i == 1 && tt.missingReplica {
					require.Empty(t, bodies)
					continue
				}
				require.Len(t, bodies, 1)
				var actual map[string]json.RawMessage
				require.NoError(t, json.Unmarshal(<-bodies, &actual))
				for _, field := range chatFields {
					if field == toolcalling.FieldToolChoice && tt.prefillChoice != "" {
						choice := tt.prefillChoice
						if i == 1 {
							choice = tt.decodeChoice
						}
						encoded, err := json.Marshal(choice)
						require.NoError(t, err)
						require.JSONEq(t, string(encoded), string(actual[string(field)]))
					} else {
						require.Equal(t, original[string(field)], actual[string(field)])
					}
				}
			}
		})
	}
}

type parallelToolChoice struct {
	prefill, decode string
	marshaled       bool
}

func (choice *parallelToolChoice) MarshalJSON() ([]byte, error) {
	value := choice.prefill
	if choice.marshaled {
		value = choice.decode
	}
	choice.marshaled = true
	return json.Marshal(value)
}

func TestHandleNIXLV2ParallelWriteMetrics(t *testing.T) {
	tests := []struct {
		name          string
		prefillStatus int
		decodeStatus  int
		wantCode      int
		wantDelta     stageMetrics
	}{
		{
			name:          "prefill and decode succeed",
			prefillStatus: http.StatusOK,
			decodeStatus:  http.StatusOK,
			wantCode:      http.StatusOK,
			wantDelta:     stageMetrics{prefillCount: 1, decodeCount: 1},
		},
		{
			name:          "prefill 5xx is not attributed to decode",
			prefillStatus: http.StatusInternalServerError,
			decodeStatus:  http.StatusOK,
			wantCode:      http.StatusInternalServerError,
			wantDelta:     stageMetrics{prefillCount: 1, prefillErrors: 1},
		},
		{
			name:          "decode 500 after prefill success records decode error",
			prefillStatus: http.StatusOK,
			decodeStatus:  http.StatusInternalServerError,
			wantCode:      http.StatusInternalServerError,
			wantDelta:     stageMetrics{prefillCount: 1, decodeCount: 1, decodeErrors: 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			prefill := httptest.NewServer(statusHandler(tt.prefillStatus, `{}`))
			defer prefill.Close()
			prefillURL, err := url.Parse(prefill.URL)
			require.NoError(t, err)

			s := NewProxy(Config{
				Port:                       "0",
				DecoderURL:                 prefillURL,
				KVConnector:                constants.KVConnectorNIXLV2,
				MoRIIOWriteMode:            true,
				MoRIIOParallelDispatch:     true,
				MoRIIODecodePodIP:          "127.0.0.1",
				MoRIIODecodeNotifyPort:     61005,
				MoRIIODecodeHandshakePort:  6301,
				MoRIIOPrefillNotifyPort:    61006,
				MoRIIOPrefillHandshakePort: 6302,
				MoRIIOTPSize:               1,
				MoRIIODPSize:               1,
			})
			s.logger = log.Log
			// The parallel path synthesises do_remote_prefill for decode; the serial
			// path would copy the (empty) prefill response instead.
			var sawSynthesizedKV atomic.Bool
			decode := statusHandler(tt.decodeStatus, `{"choices":[]}`)
			s.decoderProxy = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var got map[string]any
				if json.NewDecoder(r.Body).Decode(&got) == nil {
					if kv, ok := got[reqcommon.FieldKVTransferParams].(map[string]any); ok && kv[reqcommon.FieldDoRemotePrefill] == true {
						sawSynthesizedKV.Store(true)
					}
				}
				decode.ServeHTTP(w, r)
			})

			before := snapshotStageMetrics(t)
			rw := httptest.NewRecorder()
			s.handleNIXLV2(rw, chatRequest(t, textChatBody()), prefillURL.Host, "", reqcommon.APITypeChatCompletions)

			assert.Equal(t, tt.wantCode, rw.Code)
			assert.Equal(t, tt.wantDelta, snapshotStageMetrics(t).delta(before))
			if tt.prefillStatus == http.StatusOK {
				assert.True(t, sawSynthesizedKV.Load(), "request must take the parallel dispatch path")
			}
		})
	}
}

// A decode response that cannot be delivered when the cached-tokens rewriter
// flushes its buffered body is a decode error.
func TestHandleNIXLV2SerialFinalizeFailureMetrics(t *testing.T) {
	prefill := httptest.NewServer(statusHandler(http.StatusOK, `{"kv_transfer_params":{}}`))
	defer prefill.Close()
	prefillURL, err := url.Parse(prefill.URL)
	require.NoError(t, err)

	s := NewProxy(Config{Port: "0", DecoderURL: prefillURL})
	s.logger = log.Log
	// An incomplete JSON document stays buffered until finalize.
	s.decoderProxy = statusHandler(http.StatusOK, `{"choices":[`)

	before := snapshotStageMetrics(t)
	s.handleNIXLV2(&failingWriter{header: http.Header{}}, chatRequest(t, textChatBody()), prefillURL.Host, "", reqcommon.APITypeChatCompletions)

	assert.Equal(t, stageMetrics{prefillCount: 1, decodeCount: 1, decodeErrors: 1}, snapshotStageMetrics(t).delta(before))
}

func TestHandleNIXLV2ParallelWriteFailureMetrics(t *testing.T) {
	tests := []struct {
		name string
		// stop is closed before the prefill server shuts down.
		prefill   func(stop <-chan struct{}) http.Handler
		decoder   http.Handler
		wantCode  int
		wantPanic bool
		wantDelta stageMetrics
	}{
		{
			name: "KV-wait timeout records a prefill error and the decode duration",
			prefill: func(stop <-chan struct{}) http.Handler {
				return http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
					select {
					case <-r.Context().Done():
					case <-stop:
					}
				})
			},
			decoder:   statusHandler(http.StatusOK, `{"choices":[]}`),
			wantCode:  http.StatusGatewayTimeout,
			wantDelta: stageMetrics{prefillCount: 1, prefillErrors: 1, decodeCount: 1},
		},
		{
			name: "decode abort after prefill success records a decode error",
			prefill: func(<-chan struct{}) http.Handler {
				return statusHandler(http.StatusOK, `{}`)
			},
			decoder: http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
				panic(http.ErrAbortHandler)
			}),
			wantCode:  http.StatusBadGateway,
			wantDelta: stageMetrics{prefillCount: 1, decodeCount: 1, decodeErrors: 1},
		},
		{
			name: "decode abort after its 200 reached the client records a decode error",
			prefill: func(<-chan struct{}) http.Handler {
				return statusHandler(http.StatusOK, `{}`)
			},
			decoder: http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(http.StatusOK)
				_, _ = io.WriteString(w, "data: {}\n\n")
				// Let the commit point relay the 200 before the stream breaks.
				time.Sleep(100 * time.Millisecond)
				panic(http.ErrAbortHandler)
			}),
			wantCode:  http.StatusOK,
			wantPanic: true,
			wantDelta: stageMetrics{prefillCount: 1, decodeCount: 1, decodeErrors: 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stop := make(chan struct{})
			prefill := httptest.NewServer(tt.prefill(stop))
			defer prefill.Close()
			defer close(stop)
			prefillURL, err := url.Parse(prefill.URL)
			require.NoError(t, err)

			s := NewProxy(Config{
				Port:                            "0",
				DecoderURL:                      prefillURL,
				KVConnector:                     constants.KVConnectorNIXLV2,
				MoRIIOWriteMode:                 true,
				MoRIIOParallelDispatch:          true,
				MoRIIOParallelDecodeWaitTimeout: 200 * time.Millisecond,
				MoRIIODecodePodIP:               "127.0.0.1",
				MoRIIODecodeNotifyPort:          61005,
				MoRIIODecodeHandshakePort:       6301,
				MoRIIOPrefillNotifyPort:         61006,
				MoRIIOPrefillHandshakePort:      6302,
				MoRIIOTPSize:                    1,
				MoRIIODPSize:                    1,
			})
			s.logger = log.Log
			s.decoderProxy = tt.decoder

			before := snapshotStageMetrics(t)
			beforeFields := toolFieldCounters(t)
			rw := httptest.NewRecorder()
			serve := func() {
				body := textChatBody()
				body[string(toolcalling.FieldTools)] = []any{}
				body[string(toolcalling.FieldToolChoice)] = "auto"
				s.handleNIXLV2(rw, chatRequest(t, body), prefillURL.Host, "", reqcommon.APITypeChatCompletions)
			}
			if tt.wantPanic {
				assert.PanicsWithValue(t, http.ErrAbortHandler, serve)
			} else {
				serve()
			}

			assert.Equal(t, tt.wantCode, rw.Code)
			assert.Equal(t, tt.wantDelta, snapshotStageMetrics(t).delta(before))
			fieldDelta := make(map[[2]string]float64)
			for key, value := range toolFieldCounters(t) {
				if change := value - beforeFields[key]; change != 0 {
					fieldDelta[key] = change
				}
			}
			assert.Equal(t, map[[2]string]float64{
				{string(toolcalling.FieldTools), string(toolcalling.FieldStatusPreserved)}:      1,
				{string(toolcalling.FieldToolChoice), string(toolcalling.FieldStatusPreserved)}: 1,
			}, fieldDelta)
		})
	}
}
