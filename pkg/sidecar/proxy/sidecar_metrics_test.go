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
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/controller-runtime/pkg/log"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
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
	allTypes := []string{metrics.DisaggTypePD, metrics.DisaggTypeEPD, metrics.DisaggTypeED}

	tests := []struct {
		name   string
		header http.Header
		want   string
	}{
		{
			name:   "prefill only records prefill-decode",
			header: http.Header{prefillHeader: []string{"prefill1:8000"}},
			want:   metrics.DisaggTypePD,
		},
		{
			name: "encoder and prefill records encode-prefill-decode",
			header: http.Header{
				encoderHeader: []string{"enc1:8000"},
				prefillHeader: []string{"prefill1:8000"},
			},
			want: metrics.DisaggTypeEPD,
		},
		{
			name:   "encoder only records encode-decode",
			header: http.Header{encoderHeader: []string{"enc1:8000"}},
			want:   metrics.DisaggTypeED,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := NewProxy(Config{Port: "8000"})
			s.allowlistValidator = &AllowlistValidator{}
			s.handleECConnector = func(http.ResponseWriter, *http.Request, string, []string, reqcommon.APIType) {}
			s.handlePDConnector = func(http.ResponseWriter, *http.Request, string, string, reqcommon.APIType) {}
			s.decoderProxy = http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
				t.Error("decoder passthrough must not be reached for a disaggregated request")
			})

			before := map[string]float64{}
			for _, dt := range allTypes {
				before[dt] = disaggCount(t, dt)
			}

			s.disaggregatedPrefillHandler(reqcommon.APITypeChatCompletions)(
				httptest.NewRecorder(), &http.Request{Header: tt.header})

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
	outcomes := []struct {
		name          string
		encoderStatus int
		wantCode      int
		wantDecoder   bool
		wantDelta     stageMetrics
	}{
		{
			name:          "encoder 5xx records encode error without duration",
			encoderStatus: http.StatusInternalServerError,
			wantCode:      http.StatusBadGateway,
			wantDelta:     stageMetrics{encodeErrors: 1},
		},
		{
			name:          "encoder success records encode duration",
			encoderStatus: http.StatusOK,
			wantCode:      http.StatusOK,
			wantDecoder:   true,
			wantDelta:     stageMetrics{encodeCount: 1},
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
					w.WriteHeader(http.StatusOK)
				})

				before := snapshotStageMetrics(t)
				rw := httptest.NewRecorder()
				req := chatRequest(t, userMessageRequest(imageURLItem("https://example.com/img.jpg")))
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
			s.runConcurrentPD(tt.client(), req, body, body, prefillURL.Host, KVConnectorSGLang, nil)

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
		s.runConcurrentPD(httptest.NewRecorder(), req, body, body, prefillURL.Host, KVConnectorSGLang, nil)
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

func TestStatusCapturingResponseWriter(t *testing.T) {
	t.Run("write without header records implicit 200", func(t *testing.T) {
		w := &statusCapturingResponseWriter{ResponseWriter: httptest.NewRecorder()}
		_, err := w.Write([]byte("ok"))
		require.NoError(t, err)
		assert.Equal(t, http.StatusOK, w.statusCode)
		assert.False(t, w.failed())
	})
	t.Run("explicit error status is recorded and passed through", func(t *testing.T) {
		rec := httptest.NewRecorder()
		w := &statusCapturingResponseWriter{ResponseWriter: rec}
		w.WriteHeader(http.StatusServiceUnavailable)
		assert.Equal(t, http.StatusServiceUnavailable, w.statusCode)
		assert.Equal(t, http.StatusServiceUnavailable, rec.Code)
		assert.True(t, w.failed())
	})
	t.Run("first status wins", func(t *testing.T) {
		w := &statusCapturingResponseWriter{ResponseWriter: httptest.NewRecorder()}
		w.WriteHeader(http.StatusBadGateway)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("x"))
		assert.Equal(t, http.StatusBadGateway, w.statusCode)
		assert.True(t, w.failed())
	})
	t.Run("flush passes through to a flusher", func(t *testing.T) {
		fr := &flushRecorder{ResponseRecorder: httptest.NewRecorder()}
		w := &statusCapturingResponseWriter{ResponseWriter: fr}
		w.Flush()
		assert.Equal(t, 1, fr.flushes)
	})
	t.Run("flush is a no-op on a non-flusher", func(t *testing.T) {
		w := &statusCapturingResponseWriter{ResponseWriter: &failingWriter{header: http.Header{}}}
		assert.NotPanics(t, w.Flush)
	})
	t.Run("underlying write error marks failed despite 200", func(t *testing.T) {
		fw := &failingWriter{header: http.Header{}}
		w := &statusCapturingResponseWriter{ResponseWriter: fw}
		w.WriteHeader(http.StatusOK)
		_, err := w.Write([]byte("x"))
		require.Error(t, err)
		assert.Equal(t, http.StatusOK, w.statusCode)
		assert.True(t, w.failed())
	})
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
				KVConnector:                KVConnectorNIXLV2,
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
					if kv, ok := got[requestFieldKVTransferParams].(map[string]any); ok && kv[requestFieldDoRemotePrefill] == true {
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
