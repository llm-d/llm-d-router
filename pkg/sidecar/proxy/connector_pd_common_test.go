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
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/controller-runtime/pkg/log"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
)

// TestConcurrentPDCancellation covers which request a concurrent P/D failure
// cancels. Decode cannot receive KV after a failed prefill, and SGLang does not
// notify decode of it, so decode would otherwise wait for its KV transfer timeout.
func TestConcurrentPDCancellation(t *testing.T) {
	tests := []struct {
		name string
		// prefillStatus is returned once decode has started; zero blocks prefill until canceled.
		prefillStatus int
		// decode runs on the decode server once prefill has started.
		decode              func(w http.ResponseWriter, r *http.Request, cancelClient context.CancelFunc)
		wantPrefillCanceled bool
	}{
		{
			name:          "prefill failure cancels decode",
			prefillStatus: http.StatusServiceUnavailable,
			decode: func(_ http.ResponseWriter, r *http.Request, _ context.CancelFunc) {
				<-r.Context().Done()
			},
		},
		{
			name: "client disconnect cancels prefill",
			decode: func(_ http.ResponseWriter, _ *http.Request, cancelClient context.CancelFunc) {
				cancelClient()
			},
			wantPrefillCanceled: true,
		},
		{
			name: "successful decode leaves prefill running",
			decode: func(w http.ResponseWriter, _ *http.Request, _ context.CancelFunc) {
				w.WriteHeader(http.StatusOK)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 2*time.Second)
			defer cancel()

			prefillStarted := make(chan struct{})
			prefillCanceled := make(chan struct{})
			prefillRelease := make(chan struct{})
			decodeStarted := make(chan struct{})
			prefill := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = io.Copy(io.Discard, r.Body)
				close(prefillStarted)
				if tt.prefillStatus != 0 {
					select {
					case <-decodeStarted:
						w.WriteHeader(tt.prefillStatus)
					case <-r.Context().Done():
					}
					return
				}
				select {
				case <-r.Context().Done():
					close(prefillCanceled)
				case <-prefillRelease:
				}
			}))
			defer prefill.Close()
			decode := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = io.Copy(io.Discard, r.Body)
				close(decodeStarted)
				select {
				case <-prefillStarted:
					tt.decode(w, r, cancel)
				case <-r.Context().Done():
				}
			}))
			defer decode.Close()

			prefillURL, err := url.Parse(prefill.URL)
			require.NoError(t, err)
			decodeURL, err := url.Parse(decode.URL)
			require.NoError(t, err)
			s := NewProxy(Config{Port: "0", DecoderURL: decodeURL})
			s.decoderProxy = s.createDecoderProxyHandler(decodeURL, false)

			body := []byte(`{"input_ids":[1]}`)
			req := httptest.NewRequest(http.MethodPost, reqcommon.PathSGLangGenerate, bytes.NewReader(body)).WithContext(ctx)
			before := snapshotStageMetrics(t)
			s.runConcurrentPD(httptest.NewRecorder(), req, body, body, prefillURL.Host, constants.KVConnectorSGLang, nil, nil)
			require.NotErrorIs(t, ctx.Err(), context.DeadlineExceeded, "decode must not wait for the client deadline")

			// net/http cancels the request context after the handler returns.
			cancel()
			if tt.wantPrefillCanceled {
				select {
				case <-prefillCanceled:
				case <-time.After(time.Second):
					t.Fatal("client cancellation did not reach prefill")
				}
			} else {
				select {
				case <-prefillCanceled:
					t.Fatal("prefill canceled after decode returned")
				case <-time.After(20 * time.Millisecond):
				}
				close(prefillRelease)
			}
			// Wait for the prefill sample so it cannot leak into a later test's delta.
			require.Eventually(t, func() bool {
				return snapshotStageMetrics(t).delta(before).prefillCount == 1
			}, time.Second, 10*time.Millisecond)
		})
	}
}

const concurrentPDTestWait = 5 * time.Second

// heldPrefill is a prefill backend that holds its request until the request
// context ends and records that it was cancelled.
type heldPrefill struct {
	server    *httptest.Server
	started   chan struct{}
	cancelled atomic.Bool
}

func newHeldPrefill(t *testing.T) *heldPrefill {
	t.Helper()
	p := &heldPrefill{started: make(chan struct{})}
	p.server = httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		// net/http only watches for a closed connection once the body is consumed.
		_, _ = io.ReadAll(r.Body)
		close(p.started)
		<-r.Context().Done()
		p.cancelled.Store(true)
	}))
	t.Cleanup(p.server.Close)
	return p
}

func (p *heldPrefill) host() string {
	return strings.TrimPrefix(p.server.URL, "http://")
}

// requireCancelled waits for the backend to observe the cancelled connection,
// which net/http reports asynchronously.
func (p *heldPrefill) requireCancelled(t *testing.T) {
	t.Helper()
	require.Eventually(t, p.cancelled.Load, concurrentPDTestWait, 10*time.Millisecond,
		"prefill backend did not observe the cancellation")
}

// runConcurrentPDAgainst runs runConcurrentPD with a prefill backend that never
// responds and returns the stage metrics delta once the prefill duration sample,
// the last metric the prefill goroutine records, has been observed. wantPanic is
// the value the call is expected to panic with; nil for a normal return.
func runConcurrentPDAgainst(ctx context.Context, t *testing.T, prefill *heldPrefill, decoder http.Handler, wantPanic any) stageMetrics {
	t.Helper()
	decoderURL, err := url.Parse("http://decoder:8000")
	require.NoError(t, err)
	s := NewProxy(Config{Port: "0", DecoderURL: decoderURL})
	s.logger = log.Log
	s.decoderProxy = decoder

	body := []byte(`{"model":"m","messages":[]}`)
	req := httptest.NewRequest(http.MethodPost, reqcommon.PathChatCompletions, bytes.NewReader(body)).WithContext(ctx)
	before := snapshotStageMetrics(t)
	func() {
		defer func() { require.Equal(t, wantPanic, recover()) }()
		s.runConcurrentPD(httptest.NewRecorder(), req, body, body, prefill.host(), constants.KVConnectorMooncake, nil, nil)
	}()

	prefill.requireCancelled(t)
	var delta stageMetrics
	require.Eventually(t, func() bool {
		delta = snapshotStageMetrics(t).delta(before)
		return delta.prefillCount == 1
	}, concurrentPDTestWait, 10*time.Millisecond, "prefill sample was not recorded")
	return delta
}

// A decode that fails ends the request, so prefill is cancelled rather than
// left to run the whole prompt.
func TestRunConcurrentPDDecodeFailureCancelsPrefill(t *testing.T) {
	prefill := newHeldPrefill(t)
	decoder := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		<-prefill.started
		w.WriteHeader(http.StatusInternalServerError)
	})

	delta := runConcurrentPDAgainst(context.Background(), t, prefill, decoder, nil)
	assert.Equal(t, stageMetrics{prefillCount: 1, decodeCount: 1, decodeErrors: 1}, delta)
}

// A decode that aborts after sending headers (the reverse proxy panics with
// http.ErrAbortHandler) cancels prefill.
func TestRunConcurrentPDDecodeAbortCancelsPrefill(t *testing.T) {
	prefill := newHeldPrefill(t)
	decoder := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		<-prefill.started
		w.WriteHeader(http.StatusOK)
		panic(http.ErrAbortHandler)
	})

	delta := runConcurrentPDAgainst(context.Background(), t, prefill, decoder, http.ErrAbortHandler)
	assert.Equal(t, stageMetrics{prefillCount: 1, decodeCount: 1, decodeErrors: 1}, delta)
}

// A prefill cancelled because the client left is not a prefill error.
func TestRunConcurrentPDClientCancelIsNotPrefillError(t *testing.T) {
	prefill := newHeldPrefill(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	decoder := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		<-prefill.started
		cancel()
		w.WriteHeader(http.StatusOK)
	})

	delta := runConcurrentPDAgainst(ctx, t, prefill, decoder, nil)
	assert.Equal(t, stageMetrics{prefillCount: 1, decodeCount: 1}, delta)
}

// A prefill that outlives a successful decode by more than the grace period
// is cancelled.
func TestRunConcurrentPDGraceCancelsPrefill(t *testing.T) {
	previous := concurrentPrefillGrace
	concurrentPrefillGrace = 50 * time.Millisecond
	t.Cleanup(func() { concurrentPrefillGrace = previous })

	prefill := newHeldPrefill(t)
	decoder := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		<-prefill.started
		w.WriteHeader(http.StatusOK)
	})

	delta := runConcurrentPDAgainst(context.Background(), t, prefill, decoder, nil)
	assert.Equal(t, stageMetrics{prefillCount: 1, decodeCount: 1}, delta)
}
