package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// errorCaptureSink is a logr.LogSink that records every Error call. Used to
// verify per-goroutine error visibility in the encoder fan-out path.
type errorCaptureSink struct {
	mu     sync.Mutex
	errors []capturedError
}

type capturedError struct {
	err error
	msg string
	kv  []any
}

func (c *errorCaptureSink) Init(_ logr.RuntimeInfo)        {}
func (c *errorCaptureSink) Enabled(_ int) bool             { return true }
func (c *errorCaptureSink) Info(_ int, _ string, _ ...any) {}
func (c *errorCaptureSink) Error(err error, msg string, kv ...any) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.errors = append(c.errors, capturedError{err: err, msg: msg, kv: kv})
}
func (c *errorCaptureSink) WithValues(_ ...any) logr.LogSink { return c }
func (c *errorCaptureSink) WithName(_ string) logr.LogSink   { return c }

func (c *errorCaptureSink) snapshot() []capturedError {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]capturedError, len(c.errors))
	copy(out, c.errors)
	return out
}

// TestFanoutEncoderCollectAggregates verifies that ec_transfer_params from
// each encoder response are merged into a single flat map keyed by the
// per-image mm_hash.
func TestFanoutEncoderCollectAggregates(t *testing.T) {
	var seq atomic.Int32
	// Each encoder response carries a distinct hash key so the merged map
	// retains both entries instead of collapsing under a shared key.
	encoderBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		i := seq.Add(1) - 1
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprintf(w, `{
			"choices": [{"message": {"content": ""}}],
			"ec_transfer_params": {"hash-%d": {"peer_host": "10.0.0.%d"}}
		}`, i, i)
	}))
	defer encoderBackend.Close()

	encoderURL, err := url.Parse(encoderBackend.URL)
	assert.NoError(t, err)
	srv := NewProxy(Config{Port: "0", DecoderURL: encoderURL})
	srv.logger = log.Log

	req := userMessageRequest(
		imageURLItem("https://example.com/img1.jpg"),
		imageURLItem("https://example.com/img2.jpg"),
	)

	params, contributed, total, err := srv.fanoutEncoderCollect(context.Background(), req, []string{encoderURL.Host}, "test-req-id")
	assert.NoError(t, err)
	assert.Equal(t, 2, total, "total item count")
	assert.Equal(t, 2, contributed, "both encoder responses carried ec_transfer_params")
	assert.Len(t, params, 2, "one flat-map entry per distinct hash")
	for k, v := range params {
		entry, ok := v.(map[string]any)
		assert.Truef(t, ok, "params[%q] should be a map", k)
		assert.Containsf(t, entry, "peer_host", "params[%q] should carry transfer metadata", k)
	}
}

// TestFanoutEncoderCollectMissingField verifies the warn-and-continue
// behaviour: an encoder response without ec_transfer_params yields a nil
// slot but does NOT fail the request. Mirrors NIXLv2's tolerance for
// missing kv_transfer_params.
func TestFanoutEncoderCollectMissingField(t *testing.T) {
	encoderBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// No ec_transfer_params field.
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":""}}]}`))
	}))
	defer encoderBackend.Close()

	encoderURL, err := url.Parse(encoderBackend.URL)
	assert.NoError(t, err)
	srv := NewProxy(Config{Port: "0", DecoderURL: encoderURL})
	srv.logger = log.Log

	req := userMessageRequest(imageURLItem("https://example.com/img.jpg"))
	params, contributed, total, err := srv.fanoutEncoderCollect(context.Background(), req, []string{encoderURL.Host}, "test-req-id")
	assert.NoError(t, err, "missing ec_transfer_params must not fail the request")
	assert.Equal(t, 1, total, "one item processed")
	assert.Equal(t, 0, contributed, "no encoder response carried ec_transfer_params")
	assert.Empty(t, params, "no entries merged into the flat map")
}

// TestFanoutEncoderCollectEncoderError verifies that a non-2xx encoder
// response is hard-fail (consistent with fanoutEncoderPrimer).
func TestFanoutEncoderCollectEncoderError(t *testing.T) {
	encoderBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":"boom"}`))
	}))
	defer encoderBackend.Close()

	encoderURL, err := url.Parse(encoderBackend.URL)
	assert.NoError(t, err)
	srv := NewProxy(Config{Port: "0", DecoderURL: encoderURL})
	srv.logger = log.Log

	req := userMessageRequest(imageURLItem("https://example.com/img.jpg"))
	_, _, _, err = srv.fanoutEncoderCollect(context.Background(), req, []string{encoderURL.Host}, "test-req-id")
	assert.Error(t, err, "5xx from encoder must surface as an error")
}

// TestHandleECEPDThreadsParamsToPrefill verifies that handleECNIXL mutates
// the prefill request body to carry a flat ec_transfer_params map keyed by
// the per-image mm_hash, and sets cache_hit_threshold=0. Bypasses the
// real P/D connector by stubbing s.handlePDConnector. The contract under
// test is "what gets handed to the P/D connector", not the P/D connector's
// downstream behaviour.
func TestHandleECEPDThreadsParamsToPrefill(t *testing.T) {
	var seq atomic.Int32
	// Distinct hash key per encoder response so the merged map retains
	// both entries instead of collapsing under a shared key.
	encoderBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		i := seq.Add(1) - 1
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprintf(w, `{
			"choices": [{"message": {"content": ""}}],
			"ec_transfer_params": {"hash-%d": {"peer_host": "10.0.0.%d", "peer_port": 5500}}
		}`, i, i)
	}))
	defer encoderBackend.Close()

	encoderURL, err := url.Parse(encoderBackend.URL)
	assert.NoError(t, err)
	srv := NewProxy(Config{Port: "0", DecoderURL: encoderURL})
	srv.logger = log.Log

	// Capture what handleECNIXL hands to the P/D connector instead of
	// running real prefill→decode plumbing.
	var capturedBody []byte
	srv.handlePDConnector = func(_ http.ResponseWriter, r *http.Request, _ string, _ APIType) {
		buf, err := io.ReadAll(r.Body)
		assert.NoError(t, err)
		capturedBody = buf
	}

	reqBody, _ := json.Marshal(userMessageRequest(
		imageURLItem("https://example.com/img1.jpg"),
		imageURLItem("https://example.com/img2.jpg"),
	))
	httpReq := httptest.NewRequest(http.MethodPost, ChatCompletionsPath, io.NopCloser(bytes.NewReader(reqBody)))
	rw := httptest.NewRecorder()

	srv.handleECNIXL(rw, httpReq, "fake-prefiller:8000", []string{encoderURL.Host})

	if !assert.NotNil(t, capturedBody, "handlePDConnector should have been invoked") {
		return
	}
	var parsed map[string]any
	assert.NoError(t, json.Unmarshal(capturedBody, &parsed))

	ec, ok := parsed[requestFieldECTransferParams].(map[string]any)
	assert.True(t, ok, "prefill body should carry ec_transfer_params as an object")
	assert.Len(t, ec, 2, "one entry per distinct hash from the encoder responses")
	for k, v := range ec {
		entry, ok := v.(map[string]any)
		assert.Truef(t, ok, "ec[%q] should be an object", k)
		assert.Containsf(t, entry, "peer_host", "ec[%q] should carry transfer metadata", k)
	}

	threshold, ok := parsed[requestFieldCacheHitThreshold]
	assert.True(t, ok, "cache_hit_threshold should be set")
	// JSON numbers unmarshal to float64.
	assert.Equal(t, float64(0), threshold, "cache_hit_threshold should be 0")
}

// TestHandleECEPDAllMissingDoesNotAddField verifies the all-missing
// degradation branch: when every encoder response lacks ec_transfer_params,
// the connector forwards the prefill request WITHOUT the field set (so
// downstream behaviour matches primer-mode rather than threading nils).
// cache_hit_threshold should still be set.
func TestHandleECEPDAllMissingDoesNotAddField(t *testing.T) {
	encoderBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Encoder 2xx but no ec_transfer_params.
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":""}}]}`))
	}))
	defer encoderBackend.Close()

	encoderURL, err := url.Parse(encoderBackend.URL)
	assert.NoError(t, err)
	srv := NewProxy(Config{Port: "0", DecoderURL: encoderURL})
	srv.logger = log.Log

	var capturedBody []byte
	srv.handlePDConnector = func(_ http.ResponseWriter, r *http.Request, _ string, _ APIType) {
		buf, err := io.ReadAll(r.Body)
		assert.NoError(t, err)
		capturedBody = buf
	}

	reqBody, _ := json.Marshal(userMessageRequest(
		imageURLItem("https://example.com/img1.jpg"),
		imageURLItem("https://example.com/img2.jpg"),
	))
	httpReq := httptest.NewRequest(http.MethodPost, ChatCompletionsPath, io.NopCloser(bytes.NewReader(reqBody)))
	rw := httptest.NewRecorder()

	srv.handleECNIXL(rw, httpReq, "fake-prefiller:8000", []string{encoderURL.Host})

	if !assert.NotNil(t, capturedBody, "handlePDConnector should have been invoked") {
		return
	}
	var parsed map[string]any
	assert.NoError(t, json.Unmarshal(capturedBody, &parsed))

	_, ok := parsed[requestFieldECTransferParams]
	assert.False(t, ok, "prefill body must NOT carry ec_transfer_params when all encoder responses lacked it")

	threshold, ok := parsed[requestFieldCacheHitThreshold]
	assert.True(t, ok, "cache_hit_threshold should still be set even when ec params are absent")
	assert.Equal(t, float64(0), threshold)
}

// TestHandleECEPDPartiallyPopulated verifies the partial-populated branch:
// when some encoder responses carry ec_transfer_params and others don't,
// the connector attaches the flat map containing only the contributed
// entries (missing items contribute no keys).
//
// The backend routes by the per-item x-request-id header
// (format "<reqID>-enc-<idx>") set by fanoutEncoder. Item 0 returns params;
// item 1 does not. This is deterministic regardless of goroutine scheduling.
func TestHandleECEPDPartiallyPopulated(t *testing.T) {
	encoderBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Parse the item index from the trailing "-enc-<idx>" suffix.
		reqID := r.Header.Get("x-request-id")
		var idx int
		_, _ = fmt.Sscanf(reqID[len(reqID)-1:], "%d", &idx)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		if idx == 0 {
			_, _ = fmt.Fprintf(w, `{
				"choices": [{"message": {"content": ""}}],
				"ec_transfer_params": {"hash-0": {"peer_host": "10.0.0.0"}}
			}`)
		} else {
			_, _ = w.Write([]byte(`{"choices":[{"message":{"content":""}}]}`))
		}
	}))
	defer encoderBackend.Close()

	encoderURL, err := url.Parse(encoderBackend.URL)
	assert.NoError(t, err)
	srv := NewProxy(Config{Port: "0", DecoderURL: encoderURL})
	srv.logger = log.Log

	var capturedBody []byte
	srv.handlePDConnector = func(_ http.ResponseWriter, r *http.Request, _ string, _ APIType) {
		buf, err := io.ReadAll(r.Body)
		assert.NoError(t, err)
		capturedBody = buf
	}

	reqBody, _ := json.Marshal(userMessageRequest(
		imageURLItem("https://example.com/img1.jpg"),
		imageURLItem("https://example.com/img2.jpg"),
	))
	httpReq := httptest.NewRequest(http.MethodPost, ChatCompletionsPath, io.NopCloser(bytes.NewReader(reqBody)))
	rw := httptest.NewRecorder()

	srv.handleECNIXL(rw, httpReq, "fake-prefiller:8000", []string{encoderURL.Host})

	if !assert.NotNil(t, capturedBody, "handlePDConnector should have been invoked") {
		return
	}
	var parsed map[string]any
	assert.NoError(t, json.Unmarshal(capturedBody, &parsed))

	ec, ok := parsed[requestFieldECTransferParams].(map[string]any)
	assert.True(t, ok, "prefill body should carry ec_transfer_params (at least one item populated)")
	assert.Len(t, ec, 1, "only item 0 contributes a hash key; item 1 has no ec_transfer_params")
	entry, ok := ec["hash-0"].(map[string]any)
	assert.True(t, ok, "ec[\"hash-0\"] should be an object")
	assert.Contains(t, entry, "peer_host", "ec[\"hash-0\"] should carry transfer metadata")
}

// TestFanoutEncoderFailFastCancellation verifies the errgroup fail-fast
// behavior: when one encoder goroutine returns an error, the function must
// return well before a slow sibling would finish on its own. The test uses
// wall-clock time as the observable signal.
//
// A barrier (allConnected) ensures both backend handlers have started before
// the failing one responds, preventing the race where the error goroutine
// cancels gctx before the sibling even connects.
func TestFanoutEncoderFailFastCancellation(t *testing.T) {
	const slowEncoderDelay = 5 * time.Second

	var seq atomic.Int32
	allConnected := make(chan struct{})

	encoderBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		idx := int(seq.Add(1) - 1)
		if idx == 1 {
			close(allConnected)
		}
		<-allConnected // both handlers wait until second has started

		if idx == 0 {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"error":"boom"}`))
			return
		}
		// Sibling: block until client drops the connection or the slow timeout
		// expires, whichever comes first. Without fail-fast this takes
		// slowEncoderDelay; with fail-fast the transport cancels the request
		// and grp.Wait() returns before the timeout fires.
		select {
		case <-r.Context().Done():
		case <-time.After(slowEncoderDelay):
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{}`))
	}))
	defer encoderBackend.Close()

	encoderURL, err := url.Parse(encoderBackend.URL)
	assert.NoError(t, err)
	srv := NewProxy(Config{Port: "0", DecoderURL: encoderURL})
	srv.logger = log.Log

	req := userMessageRequest(
		imageURLItem("https://example.com/img1.jpg"),
		imageURLItem("https://example.com/img2.jpg"),
	)

	start := time.Now()
	_, _, _, err = srv.fanoutEncoderCollect(context.Background(), req, []string{encoderURL.Host}, "test-cancel")
	elapsed := time.Since(start)

	assert.Error(t, err, "5xx from one encoder must surface as an error")
	assert.Less(t, elapsed, slowEncoderDelay/2,
		"fanoutEncoderCollect must return before half the slow sibling's delay (%s); got %s", slowEncoderDelay, elapsed)
}

// TestFanoutEncoderAllFail verifies that when every encoder goroutine returns
// an error, fanoutEncoderCollect still surfaces an error (the first one).
func TestFanoutEncoderAllFail(t *testing.T) {
	encoderBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":"boom"}`))
	}))
	defer encoderBackend.Close()

	encoderURL, err := url.Parse(encoderBackend.URL)
	assert.NoError(t, err)
	srv := NewProxy(Config{Port: "0", DecoderURL: encoderURL})
	srv.logger = log.Log

	req := userMessageRequest(
		imageURLItem("https://example.com/img1.jpg"),
		imageURLItem("https://example.com/img2.jpg"),
		imageURLItem("https://example.com/img3.jpg"),
	)
	_, _, _, err = srv.fanoutEncoderCollect(context.Background(), req, []string{encoderURL.Host}, "test-all-fail")
	assert.Error(t, err, "all-fail must surface an error")
}

// TestFanoutEncoderParentContextCancel verifies that canceling the caller's
// context (r.Context() from handleECNIXL) propagates through errgroup's derived
// context to the in-flight HTTP requests, causing fanoutEncoderCollect to return
// early rather than waiting for slow encoders.
func TestFanoutEncoderParentContextCancel(t *testing.T) {
	const slowEncoderDelay = 5 * time.Second

	encoderBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Block until the client cancels or the slow timeout fires.
		select {
		case <-r.Context().Done():
		case <-time.After(slowEncoderDelay):
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{}`))
	}))
	defer encoderBackend.Close()

	encoderURL, err := url.Parse(encoderBackend.URL)
	assert.NoError(t, err)
	srv := NewProxy(Config{Port: "0", DecoderURL: encoderURL})
	srv.logger = log.Log

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	req := userMessageRequest(imageURLItem("https://example.com/img.jpg"))

	// Cancel the parent context after a short delay to simulate client disconnection.
	go func() {
		time.Sleep(100 * time.Millisecond)
		cancel()
	}()

	start := time.Now()
	_, _, _, err = srv.fanoutEncoderCollect(ctx, req, []string{encoderURL.Host}, "test-ctx-cancel")
	elapsed := time.Since(start)

	assert.Error(t, err, "canceled parent context must surface as an error")
	assert.Less(t, elapsed, slowEncoderDelay/2,
		"fanoutEncoderCollect must return after parent context cancellation, not wait for slow encoder; got %s", elapsed)
}

// TestFanoutEncoderPerErrorVisibility verifies the "sibling visibility"
// guarantee: every failing goroutine logs its own error before returning,
// even though grp.Wait surfaces only the first error to the caller.
//
// The reviewer concern: the previous implementation used a buffered errChan
// and only the error that won the race was reachable. Operators lost N-1
// errors to silent discard. This test asserts each failed item produces a
// distinct log entry.
func TestFanoutEncoderPerErrorVisibility(t *testing.T) {
	// Barrier: hold all encoder responses until every goroutine has connected.
	// Without this barrier, fail-fast cancellation could prevent later
	// goroutines from ever issuing their failing response, defeating the
	// "every failure is logged" assertion.
	var connected atomic.Int32
	allConnected := make(chan struct{})
	const failingItems = 3

	encoderBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if connected.Add(1) == failingItems {
			close(allConnected)
		}
		<-allConnected
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":"boom"}`))
	}))
	defer encoderBackend.Close()

	encoderURL, err := url.Parse(encoderBackend.URL)
	assert.NoError(t, err)
	srv := NewProxy(Config{Port: "0", DecoderURL: encoderURL})

	sink := &errorCaptureSink{}
	srv.logger = logr.New(sink)

	req := userMessageRequest(
		imageURLItem("https://example.com/img1.jpg"),
		imageURLItem("https://example.com/img2.jpg"),
		imageURLItem("https://example.com/img3.jpg"),
	)
	_, _, _, err = srv.fanoutEncoderCollect(context.Background(), req, []string{encoderURL.Host}, "test-visibility")
	assert.Error(t, err, "all encoders return 5xx; an error must surface")

	captured := sink.snapshot()

	// Build the set of item indices observed in error log lines.
	seenItems := make(map[int]struct{})
	for _, e := range captured {
		// kv is "encoder fanout" key-value pairs: ["item", idx, "requestID", ...]
		for i := 0; i+1 < len(e.kv); i += 2 {
			if e.kv[i] == "item" {
				if idx, ok := e.kv[i+1].(int); ok {
					seenItems[idx] = struct{}{}
				}
			}
		}
	}
	assert.Lenf(t, seenItems, failingItems,
		"expected one error log per failed item; got logs for items %v (raw=%d entries)", seenItems, len(captured))
}

// TestFanoutEncoderCollectEmptyHostPorts verifies that fanoutEncoder returns
// an error immediately when encoderHostPorts is empty, rather than panicking
// with a division-by-zero on the round-robin index.
func TestFanoutEncoderCollectEmptyHostPorts(t *testing.T) {
	srv := NewProxy(Config{Port: "0"})
	srv.logger = log.Log

	req := userMessageRequest(imageURLItem("https://example.com/img.jpg"))
	_, _, _, err := srv.fanoutEncoderCollect(context.Background(), req, nil, "test-empty-hosts")
	assert.Error(t, err, "empty encoderHostPorts must return an error, not panic")
}

// TestHandleECNIXLEmptyEncodeEndPoints verifies that handleECNIXL skips the
// encoder block entirely and calls runPDPipeline directly when encodeEndPoints
// is empty. The prefill request must still reach the P/D connector unmodified
// (no ec_transfer_params), but cache_hit_threshold must be set to 0.
func TestHandleECNIXLEmptyEncodeEndPoints(t *testing.T) {
	decoderBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer decoderBackend.Close()

	decoderURL, err := url.Parse(decoderBackend.URL)
	assert.NoError(t, err)
	srv := NewProxy(Config{Port: "0", DecoderURL: decoderURL})
	srv.logger = log.Log

	var capturedBody []byte
	srv.handlePDConnector = func(_ http.ResponseWriter, r *http.Request, _ string, _ APIType) {
		buf, err := io.ReadAll(r.Body)
		assert.NoError(t, err)
		capturedBody = buf
	}

	reqBody, _ := json.Marshal(userMessageRequest(imageURLItem("https://example.com/img.jpg")))
	httpReq := httptest.NewRequest(http.MethodPost, ChatCompletionsPath, io.NopCloser(bytes.NewReader(reqBody)))
	rw := httptest.NewRecorder()

	srv.handleECNIXL(rw, httpReq, "fake-prefiller:8000", nil)

	if !assert.NotNil(t, capturedBody, "handlePDConnector should have been invoked") {
		return
	}
	var parsed map[string]any
	assert.NoError(t, json.Unmarshal(capturedBody, &parsed))
	_, hasEC := parsed[requestFieldECTransferParams]
	assert.False(t, hasEC, "ec_transfer_params must not be set when no encoders were called")
	threshold, ok := parsed[requestFieldCacheHitThreshold]
	assert.True(t, ok, "cache_hit_threshold must be set")
	assert.Equal(t, float64(0), threshold)
}

// TestHandleECNIXLTextOnlyRequest verifies the text-only path: when the
// request body has no multimodal items, fanoutEncoderCollect returns total=0
// and the encoder block is skipped. The request still reaches the P/D
// connector with cache_hit_threshold=0 but without ec_transfer_params.
func TestHandleECNIXLTextOnlyRequest(t *testing.T) {
	// Encoder backend should never be called for a text-only request.
	encoderCalled := false
	encoderBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		encoderCalled = true
		w.WriteHeader(http.StatusOK)
	}))
	defer encoderBackend.Close()

	encoderURL, err := url.Parse(encoderBackend.URL)
	assert.NoError(t, err)
	srv := NewProxy(Config{Port: "0", DecoderURL: encoderURL})
	srv.logger = log.Log

	var capturedBody []byte
	srv.handlePDConnector = func(_ http.ResponseWriter, r *http.Request, _ string, _ APIType) {
		buf, err := io.ReadAll(r.Body)
		assert.NoError(t, err)
		capturedBody = buf
	}

	// Text-only message: no image_url / audio_url / video_url items.
	textOnly := map[string]any{
		"model": "test-model",
		"messages": []any{
			map[string]any{"role": "user", "content": "hello world"},
		},
	}
	reqBody, _ := json.Marshal(textOnly)
	httpReq := httptest.NewRequest(http.MethodPost, ChatCompletionsPath, io.NopCloser(bytes.NewReader(reqBody)))
	rw := httptest.NewRecorder()

	srv.handleECNIXL(rw, httpReq, "fake-prefiller:8000", []string{encoderURL.Host})

	assert.False(t, encoderCalled, "encoder backend must not be called for a text-only request")
	if !assert.NotNil(t, capturedBody, "handlePDConnector should have been invoked") {
		return
	}
	var parsed map[string]any
	assert.NoError(t, json.Unmarshal(capturedBody, &parsed))
	_, hasEC := parsed[requestFieldECTransferParams]
	assert.False(t, hasEC, "ec_transfer_params must not be set for a text-only request")
	threshold, ok := parsed[requestFieldCacheHitThreshold]
	assert.True(t, ok, "cache_hit_threshold must be set")
	assert.Equal(t, float64(0), threshold)
}

// TestHandleECNIXLDecoderDirect verifies the decoder-direct branch of
// runPDPipeline: when prefillEndPoint is empty, the request bypasses
// handlePDConnector and goes straight to the decoder proxy. handlePDConnector
// must not be called.
func TestHandleECNIXLDecoderDirect(t *testing.T) {
	var decoderBody []byte
	decoderBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		buf, err := io.ReadAll(r.Body)
		if err == nil {
			decoderBody = buf
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"choices":[]}`))
	}))
	defer decoderBackend.Close()

	encoderBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":""}}],"ec_transfer_params":{"hash-0":{"peer_host":"10.0.0.1"}}}`))
	}))
	defer encoderBackend.Close()

	decoderURL, err := url.Parse(decoderBackend.URL)
	assert.NoError(t, err)
	encoderURL, err := url.Parse(encoderBackend.URL)
	assert.NoError(t, err)

	srv := NewProxy(Config{Port: "0", DecoderURL: decoderURL})
	srv.logger = log.Log
	srv.decoderProxy = httputil.NewSingleHostReverseProxy(decoderURL)

	pdConnectorCalled := false
	srv.handlePDConnector = func(_ http.ResponseWriter, _ *http.Request, _ string, _ APIType) {
		pdConnectorCalled = true
	}

	reqBody, _ := json.Marshal(userMessageRequest(imageURLItem("https://example.com/img.jpg")))
	httpReq := httptest.NewRequest(http.MethodPost, ChatCompletionsPath, io.NopCloser(bytes.NewReader(reqBody)))
	rw := httptest.NewRecorder()

	// Empty prefillEndPoint triggers the decoder-direct branch.
	srv.handleECNIXL(rw, httpReq, "", []string{encoderURL.Host})

	assert.False(t, pdConnectorCalled, "handlePDConnector must not be called when prefillEndPoint is empty")
	if !assert.NotNil(t, decoderBody, "decoder backend should have received the request") {
		return
	}
	var parsed map[string]any
	assert.NoError(t, json.Unmarshal(decoderBody, &parsed))
	_, hasEC := parsed[requestFieldECTransferParams]
	assert.True(t, hasEC, "decoder-direct request should carry ec_transfer_params from encoder")
	threshold, ok := parsed[requestFieldCacheHitThreshold]
	assert.True(t, ok, "cache_hit_threshold must be set")
	assert.Equal(t, float64(0), threshold)
}
