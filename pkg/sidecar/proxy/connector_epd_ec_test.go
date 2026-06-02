package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

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

	params, contributed, total, err := srv.fanoutEncoderCollect(req, []string{encoderURL.Host}, "test-req-id")
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
	params, contributed, total, err := srv.fanoutEncoderCollect(req, []string{encoderURL.Host}, "test-req-id")
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
	_, _, _, err = srv.fanoutEncoderCollect(req, []string{encoderURL.Host}, "test-req-id")
	assert.Error(t, err, "5xx from encoder must surface as an error")
}

// TestHandleECEPDThreadsParamsToPrefill verifies that handleECEPD mutates
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

	// Capture what handleECEPD hands to the P/D connector instead of
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

	srv.handleECEPD(rw, httpReq, "fake-prefiller:8000", []string{encoderURL.Host})

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

	srv.handleECEPD(rw, httpReq, "fake-prefiller:8000", []string{encoderURL.Host})

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
func TestHandleECEPDPartiallyPopulated(t *testing.T) {
	// Alternate: first request returns params, second doesn't, etc.
	var seq atomic.Int32
	encoderBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		i := seq.Add(1) - 1
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		if i%2 == 0 {
			_, _ = fmt.Fprintf(w, `{
				"choices": [{"message": {"content": ""}}],
				"ec_transfer_params": {"hash-%d": {"peer_host": "10.0.0.%d"}}
			}`, i, i)
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

	srv.handleECEPD(rw, httpReq, "fake-prefiller:8000", []string{encoderURL.Host})

	if !assert.NotNil(t, capturedBody, "handlePDConnector should have been invoked") {
		return
	}
	var parsed map[string]any
	assert.NoError(t, json.Unmarshal(capturedBody, &parsed))

	ec, ok := parsed[requestFieldECTransferParams].(map[string]any)
	assert.True(t, ok, "prefill body should carry ec_transfer_params (at least one item populated)")
	assert.Len(t, ec, 1, "only the populated encoder response contributes a hash key")
	for k, v := range ec {
		entry, ok := v.(map[string]any)
		assert.Truef(t, ok, "ec[%q] should be an object", k)
		assert.Containsf(t, entry, "peer_host", "ec[%q] should carry transfer metadata", k)
	}
}
