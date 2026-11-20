/*
Copyright 2025 The llm-d Authors.

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
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	. "github.com/onsi/ginkgo/v2" //nolint:revive
	. "github.com/onsi/gomega"    //nolint:revive
	"github.com/stretchr/testify/require"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
)

var _ = Describe("Mooncake Connector", func() {

	var testInfo *sidecarTestInfo
	var bootstrapServer *httptest.Server
	var bootstrapResponse string

	BeforeEach(func() {
		// single dp rank bootstrap response
		bootstrapResponse = `{"0": {"engine_id": "test-engine-abc123", "worker_addr": {"0": {"0": "10.0.0.1:5000"}}}}`

		// start a mock bootstrap server that returns map
		bootstrapServer = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(bootstrapResponse))
		}))
		DeferCleanup(bootstrapServer.Close)

		// Extract the bootstrap server port
		bootstrapURL, err := url.Parse(bootstrapServer.URL)
		Expect(err).ToNot(HaveOccurred())
		var bootstrapPort int
		_, err = fmt.Sscanf(bootstrapURL.Port(), "%d", &bootstrapPort)
		Expect(err).ToNot(HaveOccurred())

		testInfo = sidecarConnectionTestSetup(constants.KVConnectorMooncake)
		testInfo.proxy.config.MooncakeBootstrapPort = bootstrapPort
	})

	It("should send concurrent requests with correct mooncake kv_transfer_params", func() {
		proxyBaseAddr := testInfo.startProxy()

		body := chatCompletionsRequestBodyWithMaxCompletionCap
		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+reqcommon.PathChatCompletions, bytes.NewReader([]byte(body)))
		Expect(err).ToNot(HaveOccurred())

		// Use the bootstrap server's host as the prefill host so engine_id discovery works
		prefillHostPort := testInfo.prefillBackend.URL[len("http://"):]
		req.Header.Add(routing.PrefillEndpointHeader, prefillHostPort)

		resp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())

		if resp.StatusCode != 200 {
			bp, _ := io.ReadAll(resp.Body)
			Fail(string(bp))
		}

		// Wait for async prefill request to be fully recorded
		Eventually(func() int {
			return len(testInfo.prefillHandler.GetCompletionRequests())
		}).Should(Equal(1))

		// Validate prefill request
		prefillReqs := testInfo.prefillHandler.GetCompletionRequests()
		Expect(prefillReqs).To(HaveLen(1))
		preq := prefillReqs[0]

		// Prefill should have kv_transfer_params with do_remote_decode=true
		Expect(preq).To(HaveKey(reqcommon.FieldKVTransferParams))
		prefillKVParams, ok := preq[reqcommon.FieldKVTransferParams].(map[string]any)
		Expect(ok).To(BeTrue())
		Expect(prefillKVParams[reqcommon.FieldDoRemoteDecode]).To(BeTrue())
		Expect(prefillKVParams[reqcommon.FieldDoRemotePrefill]).To(BeFalse())
		Expect(prefillKVParams[requestFieldTransferID]).ToNot(BeEmpty())

		// Prefill should have max_tokens=1, max_completion_tokens=1 and stream=false
		Expect(preq[reqcommon.FieldMaxTokens]).To(BeNumerically("==", 1))
		Expect(preq).To(HaveKeyWithValue(reqcommon.FieldMaxCompletionTokens, BeNumerically("==", 1)))
		Expect(preq[reqcommon.FieldStream]).To(BeFalse())

		// Validate decode request
		Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		decodeReqs := testInfo.decodeHandler.GetCompletionRequests()
		Expect(decodeReqs).To(HaveLen(1))
		dreq := decodeReqs[0]

		// Decode should have kv_transfer_params with do_remote_prefill=true
		Expect(dreq).To(HaveKey(reqcommon.FieldKVTransferParams))
		decodeKVParams, ok := dreq[reqcommon.FieldKVTransferParams].(map[string]any)
		Expect(ok).To(BeTrue())
		Expect(decodeKVParams[reqcommon.FieldDoRemotePrefill]).To(BeTrue())
		Expect(decodeKVParams[reqcommon.FieldDoRemoteDecode]).To(BeFalse())
		Expect(decodeKVParams[requestFieldTransferID]).To(HavePrefix("xfer-"))
		Expect(decodeKVParams[reqcommon.FieldRemoteEngineID]).To(Equal("test-engine-abc123"))
		Expect(decodeKVParams[requestFieldRemoteBootstrapAddr]).To(ContainSubstring(fmt.Sprintf(":%d", testInfo.proxy.config.MooncakeBootstrapPort)))

		// Transfer IDs must match between prefill and decode
		Expect(decodeKVParams[requestFieldTransferID]).To(Equal(prefillKVParams[requestFieldTransferID]))

		// Prefill must be pinned to the dp rank whose engine_id is sent to decode
		prefillHeaders := testInfo.prefillHandler.GetCompletionHeaders()
		Expect(prefillHeaders).To(HaveLen(1))
		Expect(prefillHeaders[0].Get(requestHeaderDataParallelRank)).To(Equal("0"))

		// Decode should preserve original max_tokens and max_completion_tokens from request
		Expect(dreq[reqcommon.FieldMaxTokens]).To(BeNumerically("==", 50))
		Expect(dreq).To(HaveKeyWithValue(reqcommon.FieldMaxCompletionTokens, BeNumerically("==", 100)))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	It("should strip min_tokens from the prefill request and restore it in decode", func() {
		proxyBaseAddr := testInfo.startProxy()

		body := chatCompletionsRequestBodyWithMinCap
		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+reqcommon.PathChatCompletions, bytes.NewReader([]byte(body)))
		Expect(err).ToNot(HaveOccurred())

		prefillHostPort := testInfo.prefillBackend.URL[len("http://"):]
		req.Header.Add(routing.PrefillEndpointHeader, prefillHostPort)

		resp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())
		if resp.StatusCode != 200 {
			bp, _ := io.ReadAll(resp.Body)
			Fail(string(bp))
		}

		Eventually(func() int {
			return len(testInfo.prefillHandler.GetCompletionRequests())
		}).Should(Equal(1))

		preq := testInfo.prefillHandler.GetCompletionRequests()[0]
		Expect(preq[reqcommon.FieldMaxTokens]).To(BeNumerically("==", 1))
		Expect(preq).ToNot(HaveKey(reqcommon.FieldMinTokens))

		Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		dreq := testInfo.decodeHandler.GetCompletionRequests()[0]
		Expect(dreq).To(HaveKeyWithValue(reqcommon.FieldMinTokens, BeNumerically("==", 5)))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	It("should cap sampling_params in the prefill request and restore originals in decode", func() {
		expectGenerateRequestTokenLimits(testInfo)

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	It("should not panic when prefill response is slower than decode response", func() {
		// Stop previously injected servers
		testInfo.decodeBackend.Close()
		testInfo.prefillBackend.Close()

		var prefillFinished atomic.Bool

		// create a delay on prefill
		slowPrefill := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			testInfo.prefillHandler.ServeHTTP(w, r)
			time.Sleep(300 * time.Millisecond)
			prefillFinished.Store(true)
		})
		testInfo.prefillBackend = httptest.NewServer(slowPrefill)

		fastDecode := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			testInfo.decodeHandler.ServeHTTP(w, r)
		})
		testInfo.decodeBackend = httptest.NewServer(fastDecode)
		testInfo.decodeURL, _ = url.Parse(testInfo.decodeBackend.URL)

		var bootstrapPort int
		bURL, _ := url.Parse(bootstrapServer.URL)
		_, _ = fmt.Sscanf(bURL.Port(), "%d", &bootstrapPort)

		cfg := Config{
			Port:                  "0",
			DecoderURL:            testInfo.decodeURL,
			KVConnector:           constants.KVConnectorMooncake,
			MooncakeBootstrapPort: bootstrapPort,
		}
		testInfo.proxy = NewProxy(cfg)

		proxyBaseAddr := testInfo.startProxy()

		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+reqcommon.PathChatCompletions, bytes.NewReader([]byte(chatCompletionsRequestBody)))
		Expect(err).ToNot(HaveOccurred())

		prefillHostPort := testInfo.prefillBackend.URL[len("http://"):]
		req.Header.Add(routing.PrefillEndpointHeader, prefillHostPort)

		resp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())
		if resp.StatusCode != 200 {
			bp, _ := io.ReadAll(resp.Body)
			Fail(string(bp))
		}

		Eventually(prefillFinished.Load).Should(BeTrue()) // use default timeout from gomega
		Expect(testInfo.prefillHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	It("should pin prefill to the selected dp rank and send its engine_id to decode", func() {
		engineByRank := map[string]string{"0": "engine-dp0", "1": "engine-dp1"}
		bootstrapResponse = `{
			"0": {"engine_id": "engine-dp0", "worker_addr": {"0": {"0": "10.0.0.1:5000"}}},
			"1": {"engine_id": "engine-dp1", "worker_addr": {"0": {"0": "10.0.0.2:5000"}}}
		}`

		proxyBaseAddr := testInfo.startProxy()

		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+reqcommon.PathChatCompletions, bytes.NewReader([]byte(chatCompletionsRequestBody)))
		Expect(err).ToNot(HaveOccurred())

		prefillHostPort := testInfo.prefillBackend.URL[len("http://"):]
		req.Header.Add(routing.PrefillEndpointHeader, prefillHostPort)

		resp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())
		if resp.StatusCode != 200 {
			bp, _ := io.ReadAll(resp.Body)
			Fail(string(bp))
		}

		// Wait for async prefill request to be fully recorded
		Eventually(func() int {
			return len(testInfo.prefillHandler.GetCompletionRequests())
		}).Should(Equal(1))

		// header includes rank_id
		prefillHeaders := testInfo.prefillHandler.GetCompletionHeaders()
		Expect(prefillHeaders).To(HaveLen(1))
		pinnedRank := prefillHeaders[0].Get(requestHeaderDataParallelRank)
		Expect(engineByRank).To(HaveKey(pinnedRank))

		// decode payload body has rank_id's engine_id
		decodeReqs := testInfo.decodeHandler.GetCompletionRequests()
		Expect(decodeReqs).To(HaveLen(1))
		decodeKVParams, ok := decodeReqs[0][reqcommon.FieldKVTransferParams].(map[string]any)
		Expect(ok).To(BeTrue())
		Expect(decodeKVParams[reqcommon.FieldRemoteEngineID]).To(Equal(engineByRank[pinnedRank]))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})
})

// mooncakeEngineMapTest wires a Mooncake sidecar to a prefill backend, a decode
// backend reached through the production decoder proxy, and a bootstrap server
// whose engine_id changes on its second query, the observable effect of a
// prefill container restarting in place.
type mooncakeEngineMapTest struct {
	proxy       *Server
	prefillHost string

	bootstrapQueries atomic.Int32
	// prefillStatusCode and decodeStatusCode are returned by every request to
	// that stage once set; 0 means 200.
	prefillStatusCode atomic.Int32
	decodeStatusCode  atomic.Int32
	// decodeHold keeps the decode backend in flight until the prefill leg
	// cancels the decode request, so a failed prefill is relayed to the client
	// by the decoder proxy instead of racing a 200 from the decode backend.
	decodeHold atomic.Bool

	decodeMu      sync.Mutex
	decodeEngines []string // remote_engine_id carried by each decode request, in order
}

func newMooncakeEngineMapTest(t *testing.T) *mooncakeEngineMapTest {
	t.Helper()
	fx := &mooncakeEngineMapTest{}

	bootstrap := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		engineID := "engine-v1"
		if fx.bootstrapQueries.Add(1) > 1 {
			engineID = "engine-v2"
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"0": {"engine_id": %q}}`, engineID)
	}))
	t.Cleanup(bootstrap.Close)
	bootstrapURL, err := url.Parse(bootstrap.URL)
	require.NoError(t, err)
	bootstrapPort, err := strconv.Atoi(bootstrapURL.Port())
	require.NoError(t, err)

	prefill := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.WriteHeader(fx.statusOrDefault(&fx.prefillStatusCode))
	}))
	t.Cleanup(prefill.Close)
	prefillURL, err := url.Parse(prefill.URL)
	require.NoError(t, err)
	fx.prefillHost = prefillURL.Host

	decode := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			KVTransferParams struct {
				RemoteEngineID string `json:"remote_engine_id"`
			} `json:"kv_transfer_params"`
		}
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &body)
		fx.decodeMu.Lock()
		fx.decodeEngines = append(fx.decodeEngines, body.KVTransferParams.RemoteEngineID)
		fx.decodeMu.Unlock()

		if fx.decodeHold.Load() {
			// The prefill failure cancels this request; the decoder proxy
			// relays the prefill status to the client.
			<-r.Context().Done()
			return
		}
		w.WriteHeader(fx.statusOrDefault(&fx.decodeStatusCode))
	}))
	t.Cleanup(decode.Close)
	decodeURL, err := url.Parse(decode.URL)
	require.NoError(t, err)

	fx.proxy = NewProxy(Config{Port: "0", DecoderURL: decodeURL, KVConnector: constants.KVConnectorMooncake, MooncakeBootstrapPort: bootstrapPort})
	fx.proxy.decoderProxy = fx.proxy.createDecoderProxyHandler(decodeURL, false)

	return fx
}

// send runs one chat completions request through the Mooncake handler and
// returns the status the client receives.
func (fx *mooncakeEngineMapTest) send(t *testing.T) int {
	t.Helper()
	r := withContextLogger(fx.proxy, httptest.NewRequest(http.MethodPost, reqcommon.PathChatCompletions, strings.NewReader(chatCompletionsRequestBody)))
	rec := httptest.NewRecorder()
	fx.proxy.handleMooncake(rec, r, fx.prefillHost, reqcommon.APITypeChatCompletions)
	return rec.Code
}

// decodeEngineIDs returns the remote_engine_id named by each decode request so far.
func (fx *mooncakeEngineMapTest) decodeEngineIDs() []string {
	fx.decodeMu.Lock()
	defer fx.decodeMu.Unlock()
	return append([]string(nil), fx.decodeEngines...)
}

// statusOrDefault returns the configured status of a stage, or 200 when unset.
func (fx *mooncakeEngineMapTest) statusOrDefault(stage *atomic.Int32) int {
	if code := int(stage.Load()); code != 0 {
		return code
	}
	return http.StatusOK
}

// A failed request must drop the cached engine map: vLLM mints a new engine_id
// at every start, so the next request has to re-query the bootstrap server and
// name the current engine.
func TestMooncakeEngineMapDroppedAfterFailedRequest(t *testing.T) {
	fx := newMooncakeEngineMapTest(t)

	require.Equal(t, http.StatusOK, fx.send(t))

	fx.decodeStatusCode.Store(http.StatusBadGateway)
	require.Equal(t, http.StatusBadGateway, fx.send(t))
	fx.decodeStatusCode.Store(http.StatusOK)

	require.Equal(t, http.StatusOK, fx.send(t))
	require.Equal(t, []string{"engine-v1", "engine-v1", "engine-v2"}, fx.decodeEngineIDs())
	require.Equal(t, int32(2), fx.bootstrapQueries.Load(), "the request after the failure must re-query the bootstrap server")
}

// A 4xx the client observes must keep the cached engine map: the prefill is
// still the engine it was, and dropping the map on a load-shedding rejection
// makes every rejected request pay a bootstrap /query on top of the overload.
func TestMooncakeEngineMapKeptAfterClientError(t *testing.T) {
	fx := newMooncakeEngineMapTest(t)

	require.Equal(t, http.StatusOK, fx.send(t))

	fx.decodeStatusCode.Store(http.StatusTooManyRequests)
	require.Equal(t, http.StatusTooManyRequests, fx.send(t))
	fx.decodeStatusCode.Store(http.StatusOK)

	require.Equal(t, http.StatusOK, fx.send(t))
	require.Equal(t, []string{"engine-v1", "engine-v1", "engine-v1"}, fx.decodeEngineIDs())
	require.Equal(t, int32(1), fx.bootstrapQueries.Load(), "a 4xx must not force a bootstrap re-query")
}

// A prefill failure the client observes also invalidates the cached engine
// map: the decoder proxy relays the prefill status onto the response writer
// handleMooncake wraps.
func TestMooncakeEngineMapDroppedAfterPrefillError(t *testing.T) {
	fx := newMooncakeEngineMapTest(t)

	require.Equal(t, http.StatusOK, fx.send(t))

	fx.prefillStatusCode.Store(http.StatusServiceUnavailable)
	fx.decodeHold.Store(true)
	require.Equal(t, http.StatusServiceUnavailable, fx.send(t))

	_, ok := fx.proxy.mooncakeEngineIDs.Get(fx.prefillHost)
	require.False(t, ok, "the observed prefill failure did not drop the cached engine map")
}
