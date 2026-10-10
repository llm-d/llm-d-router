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
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"time"

	. "github.com/onsi/ginkgo/v2" // nolint:revive
	. "github.com/onsi/gomega"    // nolint:revive

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
	"github.com/llm-d/llm-d-router/test/sidecar/mock"
)

const (
	testNIXLPushEngineID  = "prefill-engine"
	testNIXLPushRequestID = "00000000-0000-0000-0000-000000000003"
)

// startNIXLPushParallelProxy runs startParallelCommitProxy in NIXL push mode
// with testNIXLPushIdentity cached for the prefill endpoint, so every request
// takes the parallel dispatch.
func startNIXLPushParallelProxy(prefill, decode http.Handler, mutate func(cfg *Config)) *parallelCommitEnv {
	env := startParallelCommitProxy(prefill, decode, func(cfg *Config) {
		cfg.MoRIIOWriteMode = false
		cfg.MoRIIOParallelDispatch = false
		cfg.NIXLPushMode = true
		if mutate != nil {
			mutate(cfg)
		}
	})
	env.proxy.nixlPushIdentities.put(env.prefillHost, testNIXLPushIdentity(testNIXLPushEngineID))
	return env
}

// nixlPushPrefillAnswer is a prefill response of vLLM's NixlPushConnector that
// carries testNIXLPushIdentity(engineID).
func nixlPushPrefillAnswer(engineID string) string {
	kv := map[string]any(testNIXLPushIdentity(engineID))
	kv[reqcommon.FieldDoRemotePrefill] = true
	kv[reqcommon.FieldDoRemoteDecode] = false
	kv[reqcommon.FieldRemoteBlockIDs] = []int{1, 2, 3}
	kv[requestFieldRemoteRequestID] = "cmpl-prefill"
	answer, err := json.Marshal(map[string]any{reqcommon.FieldKVTransferParams: kv})
	Expect(err).ToNot(HaveOccurred())
	return string(answer)
}

// overlapping lets prefill answer only once the decode request has arrived. A
// serial dispatch sends decode after prefill answered, so there the wrapped
// prefill answers 500.
func overlapping(prefill, decode http.Handler) (http.Handler, http.Handler) {
	decodeArrived := make(chan struct{})
	var once sync.Once
	wrappedPrefill := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-decodeArrived:
			prefill.ServeHTTP(w, r)
		case <-time.After(5 * time.Second):
			w.WriteHeader(http.StatusInternalServerError)
		}
	})
	wrappedDecode := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		once.Do(func() { close(decodeArrived) })
		decode.ServeHTTP(w, r)
	})
	return wrappedPrefill, wrappedDecode
}

// blockUntilCancelled closes arrived once it has read the request body, then
// blocks until the request is cancelled, which closes cancelled, or until stop
// is closed. The server notices a cancelled request only after its body is read.
func blockUntilCancelled(arrived, cancelled, stop chan struct{}) http.Handler {
	return http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		close(arrived)
		select {
		case <-r.Context().Done():
			close(cancelled)
		case <-stop:
		}
	})
}

func newNIXLPushMocks() (prefill, decode *mock.ChatCompletionHandler) {
	return &mock.ChatCompletionHandler{Connector: constants.KVConnectorNIXLV2, Role: mock.RolePrefill},
		&mock.ChatCompletionHandler{Connector: constants.KVConnectorNIXLV2, Role: mock.RoleDecode}
}

var _ = Describe("NIXL Connector (v2) NIXL push parallel dispatch", func() {

	expectSent := func(env *parallelCommitEnv) {
		GinkgoHelper()
		status, _, body, err := env.send(10 * time.Second)
		Expect(err).ToNot(HaveOccurred())
		Expect(status).To(Equal(http.StatusOK), body)
	}

	cachedIdentity := func(env *parallelCommitEnv) (nixlPushIdentity, bool) {
		return env.proxy.nixlPushIdentities.get(env.prefillHost)
	}

	// The identity is dropped after decode returns, which can be after the
	// client has read the response.
	expectDropped := func(env *parallelCommitEnv) {
		GinkgoHelper()
		Eventually(func() bool {
			_, cached := cachedIdentity(env)
			return cached
		}).Should(BeFalse())
	}

	It("sends the prefill and decode requests at once", func() {
		prefill, decode := overlapping(
			statusHandler(http.StatusOK, nixlPushPrefillAnswer(testNIXLPushEngineID)),
			statusHandler(http.StatusOK, `{"choices":[]}`),
		)
		env := startNIXLPushParallelProxy(prefill, decode, nil)

		expectSent(env)
	})

	It("sends the prefill request the serial push kv_transfer_params and decode the same transfer_id", func() {
		prefillMock, decodeMock := newNIXLPushMocks()
		prefill, decode := overlapping(prefillMock, decodeMock)
		env := startNIXLPushParallelProxy(prefill, decode, nil)

		expectSent(env)

		transferID, ok := kvParams(prefillMock, 0)[requestFieldTransferID].(string)
		Expect(ok).To(BeTrue())
		Expect(transferID).To(MatchRegexp(`^xfer-[0-9a-f-]{36}$`))
		Expect(kvParams(prefillMock, 0)).To(Equal(map[string]any{
			reqcommon.FieldDoRemoteDecode:  true,
			reqcommon.FieldDoRemotePrefill: false,
			reqcommon.FieldRemoteEngineID:  nil,
			reqcommon.FieldRemoteBlockIDs:  nil,
			reqcommon.FieldRemoteHost:      nil,
			reqcommon.FieldRemotePort:      nil,
			requestFieldTransferID:         transferID,
		}))
		Expect(kvParams(decodeMock, 0)).To(HaveKeyWithValue(requestFieldTransferID, transferID))
	})

	// vLLM reads remote_engine_id, remote_host, remote_port, tp_size and
	// remote_request_id without a default, so a missing one fails the decode
	// engine. It sets remote_block_ids itself.
	It("builds the decode kv_transfer_params from exactly the cached identity and the per-request fields", func() {
		prefillMock, decodeMock := newNIXLPushMocks()
		env := startNIXLPushParallelProxy(prefillMock, decodeMock, nil)
		identity := testNIXLPushIdentity(testNIXLPushEngineID)
		identity[requestFieldPPSize] = float64(1)
		identity[requestFieldDCPSize] = float64(1)
		env.proxy.nixlPushIdentities.put(env.prefillHost, identity)
		env.proxy.nixlRequestIDFn = func() (string, error) { return testNIXLPushRequestID, nil }

		expectSent(env)

		Expect(kvParams(decodeMock, 0)).To(Equal(map[string]any{
			reqcommon.FieldRemoteEngineID:  testNIXLPushEngineID,
			reqcommon.FieldRemoteHost:      testPrefillHostIP1,
			reqcommon.FieldRemotePort:      float64(5600),
			requestFieldTPSize:             float64(2),
			requestFieldPPSize:             float64(1),
			requestFieldDCPSize:            float64(1),
			requestFieldTransferMode:       nixlTransferModePush,
			reqcommon.FieldDoRemotePrefill: true,
			reqcommon.FieldDoRemoteDecode:  false,
			requestFieldRemoteRequestID:    testNIXLPushRequestID,
			requestFieldTransferID:         kvParams(prefillMock, 0)[requestFieldTransferID],
		}))

		By("sending both requests the same id as x-request-id")
		Expect(prefillMock.GetCompletionHeaders()[0].Get(reqcommon.RequestIDHeaderKey)).To(Equal(testNIXLPushRequestID))
		Expect(decodeMock.GetCompletionHeaders()[0].Get(reqcommon.RequestIDHeaderKey)).To(Equal(testNIXLPushRequestID))
	})

	It("returns the prefill error, cancels decode and drops the cached identity when prefill fails", func() {
		decodeArrived, decodeCancelled, stop := make(chan struct{}), make(chan struct{}), make(chan struct{})
		prefill := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			select {
			case <-decodeArrived:
			case <-stop:
				return
			}
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"error":"prefill boom"}`))
		})
		env := startNIXLPushParallelProxy(prefill, blockUntilCancelled(decodeArrived, decodeCancelled, stop), nil)
		// Runs before the backends close, so blocked handlers return.
		DeferCleanup(func() { close(stop) })

		status, _, body, err := env.send(8 * time.Second)
		Expect(err).ToNot(HaveOccurred())
		Expect(status).To(Equal(http.StatusInternalServerError))
		Expect(body).To(ContainSubstring("prefill boom"))
		Eventually(decodeCancelled).Should(BeClosed())
		expectDropped(env)
	})

	It("answers 504, cancels both requests and drops the cached identity when prefill times out", func() {
		prefillCancelled, decodeCancelled, stop := make(chan struct{}), make(chan struct{}), make(chan struct{})
		env := startNIXLPushParallelProxy(
			blockUntilCancelled(make(chan struct{}), prefillCancelled, stop),
			blockUntilCancelled(make(chan struct{}), decodeCancelled, stop),
			func(cfg *Config) { cfg.NIXLPushPrefillTimeout = 500 * time.Millisecond },
		)
		DeferCleanup(func() { close(stop) })

		status, _, body, err := env.send(8 * time.Second)
		Expect(err).ToNot(HaveOccurred())
		Expect(status).To(Equal(http.StatusGatewayTimeout))
		Expect(body).To(ContainSubstring(errNIXLPushPrefillTimeout.Error()))
		Eventually(prefillCancelled).Should(BeClosed())
		Eventually(decodeCancelled).Should(BeClosed())
		expectDropped(env)
	})

	It("drops the cached identity when decode answers an error status", func() {
		prefill, decode := overlapping(
			statusHandler(http.StatusOK, nixlPushPrefillAnswer(testNIXLPushEngineID)),
			statusHandler(http.StatusInternalServerError, `{"error":"decode boom"}`),
		)
		env := startNIXLPushParallelProxy(prefill, decode, nil)

		status, _, body, err := env.send(10 * time.Second)
		Expect(err).ToNot(HaveOccurred())
		Expect(status).To(Equal(http.StatusInternalServerError))
		Expect(body).To(ContainSubstring("decode boom"))
		expectDropped(env)
	})

	It("keeps the cached identity when the client goes away", func() {
		prefillArrived, decodeArrived, stop := make(chan struct{}), make(chan struct{}), make(chan struct{})
		env := startNIXLPushParallelProxy(
			blockUntilCancelled(prefillArrived, make(chan struct{}), stop),
			blockUntilCancelled(decodeArrived, make(chan struct{}), stop),
			nil,
		)
		DeferCleanup(func() { close(stop) })

		// Called directly, not through the listener, so the cache is checked
		// only after the handler returned.
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		req := httptest.NewRequestWithContext(ctx, http.MethodPost, reqcommon.PathChatCompletions,
			strings.NewReader(chatCompletionsRequestBody))
		handled := make(chan struct{})
		go func() {
			defer GinkgoRecover()
			defer close(handled)
			env.proxy.handleNIXLV2(httptest.NewRecorder(), req, env.prefillHost, "", reqcommon.APITypeChatCompletions)
		}()
		Eventually(prefillArrived).Should(BeClosed())
		Eventually(decodeArrived).Should(BeClosed())
		cancel()
		Eventually(handled).Should(BeClosed())

		_, cached := cachedIdentity(env)
		Expect(cached).To(BeTrue())
	})

	It("refreshes the cached identity from the prefill response", func() {
		prefill, decode := overlapping(
			statusHandler(http.StatusOK, nixlPushPrefillAnswer("restarted-engine")),
			statusHandler(http.StatusOK, `{"choices":[]}`),
		)
		env := startNIXLPushParallelProxy(prefill, decode, nil)

		expectSent(env)

		identity, cached := cachedIdentity(env)
		Expect(cached).To(BeTrue())
		Expect(identity).To(Equal(testNIXLPushIdentity("restarted-engine")))
	})

	It("keeps the cached identity when a successful prefill response does not parse", func() {
		prefill, decode := overlapping(
			statusHandler(http.StatusOK, "not json"),
			statusHandler(http.StatusOK, `{"choices":[]}`),
		)
		env := startNIXLPushParallelProxy(prefill, decode, nil)

		expectSent(env)

		identity, cached := cachedIdentity(env)
		Expect(cached).To(BeTrue())
		Expect(identity).To(Equal(testNIXLPushIdentity(testNIXLPushEngineID)))
	})
})
