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
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
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
	// Cached tokens of the prefill responses; the mock decode engine reports
	// another number.
	testNIXLPushCachedTokens = 7
)

const streamingChatCompletionsRequestBody = `{
				"model": "Qwen/Qwen2-0.5B",
				"messages": [
				  {"role": "user", "content": "Hello"}
				],
				"max_tokens": 50,
				"stream": true,
				"stream_options": {"include_usage": true}
			}`

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
	return nixlPushPrefillAnswerWith(testNIXLPushIdentity(engineID))
}

// nixlPushPrefillAnswerWith is a prefill response of vLLM's NixlPushConnector
// that carries identity and reports testNIXLPushCachedTokens cached tokens.
func nixlPushPrefillAnswerWith(identity nixlPushIdentity) string {
	kv := map[string]any(maps.Clone(identity))
	kv[reqcommon.FieldDoRemotePrefill] = true
	kv[reqcommon.FieldDoRemoteDecode] = false
	kv[reqcommon.FieldRemoteBlockIDs] = []int{1, 2, 3}
	kv[requestFieldRemoteRequestID] = "cmpl-prefill"
	answer, err := json.Marshal(map[string]any{
		reqcommon.FieldKVTransferParams: kv,
		reqcommon.FieldUsage: map[string]any{
			reqcommon.FieldPromptTokensDetails: map[string]any{reqcommon.FieldCachedTokens: testNIXLPushCachedTokens},
		},
	})
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

// newNIXLPushMocks returns mock engines whose prefill answers with the
// identity startNIXLPushParallelProxy caches.
func newNIXLPushMocks() (prefill, decode *mock.ChatCompletionHandler) {
	prefill = &mock.ChatCompletionHandler{Connector: constants.KVConnectorNIXLV2, Role: mock.RolePrefill,
		RawResponse: nixlPushPrefillAnswer(testNIXLPushEngineID)}
	decode = &mock.ChatCompletionHandler{Connector: constants.KVConnectorNIXLV2, Role: mock.RoleDecode}
	return prefill, decode
}

// resentDecodeBody is the response of staleThenResentDecode to a decode request
// that is sent again.
const resentDecodeBody = `{"id":"resent-decode","choices":[]}`

// staleThenResentDecode stands in for a decode engine whose first request is
// cancelled and sent again, as when it names an engine that does not run the
// prefill. It sends the kv_transfer_params of every request to kvParams. The
// first request writes the start of a response, closes arrived and blocks until
// it is cancelled, which closes cancelled, or until stop is closed. Later
// requests get resentDecodeBody.
func staleThenResentDecode(arrived, cancelled, stop chan struct{}, kvParams chan<- map[string]any) http.Handler {
	var requests atomic.Int32
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var request map[string]any
		_ = json.Unmarshal(body, &request)
		kv, _ := request[reqcommon.FieldKVTransferParams].(map[string]any)
		kvParams <- kv
		if requests.Add(1) > 1 {
			statusHandler(http.StatusOK, resentDecodeBody).ServeHTTP(w, r)
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{"id":"stale-decode",`)
		_ = http.NewResponseController(w).Flush()
		close(arrived)
		select {
		case <-r.Context().Done():
			close(cancelled)
		case <-stop:
		}
	})
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
		prefillMock.RawResponse = nixlPushPrefillAnswerWith(identity)
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
		var prefillRequests atomic.Int32
		prefill := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			prefillRequests.Add(1)
			select {
			case <-decodeArrived:
			case <-stop:
				return
			}
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"error":"prefill boom"}`))
		})
		// A status that is not retryable is returned although retries are left.
		env := startNIXLPushParallelProxy(prefill, blockUntilCancelled(decodeArrived, decodeCancelled, stop),
			func(cfg *Config) { cfg.PrefillMaxRetries = 2 })
		// Runs before the backends close, so blocked handlers return.
		DeferCleanup(func() { close(stop) })

		status, _, body, err := env.send(8 * time.Second)
		Expect(err).ToNot(HaveOccurred())
		Expect(status).To(Equal(http.StatusInternalServerError))
		Expect(body).To(ContainSubstring("prefill boom"))
		Expect(prefillRequests.Load()).To(BeEquivalentTo(1))
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

	DescribeTable("cancels decode and sends it again with the prefill response",
		func(prefillAnswer func() string, wantCached nixlPushIdentity) {
			decodeArrived, decodeCancelled, stop := make(chan struct{}), make(chan struct{}), make(chan struct{})
			decodeKV := make(chan map[string]any, 4)
			prefillMock, _ := newNIXLPushMocks()
			prefillMock.RawResponse = prefillAnswer()
			prefill := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				select {
				case <-decodeArrived:
					prefillMock.ServeHTTP(w, r)
				case <-stop:
				}
			})
			env := startNIXLPushParallelProxy(prefill, staleThenResentDecode(decodeArrived, decodeCancelled, stop, decodeKV), nil)
			DeferCleanup(func() { close(stop) })

			status, _, body, err := env.send(10 * time.Second)
			Expect(err).ToNot(HaveOccurred())
			Expect(status).To(Equal(http.StatusOK))
			Expect(body).To(Equal(resentDecodeBody))
			Eventually(decodeCancelled).Should(BeClosed())

			By("sending decode the prefill response's kv_transfer_params with the transfer_id of the dispatch")
			transferID := kvParams(prefillMock, 0)[requestFieldTransferID]
			Expect(<-decodeKV).To(HaveKeyWithValue(requestFieldTransferID, transferID))
			var answer map[string]any
			Expect(json.Unmarshal([]byte(prefillMock.RawResponse), &answer)).To(Succeed())
			want, ok := answer[reqcommon.FieldKVTransferParams].(map[string]any)
			Expect(ok).To(BeTrue())
			want[requestFieldTransferID] = transferID
			Expect(<-decodeKV).To(Equal(want))

			identity, _ := cachedIdentity(env)
			Expect(identity).To(Equal(wantCached))
		},
		Entry("when prefill answers with another NIXL push identity",
			func() string { return nixlPushPrefillAnswer("restarted-engine") },
			testNIXLPushIdentity("restarted-engine")),
		Entry("when prefill answers without a NIXL push identity",
			func() string {
				identity := testNIXLPushIdentity(testNIXLPushEngineID)
				delete(identity, requestFieldTransferMode)
				return nixlPushPrefillAnswerWith(identity)
			},
			nil),
	)

	It("cancels prefill and returns the decode error when decode fails before prefill answers", func() {
		prefillArrived, prefillCancelled, stop := make(chan struct{}), make(chan struct{}), make(chan struct{})
		decode := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			select {
			case <-prefillArrived:
			case <-stop:
				return
			}
			statusHandler(http.StatusInternalServerError, `{"error":"decode boom"}`).ServeHTTP(w, r)
		})
		env := startNIXLPushParallelProxy(blockUntilCancelled(prefillArrived, prefillCancelled, stop), decode, nil)
		DeferCleanup(func() { close(stop) })

		status, _, body, err := env.send(8 * time.Second)
		Expect(err).ToNot(HaveOccurred())
		Expect(status).To(Equal(http.StatusInternalServerError))
		Expect(body).To(ContainSubstring("decode boom"))
		Eventually(prefillCancelled).Should(BeClosed())
	})

	DescribeTable("sends the next request to a prefill endpoint",
		func(engineIDs []string, wantSerial bool) {
			answers := make(chan string, len(engineIDs)+1)
			for _, engineID := range engineIDs {
				answers <- nixlPushPrefillAnswer(engineID)
			}
			// The cached identity, so a parallel dispatch commits its decode request.
			answers <- nixlPushPrefillAnswer(engineIDs[len(engineIDs)-1])
			prefill := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				select {
				case answer := <-answers:
					statusHandler(http.StatusOK, answer).ServeHTTP(w, r)
				default:
					w.WriteHeader(http.StatusInternalServerError)
				}
			})
			_, decodeMock := newNIXLPushMocks()
			env := startNIXLPushParallelProxy(prefill, decodeMock, nil)
			var requests atomic.Int32
			env.proxy.nixlRequestIDFn = func() (string, error) {
				return fmt.Sprintf("request-%d", requests.Add(1)), nil
			}

			for range cap(answers) {
				expectSent(env)
			}

			lastRequestID := fmt.Sprintf("request-%d", cap(answers))
			var lastDecodeKV []map[string]any
			for i, header := range decodeMock.GetCompletionHeaders() {
				if header.Get(reqcommon.RequestIDHeaderKey) == lastRequestID {
					lastDecodeKV = append(lastDecodeKV, kvParams(decodeMock, i))
				}
			}
			Expect(lastDecodeKV).To(HaveLen(1))
			// Only a serial dispatch forwards the remote_block_ids of the prefill response.
			_, serial := lastDecodeKV[0][reqcommon.FieldRemoteBlockIDs]
			Expect(serial).To(Equal(wantSerial))
		},
		Entry("serially after its NIXL push identity changed twice within the window",
			[]string{"prefill-engine_dp1", "prefill-engine_dp0"}, true),
		Entry("in parallel after its NIXL push identity changed once",
			[]string{"restarted-engine", "restarted-engine"}, false),
	)

	DescribeTable("reports the cached tokens of the prefill response in the usage of the decode response",
		func(requestBody, decodeResponseType, decodeResponse string) {
			prefillMock, decodeMock := newNIXLPushMocks()
			decodeMock.RawResponseType = decodeResponseType
			decodeMock.RawResponse = decodeResponse
			prefill, decode := overlapping(prefillMock, decodeMock)
			env := startNIXLPushParallelProxy(prefill, decode, nil)

			status, _, body, err := env.sendBody(requestBody, 10*time.Second)
			Expect(err).ToNot(HaveOccurred())
			Expect(status).To(Equal(http.StatusOK), body)
			Expect(body).To(ContainSubstring(fmt.Sprintf(`"cached_tokens":%d`, testNIXLPushCachedTokens)))
		},
		Entry("in a JSON response", chatCompletionsRequestBody, "", ""),
		Entry("in a streamed response", streamingChatCompletionsRequestBody, eventStreamContentType,
			"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n"+
				"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":64,\"completion_tokens\":1,\"total_tokens\":65,\"prompt_tokens_details\":{\"cached_tokens\":49}}}\n\n"+
				"data: [DONE]\n\n"),
	)

	It("sends a request with chunked decode in parallel and the NIXL push kv_transfer_params with its first decode chunk only", func() {
		chunks := []string{chatResponse("hello ", "length", 8, 5), chatResponse("world", "stop", 9, 5)}
		decodeKV := make(chan any, len(chunks)+1)
		var decodeRequests atomic.Int32
		decodeChunks := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var request map[string]any
			_ = json.NewDecoder(r.Body).Decode(&request)
			decodeKV <- request[reqcommon.FieldKVTransferParams]
			chunk := int(decodeRequests.Add(1)) - 1
			if chunk >= len(chunks) {
				w.WriteHeader(http.StatusInternalServerError)
				return
			}
			statusHandler(http.StatusOK, chunks[chunk]).ServeHTTP(w, r)
		})
		prefillMock, _ := newNIXLPushMocks()
		prefill, decode := overlapping(prefillMock, decodeChunks)
		env := startNIXLPushParallelProxy(prefill, decode, func(cfg *Config) { cfg.DecodeChunkSize = 5 })
		env.proxy.nixlRequestIDFn = func() (string, error) { return testNIXLPushRequestID, nil }

		status, _, body, err := env.send(10 * time.Second)
		Expect(err).ToNot(HaveOccurred())
		Expect(status).To(Equal(http.StatusOK), body)
		var response map[string]any
		Expect(json.Unmarshal([]byte(body), &response)).To(Succeed())
		Expect(extractChoiceText(firstChoice(response))).To(Equal("hello world"))

		By("sending the NIXL push kv_transfer_params with the first decode chunk only")
		want := map[string]any(testNIXLPushIdentity(testNIXLPushEngineID))
		want[reqcommon.FieldDoRemotePrefill] = true
		want[reqcommon.FieldDoRemoteDecode] = false
		want[requestFieldRemoteRequestID] = testNIXLPushRequestID
		want[requestFieldTransferID] = kvParams(prefillMock, 0)[requestFieldTransferID]
		Expect(<-decodeKV).To(Equal(want))
		Expect(<-decodeKV).To(BeNil())
	})

	DescribeTable("retries prefill on the serial path after it answers a retryable status",
		func(prefillStatuses []int, wantAttempts, wantStatus int, wantBody string) {
			decodeArrived, decodeCancelled, stop := make(chan struct{}), make(chan struct{}), make(chan struct{})
			var mu sync.Mutex
			var transferIDs []any
			prefill := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var request map[string]any
				_ = json.NewDecoder(r.Body).Decode(&request)
				kv, _ := request[reqcommon.FieldKVTransferParams].(map[string]any)
				mu.Lock()
				transferIDs = append(transferIDs, kv[requestFieldTransferID])
				attempt := len(transferIDs)
				mu.Unlock()
				if attempt == 1 {
					// Fails the parallel attempt once its decode request arrived, so
					// that request is seen cancelled.
					select {
					case <-decodeArrived:
					case <-stop:
						return
					}
				}
				if attempt > len(prefillStatuses) {
					statusHandler(http.StatusOK, nixlPushPrefillAnswer(testNIXLPushEngineID)).ServeHTTP(w, r)
					return
				}
				statusHandler(prefillStatuses[attempt-1], fmt.Sprintf(`{"error":"prefill attempt %d"}`, attempt)).ServeHTTP(w, r)
			})
			decode := staleThenResentDecode(decodeArrived, decodeCancelled, stop, make(chan map[string]any, 4))
			env := startNIXLPushParallelProxy(prefill, decode, func(cfg *Config) {
				cfg.PrefillMaxRetries = 2
				cfg.PrefillRetryBackoff = time.Millisecond
			})
			DeferCleanup(func() { close(stop) })

			status, _, body, err := env.send(10 * time.Second)
			Expect(err).ToNot(HaveOccurred())
			Expect(status).To(Equal(wantStatus))
			Expect(body).To(Equal(wantBody))
			Eventually(decodeCancelled).Should(BeClosed())

			By("giving every prefill attempt its own transfer_id")
			mu.Lock()
			defer mu.Unlock()
			Expect(transferIDs).To(HaveLen(wantAttempts))
			seen := map[any]bool{}
			for _, transferID := range transferIDs {
				Expect(transferID).To(HavePrefix("xfer-"))
				seen[transferID] = true
			}
			Expect(seen).To(HaveLen(wantAttempts))

			By("keeping the cached identity")
			identity, cached := cachedIdentity(env)
			Expect(cached).To(BeTrue())
			Expect(identity).To(Equal(testNIXLPushIdentity(testNIXLPushEngineID)))
		},
		Entry("and returns the response of the attempt that succeeds",
			[]int{http.StatusServiceUnavailable}, 2, http.StatusOK, resentDecodeBody),
		Entry("no more often than --prefill-max-retries allows",
			[]int{http.StatusServiceUnavailable, http.StatusBadGateway, http.StatusGatewayTimeout}, 3,
			http.StatusGatewayTimeout, `{"error":"prefill attempt 3"}`),
		Entry("until prefill answers a status that is not retryable",
			[]int{http.StatusServiceUnavailable, http.StatusInternalServerError}, 2,
			http.StatusInternalServerError, `{"error":"prefill attempt 2"}`),
	)

	It("retries prefill with the client's messages after chunked decode of the cancelled attempt produced a chunk", func() {
		secondChunkArrived, stop := make(chan struct{}), make(chan struct{})
		var decodeRequests atomic.Int32
		decode := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = io.ReadAll(r.Body)
			switch decodeRequests.Add(1) {
			case 1:
				statusHandler(http.StatusOK, chatResponse("hello ", "length", 8, 5)).ServeHTTP(w, r)
			case 2:
				close(secondChunkArrived)
				select {
				case <-r.Context().Done():
				case <-stop:
				}
			default:
				statusHandler(http.StatusOK, chatResponse("hi", "stop", 8, 1)).ServeHTTP(w, r)
			}
		})
		prefillMessages := make(chan string, 4)
		var prefillRequests atomic.Int32
		prefill := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var request map[string]json.RawMessage
			_ = json.NewDecoder(r.Body).Decode(&request)
			prefillMessages <- string(request[reqcommon.FieldMessages])
			if prefillRequests.Add(1) > 1 {
				statusHandler(http.StatusOK, nixlPushPrefillAnswer(testNIXLPushEngineID)).ServeHTTP(w, r)
				return
			}
			// Chunked decode has added the first chunk to its request by now.
			select {
			case <-secondChunkArrived:
			case <-stop:
				return
			}
			statusHandler(http.StatusServiceUnavailable, `{"error":"prefill overloaded"}`).ServeHTTP(w, r)
		})
		env := startNIXLPushParallelProxy(prefill, decode, func(cfg *Config) {
			cfg.DecodeChunkSize = 5
			cfg.PrefillMaxRetries = 1
			cfg.PrefillRetryBackoff = time.Millisecond
		})
		DeferCleanup(func() { close(stop) })

		status, _, body, err := env.send(10 * time.Second)
		Expect(err).ToNot(HaveOccurred())
		Expect(status).To(Equal(http.StatusOK), body)
		parallelAttempt, retry := <-prefillMessages, <-prefillMessages
		Expect(retry).To(Equal(parallelAttempt))
	})
})
