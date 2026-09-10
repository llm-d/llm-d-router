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
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"

	. "github.com/onsi/ginkgo/v2" // nolint:revive
	. "github.com/onsi/gomega"    // nolint:revive

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/test/sidecar/mock"
)

const chatCompletionsRequestBody = `{
				"model": "Qwen/Qwen2-0.5B",
				"messages": [
				  {"role": "user", "content": "Hello"}
				],
				"max_tokens": 50
			}`

const chatCompletionsRequestBodyWithMaxCompletionTokens = `{
				"model": "Qwen/Qwen2-0.5B",
				"messages": [
				  {"role": "user", "content": "Hello"}
				],
				"max_tokens": 50,
				"max_completion_tokens": 100
			}`

const chatCompletionsRequestBodyWithMinTokens = `{
				"model": "Qwen/Qwen2-0.5B",
				"messages": [
				  {"role": "user", "content": "Hello"}
				],
				"max_tokens": 50,
				"min_tokens": 5
			}`

const generateRequestBodyWithTokenLimits = `{
				"model": "Qwen/Qwen2-0.5B",
				"token_ids": [1, 2, 3, 4],
				"sampling_params": {"max_tokens": 100, "min_tokens": 5}
			}`

// expectGenerateLegTokenLimits starts the proxy, posts a generate request
// carrying client token limits, and asserts the generate-API token-limit
// contract on the two legs the connector produces: the prefill leg is capped to
// a single output token inside sampling_params, and the decode leg still carries
// the client's own limits. The connectors copy the client body one level deep,
// so the two legs share that nested map; a cap written through it instead of
// replacing it takes the client's limits with it.
func expectGenerateLegTokenLimits(testInfo *sidecarTestInfo) {
	GinkgoHelper()

	proxyBaseAddr := testInfo.startProxy()

	req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+GeneratePath,
		bytes.NewReader([]byte(generateRequestBodyWithTokenLimits)))
	Expect(err).ToNot(HaveOccurred())
	req.Header.Add(routing.PrefillEndpointHeader, testInfo.prefillBackend.URL[len("http://"):])

	resp, err := http.DefaultClient.Do(req)
	Expect(err).ToNot(HaveOccurred())
	defer resp.Body.Close() //nolint:errcheck

	if resp.StatusCode != 200 {
		bp, _ := io.ReadAll(resp.Body) //nolint:errcheck
		Fail(string(bp))
	}

	expectGenerateLegTokenLimitsOn(testInfo.prefillHandler, testInfo.decodeHandler)
}

// expectGenerateLegTokenLimitsOn asserts that contract on the legs captured by
// the two mock backends, for a client body carrying max_tokens 100 and
// min_tokens 5 under sampling_params. The top-level assertions pin the caps to
// the API the client spoke: a leg builder that names the chat fields regardless
// of the API type writes max_tokens beside sampling_params instead of inside it,
// where the engine never reads it.
func expectGenerateLegTokenLimitsOn(prefillHandler, decodeHandler *mock.ChatCompletionHandler) {
	GinkgoHelper()

	// Eventually covers the connectors that dispatch prefill asynchronously and
	// is already satisfied for the synchronous ones.
	Eventually(func() int { return len(prefillHandler.GetCompletionRequests()) }).Should(Equal(1))
	Eventually(func() int { return len(decodeHandler.GetCompletionRequests()) }).Should(Equal(1))

	prefillReq := prefillHandler.GetCompletionRequests()[0]
	decodeReq := decodeHandler.GetCompletionRequests()[0]

	prefillSP, ok := prefillReq[requestFieldSamplingParams].(map[string]any)
	Expect(ok).To(BeTrue())
	Expect(prefillSP).To(HaveKeyWithValue(requestFieldMaxTokens, BeNumerically("==", 1)))
	Expect(prefillSP).ToNot(HaveKey(requestFieldMinTokens))
	Expect(prefillReq).ToNot(HaveKey(requestFieldMaxTokens))
	Expect(prefillReq).ToNot(HaveKey(requestFieldMaxCompletionTokens))

	decodeSP, ok := decodeReq[requestFieldSamplingParams].(map[string]any)
	Expect(ok).To(BeTrue())
	Expect(decodeSP).To(HaveKeyWithValue(requestFieldMaxTokens, BeNumerically("==", 100)))
	Expect(decodeSP).To(HaveKeyWithValue(requestFieldMinTokens, BeNumerically("==", 5)))
	Expect(decodeReq).ToNot(HaveKey(requestFieldMaxCompletionTokens))
}

type sidecarTestInfo struct {
	ctx            context.Context
	cancelFn       context.CancelFunc
	stoppedCh      chan struct{}
	decodeBackend  *httptest.Server
	decodeHandler  *mock.ChatCompletionHandler
	prefillBackend *httptest.Server
	prefillHandler *mock.ChatCompletionHandler
	decodeURL      *url.URL
	proxy          *Server
}

// startProxy launches the proxy in a goroutine, waits for it to be ready, and
// returns its base address. Pair with testInfo.cancelFn() / <-testInfo.stoppedCh
// for teardown.
func (testInfo *sidecarTestInfo) startProxy() string {
	go func() {
		defer GinkgoRecover()

		testInfo.proxy.allowlistValidator = &AllowlistValidator{enabled: false}
		err := testInfo.proxy.Start(testInfo.ctx)
		Expect(err).ToNot(HaveOccurred())

		testInfo.stoppedCh <- struct{}{}
	}()

	<-testInfo.proxy.readyCh
	return "http://" + testInfo.proxy.addr.String()
}

// SGLang and Mooncake excluded: async prefill requires Eventually and bootstrap server setup.
var connectors = []string{KVConnectorSharedStorage, KVConnectorNIXLV2}

var _ = Describe("Common Connector tests", func() {

	for _, connector := range connectors {
		When(fmt.Sprintf("running with the %s connector", connector), func() {
			// Regression test for commit bb181d6: Ensure that max_completion_tokens=1 in Prefill
			It("should set max_completion_tokens=1 in prefill and restore original value in decode", func() {
				testInfo := sidecarConnectionTestSetup(connector)

				By("starting the proxy")
				go func() {
					defer GinkgoRecover()

					testInfo.proxy.allowlistValidator = &AllowlistValidator{enabled: false}
					err := testInfo.proxy.Start(testInfo.ctx)
					Expect(err).ToNot(HaveOccurred())

					testInfo.stoppedCh <- struct{}{}
				}()

				<-testInfo.proxy.readyCh
				proxyBaseAddr := "http://" + testInfo.proxy.addr.String()

				By("sending a /v1/chat/completions request with max_completion_tokens set")
				body := chatCompletionsRequestBodyWithMaxCompletionTokens

				req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath, bytes.NewReader([]byte(body)))
				Expect(err).ToNot(HaveOccurred())
				req.Header.Add(routing.PrefillEndpointHeader, testInfo.prefillBackend.URL[len("http://"):])

				rp, err := http.DefaultClient.Do(req)
				Expect(err).ToNot(HaveOccurred())

				if rp.StatusCode != 200 {
					bp, _ := io.ReadAll(rp.Body) //nolint:errcheck
					Fail(string(bp))
				}

				By("verifying prefill request has max_completion_tokens=1")
				Expect(testInfo.prefillHandler.RequestCount.Load()).To(BeNumerically("==", 1))
				Expect(testInfo.prefillHandler.CompletionRequests).To(HaveLen(1))
				prefillReq := testInfo.prefillHandler.CompletionRequests[0]

				Expect(prefillReq).To(HaveKeyWithValue("max_tokens", BeNumerically("==", 1)))
				Expect(prefillReq).To(HaveKeyWithValue("max_completion_tokens", BeNumerically("==", 1)))

				By("verifying decode request has original max_completion_tokens=100")
				Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))
				Expect(testInfo.decodeHandler.CompletionRequests).To(HaveLen(1))
				decodeReq := testInfo.decodeHandler.CompletionRequests[0]

				// The decode request should have the original max_completion_tokens value
				Expect(decodeReq).To(HaveKeyWithValue("max_completion_tokens", BeNumerically("==", 100)))

				testInfo.cancelFn()
				<-testInfo.stoppedCh
			})

			// Regression test for commit bb181d6: Ensure max_completion_tokens is handled when not provided
			It("should set max_completion_tokens=1 in prefill when not provided in original request", func() {
				testInfo := sidecarConnectionTestSetup(connector)

				By("starting the proxy")
				go func() {
					defer GinkgoRecover()

					testInfo.proxy.allowlistValidator = &AllowlistValidator{enabled: false}
					err := testInfo.proxy.Start(testInfo.ctx)
					Expect(err).ToNot(HaveOccurred())

					testInfo.stoppedCh <- struct{}{}
				}()

				<-testInfo.proxy.readyCh
				proxyBaseAddr := "http://" + testInfo.proxy.addr.String()

				By("sending a /v1/chat/completions request without max_completion_tokens")
				//nolint:goconst
				body := `{
				    "model": "Qwen/Qwen2-0.5B",
				    "messages": [
				      {"role": "user", "content": "Hello"}
				    ],
				    "max_tokens": 50
			    }`

				req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath, bytes.NewReader([]byte(body)))
				Expect(err).ToNot(HaveOccurred())
				req.Header.Add(routing.PrefillEndpointHeader, testInfo.prefillBackend.URL[len("http://"):])

				rp, err := http.DefaultClient.Do(req)
				Expect(err).ToNot(HaveOccurred())

				if rp.StatusCode != 200 {
					bp, _ := io.ReadAll(rp.Body) //nolint:errcheck
					Fail(string(bp))
				}

				By("verifying prefill request has max_completion_tokens=1")
				Expect(testInfo.prefillHandler.RequestCount.Load()).To(BeNumerically("==", 1))
				Expect(testInfo.prefillHandler.CompletionRequests).To(HaveLen(1))
				prefillReq := testInfo.prefillHandler.CompletionRequests[0]

				Expect(prefillReq).To(HaveKeyWithValue("max_tokens", BeNumerically("==", 1)))
				Expect(prefillReq).To(HaveKeyWithValue("max_completion_tokens", BeNumerically("==", 1)))

				By("verifying decode request does not have max_completion_tokens since it wasn't in original request")
				Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))
				Expect(testInfo.decodeHandler.CompletionRequests).To(HaveLen(1))
				decodeReq := testInfo.decodeHandler.CompletionRequests[0]

				// The decode request should not have max_completion_tokens if it wasn't in the original request
				Expect(decodeReq).ToNot(HaveKey("max_completion_tokens"))

				testInfo.cancelFn()
				<-testInfo.stoppedCh
			})

			// Regression test: a client min_tokens above the prefill leg's
			// max_tokens=1 cap trips vLLM's min_tokens<=max_tokens validation.
			It("should cap min_tokens in prefill and restore original value in decode", func() {
				testInfo := sidecarConnectionTestSetup(connector)

				By("starting the proxy")
				go func() {
					defer GinkgoRecover()

					testInfo.proxy.allowlistValidator = &AllowlistValidator{enabled: false}
					err := testInfo.proxy.Start(testInfo.ctx)
					Expect(err).ToNot(HaveOccurred())

					testInfo.stoppedCh <- struct{}{}
				}()

				<-testInfo.proxy.readyCh
				proxyBaseAddr := "http://" + testInfo.proxy.addr.String()

				By("sending a /v1/chat/completions request with min_tokens set")
				body := chatCompletionsRequestBodyWithMinTokens

				req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath, bytes.NewReader([]byte(body)))
				Expect(err).ToNot(HaveOccurred())
				req.Header.Add(routing.PrefillEndpointHeader, testInfo.prefillBackend.URL[len("http://"):])

				rp, err := http.DefaultClient.Do(req)
				Expect(err).ToNot(HaveOccurred())

				if rp.StatusCode != 200 {
					bp, _ := io.ReadAll(rp.Body) //nolint:errcheck
					Fail(string(bp))
				}

				By("verifying prefill request's min_tokens does not exceed max_tokens=1")
				Expect(testInfo.prefillHandler.RequestCount.Load()).To(BeNumerically("==", 1))
				Expect(testInfo.prefillHandler.CompletionRequests).To(HaveLen(1))
				prefillReq := testInfo.prefillHandler.CompletionRequests[0]

				Expect(prefillReq).To(HaveKeyWithValue("max_tokens", BeNumerically("==", 1)))
				// Stripped (shared-storage) or capped to 1 (NIXLv2): either way it
				// must not exceed the prefill leg's max_tokens=1.
				if minTokens, ok := prefillReq["min_tokens"]; ok {
					Expect(minTokens).To(BeNumerically("<=", 1))
				}

				By("verifying decode request keeps the client's original min_tokens=5")
				Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))
				Expect(testInfo.decodeHandler.CompletionRequests).To(HaveLen(1))
				decodeReq := testInfo.decodeHandler.CompletionRequests[0]

				Expect(decodeReq).To(HaveKeyWithValue("min_tokens", BeNumerically("==", 5)))

				testInfo.cancelFn()
				<-testInfo.stoppedCh
			})

			It("should cap sampling_params in prefill and restore originals in decode", func() {
				testInfo := sidecarConnectionTestSetup(connector)
				expectGenerateLegTokenLimits(testInfo)

				testInfo.cancelFn()
				<-testInfo.stoppedCh
			})
		})
	}
})

var _ = Describe("IPv6 endpoint address construction", func() {
	DescribeTable("mooncake bootstrapAddr brackets IPv6 host",
		func(prefillHostPort string, port int, want string) {
			got := "http://" + net.JoinHostPort(extractHost(prefillHostPort), strconv.Itoa(port))
			Expect(got).To(Equal(want))
		},
		Entry("IPv4", "10.0.0.1:8080", 9090, "http://10.0.0.1:9090"),
		Entry("IPv6", "[fd00::1]:8080", 9090, "http://[fd00::1]:9090"),
	)

	DescribeTable("nixlv2 remoteEngineID brackets IPv6 host",
		func(prefillPodHostPort string, handshakePort int, want string) {
			host, _, err := net.SplitHostPort(prefillPodHostPort)
			if err != nil {
				host = prefillPodHostPort
			}
			got := net.JoinHostPort(host, strconv.Itoa(handshakePort))
			Expect(got).To(Equal(want))
		},
		Entry("IPv4", "10.0.0.1:8080", 61000, "10.0.0.1:61000"),
		Entry("IPv6", "[fd00::2]:8080", 61000, "[fd00::2]:61000"),
	)
})

// Every connector copies the parsed request body and writes into the copy. A
// body that is not a JSON object parses to a nil map, which accepts no writes,
// so it must be refused before it reaches a connector.
var _ = Describe("Non-object request body", func() {
	DescribeTable("is refused with 400 before the connector runs",
		func(connector, body string) {
			testInfo := sidecarConnectionTestSetup(connector)
			proxyBaseAddr := testInfo.startProxy()

			req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath, bytes.NewReader([]byte(body)))
			Expect(err).ToNot(HaveOccurred())
			req.Header.Add(routing.PrefillEndpointHeader, testInfo.prefillBackend.URL[len("http://"):])

			resp, err := http.DefaultClient.Do(req)
			Expect(err).ToNot(HaveOccurred())
			defer resp.Body.Close() //nolint:errcheck

			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest))
			respBody, err := io.ReadAll(resp.Body)
			Expect(err).ToNot(HaveOccurred())
			Expect(string(respBody)).To(ContainSubstring("BadRequestError"))

			By("verifying neither leg was dispatched")
			Expect(testInfo.prefillHandler.RequestCount.Load()).To(BeZero())
			Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeZero())

			testInfo.cancelFn()
			<-testInfo.stoppedCh
		},
		Entry("nixlv2 null", KVConnectorNIXLV2, `null`),
		Entry("shared-storage null", KVConnectorSharedStorage, `null`),
		Entry("mooncake null", KVConnectorMooncake, `null`),
		Entry("p2p null", KVConnectorOffloading, `null`),
		Entry("sglang null", KVConnectorSGLang, `null`),
	)
})

// Every entry point reads the client body through readJSONBody before it does
// anything else, so a client that drops the connection mid-body must get the
// same vLLM error envelope whichever path the sidecar is configured for. The
// decoder-only paths are covered too: they write into the parsed body as well.
var _ = Describe("Unreadable request body", func() {
	DescribeTable("is refused with a vLLM error envelope",
		func(config Config, handle func(*Server, http.ResponseWriter, *http.Request)) {
			proxy := NewProxy(config)
			w := httptest.NewRecorder()
			r := httptest.NewRequest(http.MethodPost, ChatCompletionsPath, errReader{})

			handle(proxy, w, r)

			Expect(w.Code).To(Equal(http.StatusBadRequest))
			Expect(expectErrorEnvelope(w.Body.Bytes())).To(ContainSubstring("failed to read request body"))
		},
		Entry("nixlv2", Config{Port: "0", KVConnector: KVConnectorNIXLV2},
			func(s *Server, w http.ResponseWriter, r *http.Request) {
				s.handleNIXLV2(w, r, "10.0.0.1:8080", "", reqcommon.APITypeChatCompletions)
			}),
		Entry("shared-storage", Config{Port: "0", KVConnector: KVConnectorSharedStorage},
			func(s *Server, w http.ResponseWriter, r *http.Request) {
				s.handleSharedStorage(w, r, "10.0.0.1:8080", reqcommon.APITypeChatCompletions)
			}),
		Entry("mooncake", Config{Port: "0", KVConnector: KVConnectorMooncake},
			func(s *Server, w http.ResponseWriter, r *http.Request) {
				s.handleMooncake(w, r, "10.0.0.1:8080", reqcommon.APITypeChatCompletions)
			}),
		Entry("p2p", Config{Port: "0", KVConnector: KVConnectorOffloading},
			func(s *Server, w http.ResponseWriter, r *http.Request) {
				s.handleP2P(w, r, "10.0.0.1:8080", "", reqcommon.APITypeChatCompletions)
			}),
		Entry("sglang", Config{Port: "0", KVConnector: KVConnectorSGLang},
			func(s *Server, w http.ResponseWriter, r *http.Request) {
				s.handleSGLang(w, r, "10.0.0.1:8080")
			}),
		Entry("ec-nixl", Config{Port: "0", KVConnector: KVConnectorNIXLV2, ECConnector: ECConnectorNIXL},
			func(s *Server, w http.ResponseWriter, r *http.Request) {
				s.handleECNIXL(w, r, "10.0.0.1:8080", []string{"10.0.0.2:8080"}, reqcommon.APITypeChatCompletions)
			}),
		Entry("ec-shared-storage", Config{Port: "0", KVConnector: KVConnectorSharedStorage, ECConnector: ECExampleConnector},
			func(s *Server, w http.ResponseWriter, r *http.Request) {
				s.handleECSharedStorage(w, r, "10.0.0.1:8080", []string{"10.0.0.2:8080"}, reqcommon.APITypeChatCompletions)
			}),
		Entry("p2p decoder-only pull", Config{Port: "0", KVConnector: KVConnectorOffloading},
			func(s *Server, w http.ResponseWriter, r *http.Request) {
				s.decodeWithP2PSource(w, r, "10.0.0.2:8080")
			}),
		Entry("chunked decode", Config{Port: "0", DecodeChunkSize: 16},
			func(s *Server, w http.ResponseWriter, r *http.Request) {
				s.runChunkedDecode(w, r)
			}),
	)
})

func sidecarConnectionTestSetup(connector string) *sidecarTestInfo {
	testInfo := sidecarTestInfo{}

	testInfo.ctx = newTestContext()
	testInfo.ctx, testInfo.cancelFn = context.WithCancel(testInfo.ctx)
	testInfo.stoppedCh = make(chan struct{})

	// Decoder
	testInfo.decodeHandler = &mock.ChatCompletionHandler{
		Connector: connector,
		Role:      mock.RoleDecode,
	}
	testInfo.decodeBackend = httptest.NewServer(testInfo.decodeHandler)
	DeferCleanup(testInfo.decodeBackend.Close)

	// Prefiller
	testInfo.prefillHandler = &mock.ChatCompletionHandler{
		Connector: connector,
		Role:      mock.RolePrefill,
	}
	testInfo.prefillBackend = httptest.NewServer(testInfo.prefillHandler)
	DeferCleanup(testInfo.prefillBackend.Close)

	// Proxy
	url, err := url.Parse(testInfo.decodeBackend.URL)
	Expect(err).ToNot(HaveOccurred())
	testInfo.decodeURL = url
	cfg := Config{Port: "0", DecoderURL: testInfo.decodeURL, KVConnector: connector}
	testInfo.proxy = NewProxy(cfg)

	return &testInfo
}
