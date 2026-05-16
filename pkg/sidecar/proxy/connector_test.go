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
	"net/http"
	"net/http/httptest"
	"net/url"

	. "github.com/onsi/ginkgo/v2" // nolint:revive
	. "github.com/onsi/gomega"    // nolint:revive

	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/test/sidecar/mock"
)

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
				//nolint:goconst
				body := `{
				"model": "Qwen/Qwen2-0.5B",
				"messages": [
				  {"role": "user", "content": "Hello"}
				],
				"max_tokens": 50,
				"max_completion_tokens": 100
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
		})
	}
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
