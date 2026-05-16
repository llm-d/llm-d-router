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
	"crypto/tls"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"time"

	. "github.com/onsi/ginkgo/v2" // nolint:revive
	. "github.com/onsi/gomega"    // nolint:revive
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"

	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/test/sidecar/mock"
)

func newTestContext() context.Context {
	logger := zap.New(
		zap.WriteTo(GinkgoWriter),
		zap.UseDevMode(true),
	)
	log.SetLogger(logger)
	ctx := context.Background()
	log.IntoContext(ctx, logger) // not strictly needed since we called SetLogger to set default
	return ctx
}

var _ = Describe("Reverse Proxy", func() {
	When("x-prefiller-url is not present", func() {
		DescribeTable("should forward requests to decode server",

			func(path string, secureProxy bool) {

				ctx := newTestContext()

				ackHandlerFn := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					w.WriteHeader(200)
				})

				decodeBackend := httptest.NewServer(ackHandlerFn)
				defer decodeBackend.Close()

				targetURL, err := url.Parse(decodeBackend.URL)
				Expect(err).ToNot(HaveOccurred())

				cfg := Config{
					Port:          "0",
					DecoderURL:    targetURL,
					SecureServing: secureProxy,
				}
				proxy := NewProxy(cfg)

				ctx, cancelFn := context.WithCancel(ctx)
				stoppedCh := make(chan struct{})

				go func() {
					defer GinkgoRecover()

					proxy.allowlistValidator = &AllowlistValidator{enabled: false}
					err := proxy.Start(ctx)
					Expect(err).ToNot(HaveOccurred())
					stoppedCh <- struct{}{}
				}()

				<-proxy.readyCh

				tr := &http.Transport{
					TLSClientConfig: &tls.Config{
						InsecureSkipVerify: true, // Skip certificate verification
					},
				}
				client := &http.Client{
					Transport: tr,
					Timeout:   10 * time.Second,
				}

				proxyAddr := proxy.addr.String() + path
				if secureProxy {
					proxyAddr = "https://" + proxyAddr
				} else {
					proxyAddr = "http://" + proxyAddr
				}
				resp, err := client.Get(proxyAddr)
				Expect(err).ToNot(HaveOccurred())

				_, err = io.ReadAll(resp.Body)
				Expect(err).ToNot(HaveOccurred())
				err = resp.Body.Close()
				Expect(err).ToNot(HaveOccurred())

				Expect(resp.StatusCode).To(BeNumerically("==", 200))

				cancelFn()
				<-stoppedCh
			},

			Entry("when the path is /v1/chat/completions and secure proxy is false", "/v1/chat/completions", false),
			Entry("when the path is /v1/completions and secure proxy is false", "/v1/completions", false),
			Entry("when the path is /v1/embeddings and secure proxy is false", "/v1/embeddings", false),
			Entry("when the path is /score and secure proxy is false", "/score", false),
			Entry("when the path is /healthz and secure proxy is false", "/healthz", false),

			Entry("when the path is /v1/chat/completions and secure proxy is true", "/v1/chat/completions", true),
			Entry("when the path is /v1/completions and secure proxy is true", "/v1/completions", true),
			Entry("when the path is /v1/embeddings and secure proxy is true", "/v1/embeddings", true),
			Entry("when the path is /score and secure proxy is true", "/score", true),
			Entry("when the path is /healthz and secure proxy is true", "/healthz", true),
		)
	})

	When("x-prefiller-url is present", func() {
		var decodeBackend *httptest.Server
		var decodeHandler *mock.ChatCompletionHandler
		var prefillBackend *httptest.Server
		var prefillHandler *mock.ChatCompletionHandler
		var decodeURL *url.URL

		BeforeEach(func() {
			// Decoder
			decodeHandler = &mock.ChatCompletionHandler{
				Role: mock.RoleDecode,
			}
			decodeBackend = httptest.NewServer(decodeHandler)
			DeferCleanup(decodeBackend.Close)

			// Prefiller
			prefillHandler = &mock.ChatCompletionHandler{
				Role: mock.RolePrefill,
			}
			prefillBackend = httptest.NewServer(prefillHandler)
			DeferCleanup(prefillBackend.Close)

			// Proxy
			url, err := url.Parse(decodeBackend.URL)
			Expect(err).ToNot(HaveOccurred())
			decodeURL = url
		})

		When("using NIXL connector V2", func() {
			var proxy *Server

			BeforeEach(func() {
				cfg := Config{Port: "0", DecoderURL: decodeURL, KVConnector: KVConnectorNIXLV2}
				proxy = NewProxy(cfg)

				decodeHandler.Connector = KVConnectorNIXLV2
				prefillHandler.Connector = KVConnectorNIXLV2
			})

			It("should successfully send request to 1. prefill 2. decode with the right fields (backward compatible behavior)", func() {
				ctx := newTestContext()
				ctx, cancelFn := context.WithCancel(ctx)
				stoppedCh := make(chan struct{})

				go func() {
					defer GinkgoRecover()

					proxy.allowlistValidator = &AllowlistValidator{enabled: false}
					err := proxy.Start(ctx)
					Expect(err).ToNot(HaveOccurred())
					stoppedCh <- struct{}{}
				}()

				<-proxy.readyCh
				proxyBaseAddr := "http://" + proxy.addr.String()

				By("sending a /v1/chat/completions request with prefill header")
				body := `{
        			"model": "Qwen/Qwen2-0.5B",
	        		"messages": [
    			      {"role": "user", "content": "Hello"}
        			],
        			"max_tokens": 50
				}`

				req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath, bytes.NewReader([]byte(body)))
				Expect(err).ToNot(HaveOccurred())
				req.Header.Add(routing.PrefillEndpointHeader, prefillBackend.URL)

				_, err = http.DefaultClient.Do(req)
				Expect(err).ToNot(HaveOccurred())

				Expect(prefillHandler.RequestCount.Load()).To(BeNumerically("==", 1))

				Expect(prefillHandler.CompletionRequests).To(HaveLen(1))
				prq1 := prefillHandler.CompletionRequests[0]

				Expect(prq1).ToNot(HaveKey(requestFieldDoRemoteDecode))
				Expect(prq1).To(HaveKey(requestFieldKVTransferParams))

				prq1kv, ok := prq1[requestFieldKVTransferParams].(map[string]any)
				Expect(ok).To(BeTrue())
				Expect(prq1kv).To(HaveKeyWithValue(requestFieldDoRemoteDecode, true))

				Expect(prq1).To(HaveKeyWithValue("stream", false))
				Expect(prq1).ToNot(HaveKey("stream_options"))

				Expect(prefillHandler.CompletionResponses).To(HaveLen(1))
				prp1 := prefillHandler.CompletionResponses[0]
				Expect(prp1).To(HaveKey(requestFieldKVTransferParams))

				prp1kv, ok := prp1[requestFieldKVTransferParams].(map[string]any)
				Expect(ok).To(BeTrue())

				Expect(prp1kv).To(HaveKey(requestFieldRemoteBlockIDs))
				Expect(prp1kv).To(HaveKey(requestFieldRemoteEngineID))

				Expect(decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))
				Expect(decodeHandler.CompletionRequests).To(HaveLen(1))
				drq1 := decodeHandler.CompletionRequests[0]
				Expect(drq1).To(HaveKey(requestFieldKVTransferParams))

				drq1kv, ok := drq1[requestFieldKVTransferParams].(map[string]any)
				Expect(ok).To(BeTrue())

				Expect(drq1kv).To(HaveKey(requestFieldRemoteBlockIDs))
				Expect(drq1kv).To(HaveKey(requestFieldRemoteEngineID))

				cancelFn()
				<-stoppedCh
			})

			It("should successfully send request to 1. prefill 2. decode with the right fields", func() {
				ctx := newTestContext()
				ctx, cancelFn := context.WithCancel(ctx)
				stoppedCh := make(chan struct{})

				go func() {
					defer GinkgoRecover()

					proxy.allowlistValidator = &AllowlistValidator{enabled: false}
					err := proxy.Start(ctx)
					Expect(err).ToNot(HaveOccurred())
					stoppedCh <- struct{}{}
				}()

				<-proxy.readyCh
				proxyBaseAddr := "http://" + proxy.addr.String()

				By("sending a /v1/chat/completions request with prefill header")
				body := `{
        			"model": "Qwen/Qwen2-0.5B",
	        		"messages": [
    			      {"role": "user", "content": "Hello"}
        			],
        			"max_tokens": 50
				}`

				req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath, bytes.NewReader([]byte(body)))
				Expect(err).ToNot(HaveOccurred())
				req.Header.Add(routing.PrefillEndpointHeader, prefillBackend.URL[len("http://"):])

				_, err = http.DefaultClient.Do(req)
				Expect(err).ToNot(HaveOccurred())

				Expect(prefillHandler.RequestCount.Load()).To(BeNumerically("==", 1))

				Expect(prefillHandler.CompletionRequests).To(HaveLen(1))
				prq1 := prefillHandler.CompletionRequests[0]

				Expect(prq1).ToNot(HaveKey(requestFieldDoRemoteDecode))
				Expect(prq1).To(HaveKey(requestFieldKVTransferParams))

				prq1kv, ok := prq1[requestFieldKVTransferParams].(map[string]any)
				Expect(ok).To(BeTrue())
				Expect(prq1kv).To(HaveKeyWithValue(requestFieldDoRemoteDecode, true))

				Expect(prq1).To(HaveKeyWithValue("stream", false))
				Expect(prq1).ToNot(HaveKey("stream_options"))

				Expect(prefillHandler.CompletionResponses).To(HaveLen(1))
				prp1 := prefillHandler.CompletionResponses[0]
				Expect(prp1).To(HaveKey(requestFieldKVTransferParams))

				prp1kv, ok := prp1[requestFieldKVTransferParams].(map[string]any)
				Expect(ok).To(BeTrue())

				Expect(prp1kv).To(HaveKey(requestFieldRemoteBlockIDs))
				Expect(prp1kv).To(HaveKey(requestFieldRemoteEngineID))

				Expect(decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))
				Expect(decodeHandler.CompletionRequests).To(HaveLen(1))
				drq1 := decodeHandler.CompletionRequests[0]
				Expect(drq1).To(HaveKey(requestFieldKVTransferParams))

				drq1kv, ok := drq1[requestFieldKVTransferParams].(map[string]any)
				Expect(ok).To(BeTrue())

				Expect(drq1kv).To(HaveKey(requestFieldRemoteBlockIDs))
				Expect(drq1kv).To(HaveKey(requestFieldRemoteEngineID))

				cancelFn()
				<-stoppedCh
			})
		})
	})
})
