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
	"context"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	. "github.com/onsi/ginkgo/v2" // nolint:revive
	. "github.com/onsi/gomega"    // nolint:revive
	"golang.org/x/sync/errgroup"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
	fwknet "github.com/llm-d/llm-d-router/test/framework/net"
	sidecarmock "github.com/llm-d/llm-d-router/test/sidecar/mock"
)

const (
	testDataParallelSize = 2
)

var _ = Describe("Data Parallel support", func() {
	It("should preserve the NIXLv2 request ID generator when cloning", func() {
		const requestID = "fixed-request-id"
		proxy := NewProxy(Config{})
		proxy.nixlRequestIDFn = func() (string, error) {
			return requestID, nil
		}

		clone := proxy.Clone()
		got, err := clone.nixlRequestIDFn()

		Expect(err).ToNot(HaveOccurred())
		Expect(got).To(Equal(requestID))
	})

	When("configured with --data-parallel-size > 1", func() {
		It("should start and stop multiple ranks with SSRF protection enabled", func() {
			listeners := make([]net.Listener, testDataParallelSize)
			for rank := range listeners {
				listener, err := fwknet.ReserveListener()
				Expect(err).ToNot(HaveOccurred())
				listeners[rank] = listener
				DeferCleanup(func() { _ = listener.Close() })
			}
			decoderURL, err := url.Parse("http://localhost:8000")
			Expect(err).ToNot(HaveOccurred())
			proxy := NewProxy(Config{
				Port:                 "0",
				DecoderURL:           decoderURL,
				DataParallelSize:     testDataParallelSize,
				EnableSSRFProtection: true,
			})
			proxy.allowlistValidator = newTestAllowlistValidator()
			proxy.HTTPListener = listeners[0]
			proxy.DataParallelListeners = listeners[1:]
			ctx, cancel := context.WithCancel(newTestContext())
			done := make(chan error, 1)
			go func() { done <- proxy.Start(ctx) }()
			DeferCleanup(func() {
				cancel()
				var serveErr error
				Eventually(done, 3*time.Second).Should(Receive(&serveErr))
				Expect(serveErr).ToNot(HaveOccurred())
			})

			Eventually(proxy.readyCh, 3*time.Second).Should(BeClosed())
			client := &http.Client{Timeout: time.Second}
			for _, listener := range listeners {
				baseURL := "http://" + listener.Addr().String()
				Eventually(func() bool {
					response, err := client.Get(baseURL + "/health")
					if err != nil {
						return false
					}
					defer response.Body.Close()
					return response.StatusCode == http.StatusOK
				}, 3*time.Second, 10*time.Millisecond).Should(BeTrue())
			}
			cancel()
		})

		DescribeTable("keeps inference on the selected rank", func(ecConnector string, withPrefill bool) {
			var rank0Requests, rank1Requests, encoderRequests, prefillRequests atomic.Int32
			rank1Backend := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
				rank1Requests.Add(1)
				writer.Header().Set("Content-Type", "application/json")
				_, _ = writer.Write([]byte(`{"rank":1}`))
			}))
			DeferCleanup(rank1Backend.Close)
			encoder := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
				encoderRequests.Add(1)
				writer.Header().Set("Content-Type", "application/json")
				_, _ = writer.Write([]byte(`{"ec_transfer_params":{"image":{}}}`))
			}))
			DeferCleanup(encoder.Close)
			prefill := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
				prefillRequests.Add(1)
				writer.Header().Set("Content-Type", "application/json")
				_, _ = writer.Write([]byte(`{"kv_transfer_params":{}}`))
			}))
			DeferCleanup(prefill.Close)

			// Virtual base ports are one below rank 1's bound ports,
			// avoiding the need to reserve contiguous ports.
			rank1Listener, err := fwknet.ReserveListener()
			Expect(err).ToNot(HaveOccurred())
			DeferCleanup(func() { _ = rank1Listener.Close() })
			rank1Port := rank1Listener.Addr().(*net.TCPAddr).Port
			rank1URL, err := url.Parse(rank1Backend.URL)
			Expect(err).ToNot(HaveOccurred())
			rank1DecoderPort, err := strconv.Atoi(rank1URL.Port())
			Expect(err).ToNot(HaveOccurred())
			decoderURL, err := url.Parse("http://localhost:" + strconv.Itoa(rank1DecoderPort-1))
			Expect(err).ToNot(HaveOccurred())
			proxy := NewProxy(Config{
				Port:             strconv.Itoa(rank1Port - 1),
				DecoderURL:       decoderURL,
				DataParallelSize: testDataParallelSize,
				KVConnector:      constants.KVConnectorNIXLV2,
				ECConnector:      ecConnector,
			})
			proxy.allowlistValidator = &AllowlistValidator{}
			proxy.DataParallelListeners = []net.Listener{rank1Listener}
			proxy.handler = proxy.createRoutes()
			// A connector bound to the source Server reaches this rank-0 stub.
			proxy.decoderProxy = http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
				rank0Requests.Add(1)
				writer.Header().Set("Content-Type", "application/json")
				_, _ = writer.Write([]byte(`{"rank":0}`))
			})

			ctx, cancel := context.WithCancel(newTestContext())
			group, ctx := errgroup.WithContext(ctx)
			DeferCleanup(func() {
				cancel()
				Expect(group.Wait()).To(Succeed())
			})
			Expect(proxy.startDataParallel(ctx, group)).To(Succeed())
			client := &http.Client{Timeout: 2 * time.Second}
			baseURL := "http://" + rank1Listener.Addr().String()
			Eventually(func() bool {
				response, err := client.Get(baseURL + "/health")
				if err != nil {
					return false
				}
				defer response.Body.Close()
				return response.StatusCode == http.StatusOK
			}, "3s", "20ms").Should(BeTrue())

			body := `{"model":"m","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/image.png"}}]}]}`
			// Select rank 1 by port so the legacy DP header cannot mask a wrong binding.
			request, err := http.NewRequest(http.MethodPost, baseURL+reqcommon.PathChatCompletions, strings.NewReader(body))
			Expect(err).ToNot(HaveOccurred())
			request.Header.Set("Content-Type", "application/json")
			if ecConnector != "" {
				request.Header.Set(routing.EncoderEndpointsHeader, strings.TrimPrefix(encoder.URL, "http://"))
			}
			if withPrefill {
				request.Header.Set(routing.PrefillEndpointHeader, strings.TrimPrefix(prefill.URL, "http://"))
			}
			response, err := client.Do(request)
			Expect(err).ToNot(HaveOccurred())
			defer response.Body.Close()
			responseBody, err := io.ReadAll(response.Body)
			Expect(err).ToNot(HaveOccurred())
			Expect(response.StatusCode).To(Equal(http.StatusOK), string(responseBody))
			Expect(string(responseBody)).To(MatchJSON(`{"rank":1}`))
			Expect(rank0Requests.Load()).To(BeZero())
			Expect(rank1Requests.Load()).To(Equal(int32(1)))
			if ecConnector != "" {
				Expect(encoderRequests.Load()).To(Equal(int32(1)))
			} else {
				Expect(encoderRequests.Load()).To(BeZero())
			}
			if withPrefill {
				Expect(prefillRequests.Load()).To(Equal(int32(1)))
			} else {
				Expect(prefillRequests.Load()).To(BeZero())
			}

			rank0Request := httptest.NewRequest(http.MethodPost, reqcommon.PathChatCompletions, strings.NewReader(body))
			rank0Request.Header = request.Header.Clone()
			rank0Response := httptest.NewRecorder()
			proxy.handler.ServeHTTP(rank0Response, rank0Request)
			Expect(rank0Response.Code).To(Equal(http.StatusOK))
			Expect(rank0Response.Body.String()).To(MatchJSON(`{"rank":0}`))
			Expect(rank0Requests.Load()).To(Equal(int32(1)))
			Expect(rank1Requests.Load()).To(Equal(int32(1)))
		},
			Entry("ec-example encode/decode", constants.ECExampleConnector, false),
			Entry("ec-example encode/prefill/decode", constants.ECExampleConnector, true),
			Entry("ec-nixl encode/decode", constants.ECConnectorNIXL, false),
			Entry("ec-nixl encode/prefill/decode", constants.ECConnectorNIXL, true),
			Entry("decoder only", "", false),
			Entry("prefill/decode", "", true),
		)

		It("should create an extra proxy", func() {
			ctx := newTestContext()
			ctx, cancel := context.WithCancel(ctx)
			grp, ctx := errgroup.WithContext(ctx)

			// Rank-1 clone binds this listener. Rank 0 is config.Port for
			// DP rank math and is not served in this test.
			rank1Ln, err := fwknet.ReserveListener()
			Expect(err).ToNot(HaveOccurred())
			DeferCleanup(func() { _ = rank1Ln.Close() })
			rank1Port := rank1Ln.Addr().(*net.TCPAddr).Port
			fakeProxyPort := rank1Port - 1

			// The data parallel support, assumes that the decoders are
			// listening on a set of contiguous ports. Get a free port
			// and fake the first one by being the free port minus one.
			rank1Handler := sidecarmock.GenericHandler{}
			rank1Server := httptest.NewServer(&rank1Handler)
			tempURL, err := url.Parse(rank1Server.URL)
			Expect(err).ToNot(HaveOccurred())
			tmpPort, err := strconv.Atoi(tempURL.Port())
			Expect(err).ToNot(HaveOccurred())
			fakeDecodePort := tmpPort - 1

			DeferCleanup(os.Setenv, "POD_IP", os.Getenv("POD_IP"))
			err = os.Setenv("POD_IP", testLoopbackIP)
			Expect(err).ToNot(HaveOccurred())

			decodeURL, err := url.Parse("http://localhost:" + strconv.Itoa(fakeDecodePort))
			Expect(err).ToNot(HaveOccurred())
			cfg := Config{
				Port:             strconv.Itoa(fakeProxyPort),
				DecoderURL:       decodeURL,
				KVConnector:      constants.KVConnectorNIXLV2,
				DataParallelSize: testDataParallelSize,
			}
			theProxy := NewProxy(cfg)
			theProxy.DataParallelListeners = []net.Listener{rank1Ln}
			theProxy.allowlistValidator, err = NewAllowlistValidator(false, routing.InferencePoolAPIGroup, "", "")
			Expect(err).ToNot(HaveOccurred())

			err = theProxy.startDataParallel(ctx, grp)
			Expect(err).ToNot(HaveOccurred())

			Expect(theProxy.dataParallelProxies).To(HaveLen(testDataParallelSize))
			handler := theProxy.dataParallelProxies["127.0.0.1:"+strconv.Itoa(rank1Port)]
			Expect(handler).ToNot(BeNil())

			rank1Addr := rank1Ln.Addr().String()
			healthClient := &http.Client{Timeout: 200 * time.Millisecond}
			Eventually(func() bool {
				resp, err := healthClient.Get("http://" + rank1Addr + "/health")
				if err != nil {
					return false
				}
				defer resp.Body.Close()
				return resp.StatusCode == http.StatusOK
			}, "2s", "20ms").Should(BeTrue())

			rank0Handler := sidecarmock.GenericHandler{}
			rank0Server := httptest.NewServer(&rank0Handler)
			tempURL, err = url.Parse(rank0Server.URL)
			Expect(err).ToNot(HaveOccurred())
			theProxy.config.DecoderURL = tempURL

			proxyHandler := theProxy.createRoutes()
			req := httptest.NewRequest("POST", "/v1/completions", nil)
			resp := httptest.NewRecorder()
			proxyHandler.ServeHTTP(resp, req)
			Expect(int(rank0Handler.RequestCount.Load())).To(Equal(1))
			Expect(int(rank1Handler.RequestCount.Load())).To(Equal(0))

			req.Header.Add(routing.DataParallelEndpointHeader, "127.0.0.1:"+strconv.Itoa(rank1Port))
			resp = httptest.NewRecorder()
			proxyHandler.ServeHTTP(resp, req)
			Expect(int(rank0Handler.RequestCount.Load())).To(Equal(1))
			Expect(int(rank1Handler.RequestCount.Load())).To(Equal(1))

			rank0Server.Close()
			rank1Server.Close()

			cancel()
			err = grp.Wait()
			Expect(err).ToNot(HaveOccurred())
		})
	})
})
