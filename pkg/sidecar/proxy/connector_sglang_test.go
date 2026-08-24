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
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync/atomic"
	"time"

	. "github.com/onsi/ginkgo/v2" // nolint:revive
	. "github.com/onsi/gomega"    // nolint:revive

	"github.com/llm-d/llm-d-router/pkg/common/routing"
)

var _ = Describe("SGLang Connector", func() {

	var testInfo *sidecarTestInfo

	BeforeEach(func() {
		// Mock testing setup using the SGLang connector mode
		testInfo = sidecarConnectionTestSetup(KVConnectorSGLang)
	})

	It("should use the configured bootstrap authority without deriving the prefill rank", func() {
		previousHost := sglangBootstrapHost
		previousPort := sglangBootstrapPort
		DeferCleanup(func() {
			sglangBootstrapHost = previousHost
			sglangBootstrapPort = previousPort
		})

		sglangBootstrapHost = "prefill-bootstrap.example"
		sglangBootstrapPort = 8000

		request := testInfo.proxy.addSGLangBootstrapInfo(map[string]interface{}{}, "10.0.0.8:8002", 9)

		Expect(request[requestFieldBootstrapHost]).To(Equal("prefill-bootstrap.example"))
		Expect(request[requestFieldBootstrapPort]).To(Equal(8000))
		Expect(request[requestFieldBootstrapRoom]).To(Equal(int64(9)))
		Expect(request).ToNot(HaveKey("disagg_prefill_dp_rank"))

		sglangBootstrapHost = ""
		request = testInfo.proxy.addSGLangBootstrapInfo(map[string]interface{}{}, "10.0.0.8:8002", 10)
		Expect(request[requestFieldBootstrapHost]).To(Equal("10.0.0.8"))
		Expect(request[requestFieldBootstrapRoom]).To(Equal(int64(10)))
	})

	It("should claim only the native generate path for the configured protocol", func() {
		sglangMux := testInfo.proxy.createRoutes()
		_, pattern := sglangMux.Handler(httptest.NewRequest(http.MethodPost, sglangGeneratePath, nil))
		Expect(pattern).To(Equal("POST " + sglangGeneratePath))
		_, pattern = sglangMux.Handler(httptest.NewRequest(http.MethodPost, GeneratePath, nil))
		Expect(pattern).To(Equal("/"))

		vllmProxy := NewProxy(Config{DecoderURL: testInfo.decodeURL, KVConnector: KVConnectorMooncake})
		vllmMux := vllmProxy.createRoutes()
		_, pattern = vllmMux.Handler(httptest.NewRequest(http.MethodPost, GeneratePath, nil))
		Expect(pattern).To(Equal("POST " + GeneratePath))
		_, pattern = vllmMux.Handler(httptest.NewRequest(http.MethodPost, sglangGeneratePath, nil))
		Expect(pattern).To(Equal("/"))
	})

	It("should successfully send concurrent requests to prefill and decode with bootstrap info", func() {
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

		By("sending a tokenized /generate request with prefill header")
		body := `{"input_ids":[1,2,3],"sampling_params":{"max_new_tokens":8}}`

		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+sglangGeneratePath, bytes.NewReader([]byte(body)))
		Expect(err).ToNot(HaveOccurred())

		prefillHostPort := testInfo.prefillBackend.URL[len("http://"):]
		req.Header.Add(routing.PrefillEndpointHeader, prefillHostPort)

		rp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())

		if rp.StatusCode != 200 {
			bp, _ := io.ReadAll(rp.Body) //nolint:errcheck
			Fail(string(bp))
		}

		// Because SGLang connector sends requests concurrently (prefill in goroutine),
		// wait until the prefill handler has finished processing before reading its state.
		Eventually(testInfo.prefillHandler.RequestCount.Load).Should(Equal(int32(1)))

		// Validate prefill request
		prefillReqs := testInfo.prefillHandler.GetCompletionRequests()
		Expect(prefillReqs).To(HaveLen(1))
		prq1 := prefillReqs[0]

		// Validate decode request
		Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		decodeReqs := testInfo.decodeHandler.GetCompletionRequests()
		Expect(decodeReqs).To(HaveLen(1))
		drq1 := decodeReqs[0]

		// Bootstrap validations for prefill
		Expect(prq1).To(HaveKey(requestFieldBootstrapHost))
		Expect(prq1).To(HaveKey(requestFieldBootstrapPort))
		Expect(prq1).To(HaveKey(requestFieldBootstrapRoom))

		expectedHost := extractHost(prefillHostPort)
		Expect(prq1[requestFieldBootstrapHost]).To(Equal(expectedHost))
		Expect(prq1[requestFieldBootstrapPort]).To(Equal(float64(sglangBootstrapPort)))
		Expect(prq1[requestFieldBootstrapRoom]).ToNot(BeNil())

		// Bootstrap validations for decode
		Expect(drq1).To(HaveKey(requestFieldBootstrapHost))
		Expect(drq1).To(HaveKey(requestFieldBootstrapPort))
		Expect(drq1).To(HaveKey(requestFieldBootstrapRoom))

		Expect(drq1[requestFieldBootstrapHost]).To(Equal(expectedHost))
		Expect(drq1[requestFieldBootstrapPort]).To(Equal(float64(sglangBootstrapPort)))
		Expect(drq1[requestFieldBootstrapRoom]).To(Equal(prq1[requestFieldBootstrapRoom])) // Room ID must match
		Expect(drq1["input_ids"]).To(Equal(prq1["input_ids"]))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	It("should return a prefill failure instead of an early decode success", func() {
		testInfo.decodeBackend.Close()
		testInfo.prefillBackend.Close()

		decodeFlushed := make(chan struct{})
		testInfo.prefillBackend = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			<-decodeFlushed
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"error":"prefill failed"}`))
		}))
		testInfo.decodeBackend = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"text":"must not escape"}`))
			w.(http.Flusher).Flush()
			close(decodeFlushed)
			<-r.Context().Done()
			panic(http.ErrAbortHandler)
		}))
		decodeURL, err := url.Parse(testInfo.decodeBackend.URL)
		Expect(err).ToNot(HaveOccurred())
		testInfo.proxy = NewProxy(Config{Port: "0", DecoderURL: decodeURL, KVConnector: KVConnectorSGLang})

		go func() {
			defer GinkgoRecover()
			testInfo.proxy.allowlistValidator = &AllowlistValidator{enabled: false}
			Expect(testInfo.proxy.Start(testInfo.ctx)).To(Succeed())
			testInfo.stoppedCh <- struct{}{}
		}()
		<-testInfo.proxy.readyCh

		req, err := http.NewRequest(
			http.MethodPost,
			"http://"+testInfo.proxy.addr.String()+sglangGeneratePath,
			bytes.NewBufferString(`{"input_ids":[1],"sampling_params":{"max_new_tokens":1}}`),
		)
		Expect(err).ToNot(HaveOccurred())
		req.Header.Set(routing.PrefillEndpointHeader, testInfo.prefillBackend.URL[len("http://"):])
		resp, err := (&http.Client{Timeout: 5 * time.Second}).Do(req)
		Expect(err).ToNot(HaveOccurred())
		defer resp.Body.Close()
		responseBody, err := io.ReadAll(resp.Body)
		Expect(err).ToNot(HaveOccurred())

		Expect(resp.StatusCode).To(Equal(http.StatusInternalServerError))
		Expect(string(responseBody)).To(ContainSubstring("prefill failed"))
		Expect(string(responseBody)).ToNot(ContainSubstring("must not escape"))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	It("should not panic when prefill response is slower than decode response", func() {
		// Stop previously injected servers
		testInfo.decodeBackend.Close()
		testInfo.prefillBackend.Close()

		var prefillFinished atomic.Bool

		slowPrefill := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			testInfo.prefillHandler.ServeHTTP(w, r)
			time.Sleep(300 * time.Millisecond) // Simulated load delay on KV Cache
			prefillFinished.Store(true)
		})
		testInfo.prefillBackend = httptest.NewServer(slowPrefill)

		fastDecode := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			testInfo.decodeHandler.ServeHTTP(w, r)
		})
		testInfo.decodeBackend = httptest.NewServer(fastDecode)
		testInfo.decodeURL, _ = url.Parse(testInfo.decodeBackend.URL)

		// Re-initialize proxy to fetch the new mock addresses
		cfg := Config{
			Port:        "0",
			DecoderURL:  testInfo.decodeURL,
			KVConnector: KVConnectorSGLang,
		}
		testInfo.proxy = NewProxy(cfg)

		go func() {
			defer GinkgoRecover()
			testInfo.proxy.allowlistValidator = &AllowlistValidator{enabled: false}
			err := testInfo.proxy.Start(testInfo.ctx)
			Expect(err).ToNot(HaveOccurred())
			testInfo.stoppedCh <- struct{}{}
		}()

		<-testInfo.proxy.readyCh
		proxyBaseAddr := "http://" + testInfo.proxy.addr.String()

		body := `{"input_ids":[1],"sampling_params":{"max_new_tokens":1}}`
		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+sglangGeneratePath, bytes.NewReader([]byte(body)))
		Expect(err).ToNot(HaveOccurred())

		prefillHostPort := testInfo.prefillBackend.URL[len("http://"):]
		req.Header.Add(routing.PrefillEndpointHeader, prefillHostPort)

		// Submit request. Decode output remains buffered until prefill completes.
		rp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())
		Expect(rp.StatusCode).To(Equal(200))

		// The original panicking goroutine takes 300ms total. Give it time to attempt finishing up!
		time.Sleep(500 * time.Millisecond)

		Expect(prefillFinished.Load()).To(BeTrue())
		Expect(testInfo.prefillHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	It("should bound a stalled prefill response", func() {
		previousTimeout := sglangPrefillWaitTimeout
		sglangPrefillWaitTimeout = 50 * time.Millisecond
		DeferCleanup(func() { sglangPrefillWaitTimeout = previousTimeout })

		testInfo.prefillBackend.Close()
		testInfo.prefillBackend = httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
			<-r.Context().Done()
		}))
		testInfo.proxy.decoderProxy = http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
			<-r.Context().Done()
		})

		recorder := httptest.NewRecorder()
		request := httptest.NewRequest(http.MethodPost, sglangGeneratePath, nil)
		started := time.Now()
		testInfo.proxy.handleSGLangConcurrentRequests(
			recorder,
			request,
			[]byte(`{"input_ids":[1]}`),
			testInfo.prefillBackend.URL[len("http://"):],
		)

		Expect(recorder.Code).To(Equal(http.StatusGatewayTimeout))
		Expect(time.Since(started)).To(BeNumerically("<", time.Second))
	})
})
