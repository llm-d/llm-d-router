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
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2" // nolint:revive
	. "github.com/onsi/gomega"    // nolint:revive

	"github.com/llm-d/llm-d-router/pkg/common/routing"
)

// parallelCommitEnv is a minimal proxy harness for the MoRI-IO parallel WRITE
// dispatch commit-point tests. Unlike startMoRIProxy (which wires the shared
// mock ChatCompletionHandler), it accepts arbitrary prefill/decode handlers so
// a leg can fail, block, or stream on demand.
type parallelCommitEnv struct {
	proxy       *Server
	baseAddr    string
	prefillHost string
}

func startParallelCommitProxy(prefill, decode http.Handler, mutate func(cfg *Config)) *parallelCommitEnv {
	ctx, cancelFn := context.WithCancel(newTestContext())
	stoppedCh := make(chan struct{})

	decodeBackend := httptest.NewServer(decode)
	DeferCleanup(decodeBackend.Close)

	prefillBackend := httptest.NewServer(prefill)
	DeferCleanup(prefillBackend.Close)

	decodeURL, err := url.Parse(decodeBackend.URL)
	Expect(err).ToNot(HaveOccurred())

	cfg := Config{
		Port:                       "0",
		DecoderURL:                 decodeURL,
		KVConnector:                KVConnectorNIXLV2,
		MoRIIOWriteMode:            true,
		MoRIIOParallelDispatch:     true,
		MoRIIODecodePodIP:          decodeURL.Hostname(),
		MoRIIODecodeNotifyPort:     61005,
		MoRIIODecodeHandshakePort:  6301,
		MoRIIOPrefillNotifyPort:    61006,
		MoRIIOPrefillHandshakePort: 6302,
		MoRIIOTPSize:               1,
		MoRIIODPSize:               1,
	}
	if mutate != nil {
		mutate(&cfg)
	}

	proxy := NewProxy(cfg)
	go func() {
		defer GinkgoRecover()
		proxy.allowlistValidator = &AllowlistValidator{enabled: false}
		Expect(proxy.Start(ctx)).To(Succeed())
		stoppedCh <- struct{}{}
	}()
	<-proxy.readyCh

	env := &parallelCommitEnv{
		proxy:       proxy,
		baseAddr:    "http://" + proxy.addr.String(),
		prefillHost: prefillBackend.URL[len("http://"):],
	}
	DeferCleanup(func() {
		cancelFn()
		<-stoppedCh
	})
	return env
}

// send issues a chat-completions request bounded by clientTimeout, so a hang in
// the proxy surfaces as a client error and fails the test deterministically
// rather than blocking the suite forever.
func (env *parallelCommitEnv) send(clientTimeout time.Duration) (int, http.Header, string, error) {
	req, err := http.NewRequest(http.MethodPost, env.baseAddr+ChatCompletionsPath, strings.NewReader(chatCompletionsRequestBody))
	Expect(err).ToNot(HaveOccurred())
	req.Header.Add(routing.PrefillEndpointHeader, env.prefillHost)

	client := &http.Client{Timeout: clientTimeout}
	rp, err := client.Do(req)
	if err != nil {
		return 0, nil, "", err
	}
	defer rp.Body.Close()
	body, _ := io.ReadAll(rp.Body) //nolint:errcheck
	return rp.StatusCode, rp.Header, string(body), nil
}

var _ = Describe("NIXL Connector (v2) parallel WRITE dispatch commit point", func() {

	It("returns the PREFILL error (not decode's 200) when prefill fails", func() {
		// Decode would happily return 200, but prefill fails: the client must
		// see the prefill error, and the request must not hang.
		prefill := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"error":"prefill boom"}`))
		})
		decode := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"id":"decode-should-be-discarded","choices":[]}`))
		})

		env := startParallelCommitProxy(prefill, decode, nil)

		status, _, body, err := env.send(10 * time.Second)
		Expect(err).ToNot(HaveOccurred(), "request must complete, not hang")
		Expect(status).To(Equal(http.StatusInternalServerError))
		Expect(status).ToNot(Equal(http.StatusOK))
		Expect(body).To(ContainSubstring("prefill boom"))
		Expect(body).ToNot(ContainSubstring("decode-should-be-discarded"))
	})

	It("does not hang when prefill and decode both block: the KV-wait backstop returns 504", func() {
		// Both legs block (KV that never arrives) until either the sidecar
		// cancels their request context or the test tears down. The bounded
		// backstop timeout plus the shared cancelable context must abort and
		// return 504 rather than hang.
		stop := make(chan struct{})
		block := func(_ http.ResponseWriter, r *http.Request) {
			select {
			case <-r.Context().Done():
			case <-stop:
			}
		}
		prefill := http.HandlerFunc(block)
		decode := http.HandlerFunc(block)

		env := startParallelCommitProxy(prefill, decode, func(cfg *Config) {
			cfg.MoRIIOParallelDecodeWaitTimeout = 300 * time.Millisecond
		})
		// Registered after startParallelCommitProxy, so it runs BEFORE the
		// backends' Close cleanups (LIFO), guaranteeing the blocked handlers
		// unblock and the httptest servers can shut down.
		DeferCleanup(func() { close(stop) })

		start := time.Now()
		// Client timeout is far larger than the backstop, so a 504 means the
		// backstop fired; a client-side timeout error would mean it hung.
		status, _, _, err := env.send(8 * time.Second)
		elapsed := time.Since(start)

		Expect(err).ToNot(HaveOccurred(), "request must return via the backstop, not hang")
		Expect(status).To(Equal(http.StatusGatewayTimeout))
		Expect(elapsed).To(BeNumerically("<", 5*time.Second), "should return shortly after the 300ms backstop")
	})

	It("commits and streams decode's 200 (SSE body intact) when prefill succeeds", func() {
		// Prefill succeeds (body is irrelevant to the parallel path), decode
		// streams SSE. Buffering until the commit point must preserve decode's
		// status, headers, and full body.
		prefill := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"kv_transfer_params":{}}`))
		})
		decode := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", eventStreamContentType)
			w.WriteHeader(http.StatusOK)
			flusher, _ := w.(http.Flusher)
			_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n"))
			if flusher != nil {
				flusher.Flush()
			}
			_, _ = w.Write([]byte("data: {\"choices\":[],\"usage\":{\"total_tokens\":65}}\n\ndata: [DONE]\n\n"))
			if flusher != nil {
				flusher.Flush()
			}
		})

		env := startParallelCommitProxy(prefill, decode, nil)

		status, hdr, body, err := env.send(10 * time.Second)
		Expect(err).ToNot(HaveOccurred())
		Expect(status).To(Equal(http.StatusOK))
		Expect(hdr.Get("Content-Type")).To(ContainSubstring(eventStreamContentType))
		Expect(body).To(ContainSubstring("hello"))
		Expect(body).To(ContainSubstring("[DONE]"))
	})
})
