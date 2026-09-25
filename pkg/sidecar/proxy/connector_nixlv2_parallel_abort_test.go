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
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2" // nolint:revive
	. "github.com/onsi/gomega"    // nolint:revive

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
)

// abortingDecodeHandler hijacks the decode connection and sends a chunked
// response that it never terminates, so the sidecar's reverse proxy fails
// mid-copy and panics with http.ErrAbortHandler on the decode goroutine. That
// panic is the one net/http does not recover off the request goroutine.
//
// The response is chunked rather than Content-Length delimited on purpose: the
// sidecar then streams to its own client chunked too, so an unhandled decode
// abort shows up as a cleanly terminated stream, which is the behaviour under
// test. A short Content-Length would make the client error out on its own and
// the test would pass whether or not the abort is handled.
//
// delivered is sent as one chunk before the connection is closed; beforeClose
// runs between the two, and closed is closed once the connection is gone.
func abortingDecodeHandler(delivered string, closed chan struct{}, beforeClose func()) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		defer GinkgoRecover()
		hijacker, ok := w.(http.Hijacker)
		Expect(ok).To(BeTrue(), "decode backend must support hijacking")
		conn, buf, err := hijacker.Hijack()
		Expect(err).ToNot(HaveOccurred())
		head := "HTTP/1.1 200 OK\r\n" +
			"Content-Type: " + eventStreamContentType + "\r\n" +
			"Transfer-Encoding: chunked\r\n\r\n"
		if delivered != "" {
			head += fmt.Sprintf("%x\r\n%s\r\n", len(delivered), delivered)
		}
		_, err = buf.WriteString(head)
		Expect(err).ToNot(HaveOccurred())
		Expect(buf.Flush()).To(Succeed())
		if beforeClose != nil {
			beforeClose()
		}
		// Closed without the terminating zero-length chunk.
		Expect(conn.Close()).To(Succeed())
		close(closed)
	})
}

// readBodyErr issues the request like parallelCommitEnv.send but returns the
// error from reading the body, which is where a torn-down connection surfaces.
func readBodyErr(env *parallelCommitEnv, clientTimeout time.Duration) error {
	req, err := http.NewRequest(http.MethodPost, env.baseAddr+reqcommon.PathChatCompletions, strings.NewReader(chatCompletionsRequestBody))
	Expect(err).ToNot(HaveOccurred())
	req.Header.Add(routing.PrefillEndpointHeader, env.prefillHost)

	client := &http.Client{Timeout: clientTimeout}
	rp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer rp.Body.Close()
	_, err = io.ReadAll(rp.Body)
	return err
}

var _ = Describe("NIXL Connector (v2) parallel WRITE dispatch decode abort", func() {

	It("returns 502 when decode aborts before its response reaches the client", func() {
		decodeClosed := make(chan struct{})
		// Prefill succeeds only well after decode has aborted, so the commit
		// point finds an aborted writer and the response is still unwritten.
		prefill := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			<-decodeClosed
			time.Sleep(300 * time.Millisecond)
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"kv_transfer_params":{}}`))
		})
		decode := abortingDecodeHandler("", decodeClosed, nil)

		env := startParallelCommitProxy(prefill, decode, nil)

		status, _, body, err := env.send(10 * time.Second)
		Expect(err).ToNot(HaveOccurred(), "the client must get a response, not a dropped connection")
		Expect(status).To(Equal(http.StatusBadGateway))
		Expect(status).ToNot(Equal(http.StatusOK), "an empty 200 would hide the decode failure")
		Expect(body).To(ContainSubstring("decode aborted"))
	})

	It("tears the connection down when decode aborts after its response reaches the client", func() {
		decodeClosed := make(chan struct{})
		prefill := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"kv_transfer_params":{}}`))
		})
		// One SSE event is delivered and relayed to the client, then the body is
		// truncated: the client must not see that stream as cleanly finished.
		decode := abortingDecodeHandler(
			"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n",
			decodeClosed,
			func() { time.Sleep(300 * time.Millisecond) },
		)

		env := startParallelCommitProxy(prefill, decode, nil)

		Expect(readBodyErr(env, 10*time.Second)).To(HaveOccurred(),
			"a stream that died mid-body must not terminate cleanly")
	})

	It("keeps the KV-wait timeout's 504 when decode aborts as well", func() {
		// Both failures race: the backstop owns the response, so the decode
		// abort must not write a second status over the 504.
		decodeClosed := make(chan struct{})
		stop := make(chan struct{})
		prefill := http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
			select {
			case <-r.Context().Done():
			case <-stop:
			}
		})
		decode := abortingDecodeHandler("", decodeClosed, nil)

		env := startParallelCommitProxy(prefill, decode, func(cfg *Config) {
			cfg.MoRIIOParallelDecodeWaitTimeout = 300 * time.Millisecond
		})
		DeferCleanup(func() { close(stop) })

		status, _, body, err := env.send(8 * time.Second)
		Expect(err).ToNot(HaveOccurred())
		Expect(status).To(Equal(http.StatusGatewayTimeout))
		Expect(body).To(ContainSubstring("KV-wait timeout"))
	})
})

var _ = Describe("deferredCommitWriter responseStarted", func() {

	It("reports the response as started only once decode's output is relayed", func() {
		dcw := newDeferredCommitWriter(httptest.NewRecorder())
		Expect(dcw.responseStarted()).To(BeFalse())

		Expect(dcw.commit()).To(BeTrue())
		// Committed, but decode has emitted nothing, so the commit is deferred
		// and the response is still the caller's to write.
		Expect(dcw.responseStarted()).To(BeFalse())

		dcw.WriteHeader(http.StatusOK)
		Expect(dcw.responseStarted()).To(BeTrue())
	})

	It("reports no response started after an abort", func() {
		dcw := newDeferredCommitWriter(httptest.NewRecorder())
		_, err := dcw.Write([]byte("partial"))
		Expect(err).ToNot(HaveOccurred())
		dcw.abort()
		Expect(dcw.responseStarted()).To(BeFalse())
		Expect(dcw.commit()).To(BeFalse())
	})
})
