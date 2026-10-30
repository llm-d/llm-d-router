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
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"

	. "github.com/onsi/ginkgo/v2" //nolint:revive
	. "github.com/onsi/gomega"    //nolint:revive
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

var _ = Describe("Attribution handoff", func() {
	DescribeTable("every route attributes its server span and strips the handoff before forwarding",
		func(path string) {
			recorder := tracetest.NewSpanRecorder()
			provider := sdktrace.NewTracerProvider(
				sdktrace.WithSampler(sdktrace.AlwaysSample()),
				sdktrace.WithSpanProcessor(tracing.NewRequestAttributionProcessor()),
				sdktrace.WithSpanProcessor(recorder),
			)
			prev := otel.GetTracerProvider()
			otel.SetTracerProvider(provider)
			DeferCleanup(func() {
				otel.SetTracerProvider(prev)
				_ = provider.Shutdown(context.Background())
			})

			var mu sync.Mutex
			var leaked []string
			decodeBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				mu.Lock()
				defer mu.Unlock()
				for _, h := range []string{tracing.FairnessIDHandoffHeader, tracing.FairnessSourceHandoffHeader} {
					if r.Header.Get(h) != "" {
						leaked = append(leaked, h)
					}
				}
				w.WriteHeader(http.StatusOK)
			}))
			DeferCleanup(decodeBackend.Close)
			targetURL, err := url.Parse(decodeBackend.URL)
			Expect(err).ToNot(HaveOccurred())

			proxy := NewProxy(Config{Port: "0", DecoderURL: targetURL})
			ctx, cancelFn := context.WithCancel(newTestContext())
			stoppedCh := make(chan struct{})
			go func() {
				defer GinkgoRecover()
				proxy.allowlistValidator = &AllowlistValidator{enabled: false}
				Expect(proxy.Start(ctx)).To(Succeed())
				close(stoppedCh)
			}()
			DeferCleanup(func() {
				cancelFn()
				<-stoppedCh
			})
			<-proxy.readyCh

			req, err := http.NewRequest(http.MethodPost, "http://"+proxy.addr.String()+path,
				strings.NewReader(`{"model":"m","prompt":"p"}`))
			Expect(err).ToNot(HaveOccurred())
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set(tracing.FairnessIDHandoffHeader, "team-a")
			req.Header.Set(tracing.FairnessSourceHandoffHeader, tracing.AttributionSourceHeader)
			resp, err := http.DefaultClient.Do(req)
			Expect(err).ToNot(HaveOccurred())
			_, _ = io.Copy(io.Discard, resp.Body)
			Expect(resp.Body.Close()).To(Succeed())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			mu.Lock()
			Expect(leaked).To(BeEmpty(), "handoff headers must not reach the model server")
			mu.Unlock()

			Expect(provider.ForceFlush(context.Background())).To(Succeed())
			var server sdktrace.ReadOnlySpan
			for _, span := range recorder.Ended() {
				if span.Name() == http.MethodPost+" "+path {
					server = span
				}
			}
			Expect(server).ToNot(BeNil(), "server span for %s", path)
			attrs := attribute.NewSet(server.Attributes()...)
			id, _ := attrs.Value(semconv.LLMDEPPFairnessIDKey)
			source, _ := attrs.Value(semconv.LLMDEPPFairnessSourceKey)
			Expect(id.AsString()).To(Equal("team-a"))
			Expect(source.AsString()).To(Equal(tracing.AttributionSourceHeader))
		},
		Entry("chat completions", reqcommon.PathChatCompletions),
		Entry("completions", reqcommon.PathCompletions),
		Entry("messages", reqcommon.PathMessages),
		Entry("responses", reqcommon.PathResponses),
		Entry("vLLM generate", reqcommon.PathVLLMGenerate),
		Entry("SGLang generate", reqcommon.PathSGLangGenerate),
		Entry("trailing slash", reqcommon.PathCompletions+"/"),
		Entry("catch-all decoder proxy", "/v1/embeddings"),
	)
})
