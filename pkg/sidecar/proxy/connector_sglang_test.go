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
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	. "github.com/onsi/ginkgo/v2" //nolint:revive
	. "github.com/onsi/gomega"    //nolint:revive
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
	"google.golang.org/protobuf/proto"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpctest"
	sglangapiv1 "github.com/llm-d/llm-d-router/pkg/common/grpc/sglang/gen/apiv1"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
)

var _ = Describe("SGLang Connector", func() {

	var testInfo *sidecarTestInfo

	BeforeEach(func() {
		// Mock testing setup using the SGLang connector mode
		testInfo = sidecarConnectionTestSetup(constants.KVConnectorSGLang)
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

		By("sending a /v1/chat/completions request with prefill header")
		body := `{
				"model": "Qwen/Qwen2-0.5B",
				"messages": [
				  {"role": "user", "content": "Hello"}
				],
				"max_tokens": 50
			}`

		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+reqcommon.PathChatCompletions, bytes.NewReader([]byte(body)))
		Expect(err).ToNot(HaveOccurred())

		prefillHostPort := testInfo.prefillBackend.URL[len("http://"):]
		req.Header.Add(routing.PrefillEndpointHeader, prefillHostPort)

		rp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())

		if rp.StatusCode != 200 {
			bp, _ := io.ReadAll(rp.Body)
			Fail(string(bp))
		}

		// Because SGLang connector sends requests concurrently (prefill in goroutine),
		// wait until the prefill handler has finished processing before reading its
		// state. The wait polls the recorded requests rather than RequestCount: the
		// mock counts a request on entry and records it after reading the body, so
		// the counter reaches 1 while the slice is still empty.
		Eventually(func() int { return len(testInfo.prefillHandler.GetCompletionRequests()) }).Should(Equal(1))

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
		Expect(prq1).To(HaveKey(reqcommon.FieldBootstrapHost))
		Expect(prq1).To(HaveKey(reqcommon.FieldBootstrapPort))
		Expect(prq1).To(HaveKey(reqcommon.FieldBootstrapRoom))

		expectedHost := extractHost(prefillHostPort)
		Expect(prq1[reqcommon.FieldBootstrapHost]).To(Equal(expectedHost))
		Expect(prq1[reqcommon.FieldBootstrapPort]).To(Equal(float64(sglangBootstrapPort)))
		Expect(prq1[reqcommon.FieldBootstrapRoom]).ToNot(BeNil())

		// Bootstrap validations for decode
		Expect(drq1).To(HaveKey(reqcommon.FieldBootstrapHost))
		Expect(drq1).To(HaveKey(reqcommon.FieldBootstrapPort))
		Expect(drq1).To(HaveKey(reqcommon.FieldBootstrapRoom))

		Expect(drq1[reqcommon.FieldBootstrapHost]).To(Equal(expectedHost))
		Expect(drq1[reqcommon.FieldBootstrapPort]).To(Equal(float64(sglangBootstrapPort)))
		Expect(drq1[reqcommon.FieldBootstrapRoom]).To(Equal(prq1[reqcommon.FieldBootstrapRoom])) // Room ID must match

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
			KVConnector: constants.KVConnectorSGLang,
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

		body := `{"model": "Qwen", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}`
		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+reqcommon.PathChatCompletions, bytes.NewReader([]byte(body)))
		Expect(err).ToNot(HaveOccurred())

		prefillHostPort := testInfo.prefillBackend.URL[len("http://"):]
		req.Header.Add(routing.PrefillEndpointHeader, prefillHostPort)

		// Submit request. This will complete as soon as fastDecode completes.
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

	It("should coordinate native gRPC SglangService/Generate P/D over h2c with bootstrap fields", func() {
		testInfo.decodeBackend.Close()
		testInfo.prefillBackend.Close()

		prefillBackend := newMockSGLangGRPCServer([]string{""})
		DeferCleanup(prefillBackend.Stop)
		decodeBackend := newMockSGLangGRPCServer([]string{"Hi", "Hi world"})
		DeferCleanup(decodeBackend.Stop)

		decodeURL, err := url.Parse("http://" + decodeBackend.addr)
		Expect(err).ToNot(HaveOccurred())

		testInfo.proxy = NewProxy(Config{
			Port:        "0",
			DecoderURL:  decodeURL,
			KVConnector: constants.KVConnectorSGLang,
		})
		go func() {
			defer GinkgoRecover()
			testInfo.proxy.allowlistValidator = &AllowlistValidator{enabled: false}
			Expect(testInfo.proxy.Start(testInfo.ctx)).To(Succeed())
			testInfo.stoppedCh <- struct{}{}
		}()
		<-testInfo.proxy.readyCh
		DeferCleanup(func() {
			testInfo.cancelFn()
			<-testInfo.stoppedCh
		})

		conn, err := grpc.NewClient(testInfo.proxy.addr.String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
		Expect(err).ToNot(HaveOccurred())
		DeferCleanup(func() { _ = conn.Close() })

		client := sglangapiv1.NewSglangServiceClient(conn)
		callCtx := metadata.AppendToOutgoingContext(testInfo.ctx, routing.PrefillEndpointHeader, prefillBackend.addr)
		stream, err := client.Generate(callCtx, &sglangapiv1.GenerateRequest{
			Text: &sglangapiv1.StringOrList{
				Value: &sglangapiv1.StringOrList_One{One: "Hello from gRPC"},
			},
			Stream: proto.Bool(true),
		})
		Expect(err).ToNot(HaveOccurred())

		var gotTexts []string
		for {
			item, recvErr := stream.Recv()
			if errors.Is(recvErr, io.EOF) {
				break
			}
			Expect(recvErr).ToNot(HaveOccurred())
			gotTexts = append(gotTexts, item.GetFrame().GetText())
		}
		Expect(gotTexts).To(Equal([]string{"Hi", "Hi world"}))

		Eventually(prefillBackend.RequestCount).Should(Equal(1))
		Expect(decodeBackend.RequestCount()).To(Equal(1))

		prefillReq := prefillBackend.LastRequest()
		decodeReq := decodeBackend.LastRequest()
		Expect(prefillReq.GetText().GetOne()).To(Equal("Hello from gRPC"))
		Expect(decodeReq.GetText().GetOne()).To(Equal("Hello from gRPC"))

		expectedHost := extractHost(prefillBackend.addr)
		Expect(prefillReq.GetBootstrapHost().GetOne().GetValue()).To(Equal(expectedHost))
		Expect(decodeReq.GetBootstrapHost().GetOne().GetValue()).To(Equal(expectedHost))
		Expect(prefillReq.GetBootstrapPort().GetOne().GetValue()).To(Equal(int64(sglangBootstrapPort)))
		Expect(decodeReq.GetBootstrapPort().GetOne().GetValue()).To(Equal(int64(sglangBootstrapPort)))

		roomID := prefillReq.GetBootstrapRoom().GetOne().GetValue()
		Expect(roomID).To(BeNumerically(">", 0))
		Expect(decodeReq.GetBootstrapRoom().GetOne().GetValue()).To(Equal(roomID))
	})

	It("should pass through SglangService/Generate over h2c when prefill header is absent", func() {
		testInfo.decodeBackend.Close()
		testInfo.prefillBackend.Close()

		decodeBackend := newMockSGLangGRPCServer([]string{"direct decode"})
		DeferCleanup(decodeBackend.Stop)

		decodeURL, err := url.Parse("http://" + decodeBackend.addr)
		Expect(err).ToNot(HaveOccurred())

		testInfo.proxy = NewProxy(Config{
			Port:        "0",
			DecoderURL:  decodeURL,
			KVConnector: constants.KVConnectorSGLang,
		})
		go func() {
			defer GinkgoRecover()
			testInfo.proxy.allowlistValidator = &AllowlistValidator{enabled: false}
			Expect(testInfo.proxy.Start(testInfo.ctx)).To(Succeed())
			testInfo.stoppedCh <- struct{}{}
		}()
		<-testInfo.proxy.readyCh
		DeferCleanup(func() {
			testInfo.cancelFn()
			<-testInfo.stoppedCh
		})

		conn, err := grpc.NewClient(testInfo.proxy.addr.String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
		Expect(err).ToNot(HaveOccurred())
		DeferCleanup(func() { _ = conn.Close() })

		client := sglangapiv1.NewSglangServiceClient(conn)
		stream, err := client.Generate(testInfo.ctx, &sglangapiv1.GenerateRequest{
			Text: &sglangapiv1.StringOrList{
				Value: &sglangapiv1.StringOrList_One{One: "Passthrough"},
			},
		})
		Expect(err).ToNot(HaveOccurred())

		item, err := stream.Recv()
		Expect(err).ToNot(HaveOccurred())
		Expect(item.GetFrame().GetText()).To(Equal("direct decode"))
		_, err = stream.Recv()
		Expect(err).To(MatchError(io.EOF))

		gotReq := decodeBackend.LastRequest()
		Expect(gotReq.GetBootstrapHost()).To(BeNil())
		Expect(gotReq.GetBootstrapRoom()).To(BeNil())
	})

	// startGRPCProxy replaces the HTTP mock backends with cfg and starts the
	// sidecar. It stops the sidecar when the spec ends.
	startGRPCProxy := func(cfg Config) {
		testInfo.decodeBackend.Close()
		testInfo.prefillBackend.Close()
		cfg.Port = "0"
		testInfo.proxy = NewProxy(cfg)
		go func() {
			defer GinkgoRecover()
			testInfo.proxy.allowlistValidator = &AllowlistValidator{enabled: false}
			Expect(testInfo.proxy.Start(testInfo.ctx)).To(Succeed())
			testInfo.stoppedCh <- struct{}{}
		}()
		<-testInfo.proxy.readyCh
		DeferCleanup(func() {
			testInfo.cancelFn()
			<-testInfo.stoppedCh
		})
	}

	// generateTexts sends req through the sidecar with the given gRPC
	// metadata and returns the frame texts of the stream.
	generateTexts := func(req *sglangapiv1.GenerateRequest, md ...string) []string {
		conn, err := grpc.NewClient(testInfo.proxy.addr.String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
		Expect(err).ToNot(HaveOccurred())
		DeferCleanup(func() { _ = conn.Close() })
		stream, err := sglangapiv1.NewSglangServiceClient(conn).Generate(metadata.AppendToOutgoingContext(testInfo.ctx, md...), req)
		Expect(err).ToNot(HaveOccurred())
		var texts []string
		for {
			item, recvErr := stream.Recv()
			if errors.Is(recvErr, io.EOF) {
				return texts
			}
			Expect(recvErr).ToNot(HaveOccurred())
			texts = append(texts, item.GetFrame().GetText())
		}
	}

	decoderURLOf := func(m *mockSGLangGRPCServer) *url.URL {
		u, err := url.Parse("http://" + m.addr)
		Expect(err).ToNot(HaveOccurred())
		return u
	}

	It("should replace client bootstrap fields with the sidecar's values", func() {
		prefillBackend := newMockSGLangGRPCServer([]string{""})
		DeferCleanup(prefillBackend.Stop)
		decodeBackend := newMockSGLangGRPCServer([]string{"ok"})
		DeferCleanup(decodeBackend.Stop)
		startGRPCProxy(Config{DecoderURL: decoderURLOf(decodeBackend), KVConnector: constants.KVConnectorSGLang})

		Expect(generateTexts(&sglangapiv1.GenerateRequest{
			Text: &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "hi"}},
			BootstrapHost: &sglangapiv1.OptionalStringOrList{Value: &sglangapiv1.OptionalStringOrList_Many{
				Many: &sglangapiv1.OptionalStringList{Items: []*sglangapiv1.OptionalString{{Value: proto.String("attacker.example")}}},
			}},
			BootstrapPort: &sglangapiv1.OptionalInt64OrList{Value: &sglangapiv1.OptionalInt64OrList_One{One: &sglangapiv1.OptionalInt64{Value: proto.Int64(1)}}},
			BootstrapRoom: &sglangapiv1.OptionalInt64OrList{Value: &sglangapiv1.OptionalInt64OrList_One{One: &sglangapiv1.OptionalInt64{Value: proto.Int64(42)}}},
		}, routing.PrefillEndpointHeader, prefillBackend.addr)).To(Equal([]string{"ok"}))

		Eventually(prefillBackend.RequestCount).Should(Equal(1))
		expectedHost := extractHost(prefillBackend.addr)
		for _, got := range []*sglangapiv1.GenerateRequest{prefillBackend.LastRequest(), decodeBackend.LastRequest()} {
			Expect(got.GetBootstrapHost().GetMany()).To(BeNil())
			Expect(got.GetBootstrapHost().GetOne().GetValue()).To(Equal(expectedHost))
			Expect(got.GetBootstrapPort().GetOne().GetValue()).To(Equal(int64(sglangBootstrapPort)))
			Expect(got.GetBootstrapRoom().GetOne().GetValue()).ToNot(Equal(int64(42)))
		}
		Expect(decodeBackend.LastRequest().GetBootstrapRoom().GetOne().GetValue()).
			To(Equal(prefillBackend.LastRequest().GetBootstrapRoom().GetOne().GetValue()))
	})

	It("should run P/D instead of the EC connector for a gRPC request with encoder headers", func() {
		prefillBackend := newMockSGLangGRPCServer([]string{""})
		DeferCleanup(prefillBackend.Stop)
		decodeBackend := newMockSGLangGRPCServer([]string{"ok"})
		DeferCleanup(decodeBackend.Stop)
		startGRPCProxy(Config{
			DecoderURL:  decoderURLOf(decodeBackend),
			KVConnector: constants.KVConnectorSGLang,
			ECConnector: constants.ECConnectorNIXL,
		})

		Expect(generateTexts(&sglangapiv1.GenerateRequest{
			Text: &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "hi"}},
		}, routing.PrefillEndpointHeader, prefillBackend.addr, routing.EncoderEndpointsHeader, "127.0.0.1:1")).To(Equal([]string{"ok"}))

		Eventually(prefillBackend.RequestCount).Should(Equal(1))
		Expect(decodeBackend.LastRequest().GetBootstrapRoom().GetOne().GetValue()).
			To(Equal(prefillBackend.LastRequest().GetBootstrapRoom().GetOne().GetValue()))
	})

	It("should forward SglangService/Generate decode-only when kv-connector is not sglang", func() {
		prefillBackend := newMockSGLangGRPCServer([]string{""})
		DeferCleanup(prefillBackend.Stop)
		decodeBackend := newMockSGLangGRPCServer([]string{"direct decode"})
		DeferCleanup(decodeBackend.Stop)
		startGRPCProxy(Config{DecoderURL: decoderURLOf(decodeBackend), KVConnector: constants.KVConnectorNIXLV2})

		Expect(generateTexts(&sglangapiv1.GenerateRequest{
			Text: &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "hi"}},
		}, routing.PrefillEndpointHeader, prefillBackend.addr)).To(Equal([]string{"direct decode"}))

		Expect(prefillBackend.RequestCount()).To(Equal(0))
		Expect(decodeBackend.LastRequest().GetBootstrapRoom()).To(BeNil())
	})

	compressedFrame := []byte{0x01, 0, 0, 0, 0}

	DescribeTable("should reject a bad gRPC request with a gRPC status",
		func(path, contentType string, rawBody []byte, wantCode codes.Code) {
			prefillBackend := newMockSGLangGRPCServer(nil)
			DeferCleanup(prefillBackend.Stop)
			decodeBackend := newMockSGLangGRPCServer(nil)
			DeferCleanup(decodeBackend.Stop)
			startGRPCProxy(Config{DecoderURL: decoderURLOf(decodeBackend), KVConnector: constants.KVConnectorSGLang})

			req, err := http.NewRequestWithContext(testInfo.ctx, http.MethodPost, "http://"+testInfo.proxy.addr.String()+path, bytes.NewReader(rawBody))
			Expect(err).ToNot(HaveOccurred())
			if contentType != "" {
				req.Header.Set(reqcommon.HeaderContentType, contentType)
			}
			req.Header.Set(routing.PrefillEndpointHeader, prefillBackend.addr)

			resp, err := grpctest.NewH2CClient().Do(req)
			Expect(err).ToNot(HaveOccurred())
			defer resp.Body.Close()
			_, err = io.ReadAll(resp.Body)
			Expect(err).ToNot(HaveOccurred())

			Expect(resp.StatusCode).To(Equal(http.StatusOK))
			status, ok := grpcframe.Status(resp.Header)
			if !ok {
				status, ok = grpcframe.Status(resp.Trailer)
			}
			Expect(ok).To(BeTrue())
			Expect(status).To(Equal(strconv.FormatUint(uint64(wantCode), 10)))
			Expect(prefillBackend.RequestCount()).To(Equal(0))
			Expect(decodeBackend.RequestCount()).To(Equal(0))
		},
		Entry("truncated frame header", sglangapiv1.SglangService_Generate_FullMethodName, grpcframe.MediaType,
			[]byte{0x00, 0x01}, codes.InvalidArgument),
		Entry("header declares more bytes than sent", sglangapiv1.SglangService_Generate_FullMethodName, grpcframe.MediaType,
			[]byte{0, 0, 0, 0, 10, 'a'}, codes.InvalidArgument),
		Entry("compressed frame", sglangapiv1.SglangService_Generate_FullMethodName, grpcframe.MediaType,
			compressedFrame, codes.Unimplemented),
		Entry("corrupt protobuf payload", sglangapiv1.SglangService_Generate_FullMethodName, grpcframe.MediaType,
			func() []byte {
				framed, err := grpcframe.FramePayload([]byte{0xff, 0xff, 0xff})
				Expect(err).ToNot(HaveOccurred())
				return framed
			}(), codes.InvalidArgument),
		Entry("trailing bytes after first frame", sglangapiv1.SglangService_Generate_FullMethodName, grpcframe.MediaType,
			func() []byte {
				framed, err := grpcframe.MarshalProto(&sglangapiv1.GenerateRequest{})
				Expect(err).ToNot(HaveOccurred())
				return append(framed, framed...)
			}(), codes.InvalidArgument),
		// The next two entries pin each half of the gRPC check in handleSGLang:
		// the JSON path would answer HTTP 400 without a grpc-status.
		Entry("Generate path without a gRPC content type", sglangapiv1.SglangService_Generate_FullMethodName, "",
			compressedFrame, codes.Unimplemented),
		Entry("JSON generate path with a gRPC content type", reqcommon.PathSGLangGenerate, grpcframe.MediaType,
			compressedFrame, codes.Unimplemented),
	)
})

type mockSGLangGRPCServer struct {
	sglangapiv1.UnimplementedSglangServiceServer
	addr      string
	grpcSrv   *grpc.Server
	responses []string

	mu   sync.Mutex
	reqs []*sglangapiv1.GenerateRequest
}

func newMockSGLangGRPCServer(responses []string) *mockSGLangGRPCServer {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	Expect(err).ToNot(HaveOccurred())

	m := &mockSGLangGRPCServer{
		addr:      ln.Addr().String(),
		grpcSrv:   grpc.NewServer(),
		responses: responses,
	}
	sglangapiv1.RegisterSglangServiceServer(m.grpcSrv, m)
	go func() {
		_ = m.grpcSrv.Serve(ln)
	}()
	return m
}

func (m *mockSGLangGRPCServer) Stop() {
	m.grpcSrv.Stop()
}

func (m *mockSGLangGRPCServer) RequestCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.reqs)
}

func (m *mockSGLangGRPCServer) LastRequest() *sglangapiv1.GenerateRequest {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.reqs) == 0 {
		return nil
	}
	return m.reqs[len(m.reqs)-1]
}

func (m *mockSGLangGRPCServer) Generate(req *sglangapiv1.GenerateRequest, stream grpc.ServerStreamingServer[sglangapiv1.GenerateStreamItem]) error {
	m.mu.Lock()
	m.reqs = append(m.reqs, proto.Clone(req).(*sglangapiv1.GenerateRequest))
	m.mu.Unlock()

	for _, text := range m.responses {
		if err := stream.Send(&sglangapiv1.GenerateStreamItem{
			Item: &sglangapiv1.GenerateStreamItem_Frame{
				Frame: &sglangapiv1.GenerateResponse{
					Text: text,
				},
			},
		}); err != nil {
			return err
		}
	}
	return nil
}

func (m *mockSGLangGRPCServer) HealthCheck(_ context.Context, _ *sglangapiv1.HealthCheckRequest) (*sglangapiv1.HealthCheckResponse, error) {
	return &sglangapiv1.HealthCheckResponse{Healthy: true}, nil
}
