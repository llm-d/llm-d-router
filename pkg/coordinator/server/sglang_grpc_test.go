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

package server_test

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	sglangapiv1 "github.com/llm-d/llm-d-router/pkg/common/grpc/sglang/gen/apiv1"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/kv"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
	"github.com/llm-d/llm-d-router/pkg/coordinator/server"
	"github.com/llm-d/llm-d-router/pkg/coordinator/steps"
)

func startH2CTestGateway(t *testing.T, handler http.Handler) *httptest.Server {
	t.Helper()
	srv := httptest.NewUnstartedServer(handler)
	protocols := new(http.Protocols)
	protocols.SetHTTP1(true)
	protocols.SetUnencryptedHTTP2(true)
	srv.Config.Protocols = protocols
	srv.Start()
	t.Cleanup(srv.Close)
	return srv
}

func startCoordinatorH2C(t *testing.T, cfg config.ServerConfig, p *pipeline.Pipeline, gw *gateway.Client) string {
	t.Helper()
	cfg.SecureServing = false
	srv, err := server.New(cfg, p, gw)
	if err != nil {
		t.Fatalf("server.New: %v", err)
	}
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		defer close(done)
		_ = srv.Serve(ctx, ln)
	}()
	t.Cleanup(func() {
		cancel()
		shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer shutdownCancel()
		_ = srv.Shutdown(shutdownCtx)
		<-done
	})
	return ln.Addr().String()
}

func newSGLangPDPipeline(t *testing.T, gw *gateway.Client) *pipeline.Pipeline {
	t.Helper()
	prefill, err := steps.NewPrefillStep(gw, map[string]any{"kv_connector": kv.SGLang})
	if err != nil {
		t.Fatalf("NewPrefillStep: %v", err)
	}
	decode, err := steps.NewDecodeStep(gw, map[string]any{"kv_connector": kv.SGLang})
	if err != nil {
		t.Fatalf("NewDecodeStep: %v", err)
	}
	p, err := pipeline.NewWithForwardResponseHeaders([]pipeline.Step{prefill, decode}, []string{reqcommon.DisaggregatedRevisionHeaderKey})
	if err != nil {
		t.Fatalf("NewWithForwardResponseHeaders: %v", err)
	}
	return p
}

func TestSGLangGRPC_ConcurrentPrefillDecodeStreamsAndRewritesCachedTokens(t *testing.T) {
	rendezvous := make(chan struct{})
	var (
		mu            sync.Mutex
		prefillProto  int
		decodeProto   int
		prefillGenReq *sglangapiv1.GenerateRequest
		decodeGenReq  *sglangapiv1.GenerateRequest
		decodeHeaders http.Header
	)

	gw := startH2CTestGateway(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read gateway request: %v", err)
			return
		}
		payload, err := grpcframe.ParsePayload(raw)
		if err != nil {
			t.Errorf("parse gateway gRPC payload: %v", err)
			return
		}
		var genReq sglangapiv1.GenerateRequest
		if err := proto.Unmarshal(payload, &genReq); err != nil {
			t.Errorf("unmarshal GenerateRequest: %v", err)
			return
		}

		profile := r.Header.Get(reqcommon.EPPProfileHeaderKey)
		switch profile {
		case gateway.PhasePrefill:
			mu.Lock()
			prefillProto = r.ProtoMajor
			prefillGenReq = proto.Clone(&genReq).(*sglangapiv1.GenerateRequest)
			mu.Unlock()

			w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
			w.Header().Set(routing.PrefillEndpointHeader, "10.2.3.4:30000")
			w.Header().Set(reqcommon.PeerTopologyHeaderKey, "host=node12,zone=us-east1-a")
			w.Header().Set(reqcommon.DisaggregatedRevisionHeaderKey, "rev-7")
			w.WriteHeader(http.StatusOK)
			_ = http.NewResponseController(w).Flush()

			// Block until decode reaches the gateway, proving prefill and decode
			// execute concurrently rather than sequentially.
			select {
			case <-rendezvous:
			case <-r.Context().Done():
				return
			}

			frameBytes, err := grpcframe.MarshalProto(&sglangapiv1.GenerateStreamItem{
				Item: &sglangapiv1.GenerateStreamItem_Frame{
					Frame: &sglangapiv1.GenerateResponse{
						MetaInfo: &sglangapiv1.GenerateMetaInfo{
							PromptTokens: 64,
							CachedTokens: 48,
							FinishReason: &sglangapiv1.FinishReason{
								Kind: &sglangapiv1.FinishReason_Length{
									Length: &sglangapiv1.FinishLength{Length: proto.Uint64(1)},
								},
							},
						},
					},
				},
			})
			if err != nil {
				t.Errorf("marshal prefill frame: %v", err)
				return
			}
			_, _ = w.Write(frameBytes)
			w.Header().Set(http.TrailerPrefix+grpcframe.HeaderStatus, grpcframe.StatusOK)

		case gateway.PhaseDecode:
			mu.Lock()
			decodeProto = r.ProtoMajor
			decodeGenReq = proto.Clone(&genReq).(*sglangapiv1.GenerateRequest)
			decodeHeaders = r.Header.Clone()
			mu.Unlock()

			close(rendezvous)

			w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
			w.Header().Set(reqcommon.PeerTopologyHeaderKey, "host=leak-check")
			w.Header().Set(routing.PrefillEndpointHeader, "10.9.9.9:30000")
			w.WriteHeader(http.StatusOK)

			firstFrame, err := grpcframe.MarshalProto(&sglangapiv1.GenerateStreamItem{
				Item: &sglangapiv1.GenerateStreamItem_Frame{
					Frame: &sglangapiv1.GenerateResponse{
						Text: "Hi",
						MetaInfo: &sglangapiv1.GenerateMetaInfo{
							PromptTokens:     64,
							CompletionTokens: 1,
							CachedTokens:     0,
						},
					},
				},
			})
			if err != nil {
				t.Errorf("marshal first decode frame: %v", err)
				return
			}
			_, _ = w.Write(firstFrame)
			_ = http.NewResponseController(w).Flush()

			finalFrame, err := grpcframe.MarshalProto(&sglangapiv1.GenerateStreamItem{
				Item: &sglangapiv1.GenerateStreamItem_Frame{
					Frame: &sglangapiv1.GenerateResponse{
						Text: "Hi there",
						MetaInfo: &sglangapiv1.GenerateMetaInfo{
							PromptTokens:     64,
							CompletionTokens: 2,
							CachedTokens:     0,
							FinishReason: &sglangapiv1.FinishReason{
								Kind: &sglangapiv1.FinishReason_Stop{Stop: &sglangapiv1.FinishStop{}},
							},
						},
					},
				},
			})
			if err != nil {
				t.Errorf("marshal final decode frame: %v", err)
				return
			}
			_, _ = w.Write(finalFrame)
			w.Header().Set(http.TrailerPrefix+grpcframe.HeaderStatus, grpcframe.StatusOK)

		default:
			t.Errorf("unexpected profile %q", profile)
			w.WriteHeader(http.StatusBadRequest)
		}
	}))

	gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
	addr := startCoordinatorH2C(t, config.ServerConfig{}, newSGLangPDPipeline(t, gwClient), gwClient)

	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("grpc.NewClient: %v", err)
	}
	defer conn.Close()

	client := sglangapiv1.NewSglangServiceClient(conn)
	callCtx := metadata.NewOutgoingContext(context.Background(), metadata.Pairs(
		routing.PrefillEndpointHeader, "spoofed:9999",
		reqcommon.PeerTopologyHeaderKey, "host=spoofed",
	))
	var respHeaders metadata.MD
	stream, err := client.Generate(callCtx, &sglangapiv1.GenerateRequest{
		Text: &sglangapiv1.StringOrList{
			Value: &sglangapiv1.StringOrList_One{One: "Hello Coordinator"},
		},
		Stream: proto.Bool(true),
	}, grpc.Header(&respHeaders))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}

	var items []*sglangapiv1.GenerateResponse
	for {
		item, recvErr := stream.Recv()
		if errors.Is(recvErr, io.EOF) {
			break
		}
		if recvErr != nil {
			t.Fatalf("stream.Recv: %v", recvErr)
		}
		items = append(items, item.GetFrame())
	}
	if len(items) != 2 {
		t.Fatalf("got %d frames, want 2", len(items))
	}
	if items[0].GetText() != "Hi" || items[1].GetText() != "Hi there" {
		t.Fatalf("unexpected texts: %q, %q", items[0].GetText(), items[1].GetText())
	}
	if got := items[1].GetMetaInfo().GetCachedTokens(); got != 48 {
		t.Fatalf("terminal cached_tokens = %d, want 48 from prefill", got)
	}

	if got := respHeaders.Get(reqcommon.PeerTopologyHeaderKey); len(got) != 0 {
		t.Fatalf("peer topology leaked to client: %v", got)
	}
	if got := respHeaders.Get(routing.PrefillEndpointHeader); len(got) != 0 {
		t.Fatalf("prefill endpoint leaked to client: %v", got)
	}

	mu.Lock()
	defer mu.Unlock()
	if prefillProto != 2 || decodeProto != 2 {
		t.Fatalf("expected HTTP/2 on prefill (%d) and decode (%d)", prefillProto, decodeProto)
	}
	if prefillGenReq.GetBootstrapHost() != nil {
		t.Fatalf("prefill request should not carry bootstrap_host, got %v", prefillGenReq.GetBootstrapHost())
	}
	room := prefillGenReq.GetBootstrapRoom().GetOne().GetValue()
	port := prefillGenReq.GetBootstrapPort().GetOne().GetValue()
	if room <= 0 || port <= 0 {
		t.Fatalf("expected positive bootstrap room (%d) and port (%d)", room, port)
	}
	if got := decodeGenReq.GetBootstrapHost().GetOne().GetValue(); got != "10.2.3.4" {
		t.Fatalf("decode bootstrap_host = %q, want 10.2.3.4", got)
	}
	if got := decodeGenReq.GetBootstrapRoom().GetOne().GetValue(); got != room {
		t.Fatalf("decode bootstrap_room = %d, want %d", got, room)
	}
	if got := decodeGenReq.GetBootstrapPort().GetOne().GetValue(); got != port {
		t.Fatalf("decode bootstrap_port = %d, want %d", got, port)
	}
	if got := decodeHeaders.Get(reqcommon.PeerTopologyHeaderKey); got != "host=node12,zone=us-east1-a" {
		t.Fatalf("decode peer topology = %q, want host=node12,zone=us-east1-a", got)
	}
	if got := decodeHeaders.Get(reqcommon.DisaggregatedRevisionHeaderKey); got != "rev-7" {
		t.Fatalf("decode disagg revision = %q, want rev-7", got)
	}
}

func TestSGLangGRPC_PrefillStreamErrorCancelsDecodeAndSurfacesGRPCStatus(t *testing.T) {
	decodeStarted := make(chan struct{})
	var decodeCancelled atomic.Bool

	gw := startH2CTestGateway(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Header.Get(reqcommon.EPPProfileHeaderKey) {
		case gateway.PhasePrefill:
			w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
			w.Header().Set(routing.PrefillEndpointHeader, "10.2.3.4:30000")
			w.WriteHeader(http.StatusOK)
			_ = http.NewResponseController(w).Flush()

			<-decodeStarted
			w.Header().Set(http.TrailerPrefix+grpcframe.HeaderStatus, "8")
			w.Header().Set(http.TrailerPrefix+grpcframe.HeaderMessage, "prefill KV exhausted")

		case gateway.PhaseDecode:
			w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
			w.WriteHeader(http.StatusOK)
			_ = http.NewResponseController(w).Flush()
			close(decodeStarted)
			<-r.Context().Done()
			decodeCancelled.Store(true)
		}
	}))

	gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
	addr := startCoordinatorH2C(t, config.ServerConfig{}, newSGLangPDPipeline(t, gwClient), gwClient)

	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("grpc.NewClient: %v", err)
	}
	defer conn.Close()

	client := sglangapiv1.NewSglangServiceClient(conn)
	stream, err := client.Generate(context.Background(), &sglangapiv1.GenerateRequest{
		Text: &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "fail prefill"}},
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	_, recvErr := stream.Recv()
	if status.Code(recvErr) != codes.ResourceExhausted {
		t.Fatalf("Recv code = %v (%v), want %v", status.Code(recvErr), recvErr, codes.ResourceExhausted)
	}

	deadline := time.Now().Add(2 * time.Second)
	for !decodeCancelled.Load() && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if !decodeCancelled.Load() {
		t.Fatal("expected decode upstream context to be cancelled when prefill failed")
	}
}

func TestSGLangGRPC_DecodeOnlyAndControlRPCPassthrough(t *testing.T) {
	backendSrv := grpc.NewServer()
	sglangapiv1.RegisterSglangServiceServer(backendSrv, &stubSGLangServiceServer{})
	backendLn, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen: %v", err)
	}
	go func() { _ = backendSrv.Serve(backendLn) }()
	t.Cleanup(backendSrv.Stop)

	var lastProfile atomic.Value
	gwClient := gateway.New(config.GatewayConfig{Address: "http://" + backendLn.Addr().String()})

	gw := startH2CTestGateway(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		lastProfile.Store(r.Header.Get(reqcommon.EPPProfileHeaderKey))
		body, _ := io.ReadAll(r.Body)
		resp, err := gwClient.StreamGRPC(r.Context(), r.URL.Path, body, nil)
		if err != nil {
			t.Errorf("forward to backend: %v", err)
			return
		}
		defer resp.Body.Close()
		w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
		grpcframe.CopyTrailers(w.Header(), resp.Header)
		w.WriteHeader(resp.StatusCode)
		_, _ = io.Copy(w, resp.Body)
		grpcframe.CopyTrailers(w.Header(), resp.Trailer)
	}))

	coordGW := gateway.New(config.GatewayConfig{Address: gw.URL})
	decodeStep, err := steps.NewDecodeStep(coordGW, map[string]any{"kv_connector": kv.SGLang})
	if err != nil {
		t.Fatalf("NewDecodeStep: %v", err)
	}
	addr := startCoordinatorH2C(t, config.ServerConfig{}, pipeline.New([]pipeline.Step{decodeStep}), coordGW)

	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("grpc.NewClient: %v", err)
	}
	defer conn.Close()
	client := sglangapiv1.NewSglangServiceClient(conn)

	hc, err := client.HealthCheck(context.Background(), &sglangapiv1.HealthCheckRequest{})
	if err != nil || !hc.GetHealthy() {
		t.Fatalf("HealthCheck = (%v, %v), want healthy", hc, err)
	}
	if got, _ := lastProfile.Load().(string); got != gateway.PhaseDecode {
		t.Fatalf("HealthCheck profile = %q, want %q", got, gateway.PhaseDecode)
	}

	stream, err := client.Generate(context.Background(), &sglangapiv1.GenerateRequest{
		Text: &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "decode only"}},
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	item, err := stream.Recv()
	if err != nil {
		t.Fatalf("stream.Recv: %v", err)
	}
	if item.GetFrame().GetText() != "stub: decode only" {
		t.Fatalf("text = %q, want %q", item.GetFrame().GetText(), "stub: decode only")
	}
	if _, err := stream.Recv(); !errors.Is(err, io.EOF) {
		t.Fatalf("expected EOF, got %v", err)
	}
}

func TestSGLangGRPC_RejectsOversizedAndMalformedBodies(t *testing.T) {
	gw := startH2CTestGateway(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		t.Error("gateway must not be called for rejected requests")
		w.WriteHeader(http.StatusInternalServerError)
	}))
	gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
	addr := startCoordinatorH2C(t, config.ServerConfig{MaxRequestBodySize: 1}, newSGLangPDPipeline(t, gwClient), gwClient)

	protocols := new(http.Protocols)
	protocols.SetUnencryptedHTTP2(true)
	h2cClient := &http.Client{Transport: &http.Transport{Protocols: protocols}}

	sendRaw := func(path string, body []byte) string {
		t.Helper()
		req, err := http.NewRequest(http.MethodPost, "http://"+addr+path, bytes.NewReader(body))
		if err != nil {
			t.Fatalf("NewRequest: %v", err)
		}
		req.Header.Set(reqcommon.HeaderContentType, grpcframe.MediaType)
		resp, err := h2cClient.Do(req)
		if err != nil {
			t.Fatalf("Do: %v", err)
		}
		defer resp.Body.Close()
		_, _ = io.ReadAll(resp.Body)
		code, ok := grpcframe.Status(resp.Header)
		if !ok {
			code, _ = grpcframe.Status(resp.Trailer)
		}
		return code
	}

	if got := sendRaw(sglangapiv1.SglangService_Generate_FullMethodName, []byte{0x00, 0x01}); got != "3" {
		t.Fatalf("truncated frame status = %q, want 3 (InvalidArgument)", got)
	}
	corruptFrame, err := grpcframe.FramePayload([]byte{0xff, 0xff, 0xff})
	if err != nil {
		t.Fatalf("FramePayload: %v", err)
	}
	if got := sendRaw("/sglang.api.v1.SglangService%2FGenerate", corruptFrame); got != "3" {
		t.Fatalf("percent-encoded corrupt protobuf status = %q, want 3 (InvalidArgument)", got)
	}
	oversized := bytes.Repeat([]byte("x"), config.BytesPerMB+16)
	if got := sendRaw(sglangapiv1.SglangService_Generate_FullMethodName, oversized); got != "8" {
		t.Fatalf("oversized body status = %q, want 8 (ResourceExhausted)", got)
	}
}

func TestSGLangGRPC_JSONOnlyStepsNoOp(t *testing.T) {
	backendSrv := grpc.NewServer()
	sglangapiv1.RegisterSglangServiceServer(backendSrv, &stubSGLangServiceServer{})
	backendLn, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen: %v", err)
	}
	go func() { _ = backendSrv.Serve(backendLn) }()
	t.Cleanup(backendSrv.Stop)

	gwClient := gateway.New(config.GatewayConfig{Address: "http://" + backendLn.Addr().String()})
	gw := startH2CTestGateway(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Prefer") == "if-available" {
			t.Error("conditional-decode must not probe on SGLang gRPC Generate")
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		body, _ := io.ReadAll(r.Body)
		resp, err := gwClient.StreamGRPC(r.Context(), r.URL.Path, body, nil)
		if err != nil {
			t.Errorf("forward to backend: %v", err)
			return
		}
		defer resp.Body.Close()
		w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
		grpcframe.CopyTrailers(w.Header(), resp.Header)
		w.WriteHeader(resp.StatusCode)
		_, _ = io.Copy(w, resp.Body)
		grpcframe.CopyTrailers(w.Header(), resp.Trailer)
	}))

	coordGW := gateway.New(config.GatewayConfig{Address: gw.URL})
	replaceStep, err := steps.NewReplaceMediaURLsStep(coordGW, nil)
	if err != nil {
		t.Fatalf("NewReplaceMediaURLsStep: %v", err)
	}
	renderStep, err := steps.NewRenderStep(coordGW, map[string]any{"service_address": "http://127.0.0.1:1"})
	if err != nil {
		t.Fatalf("NewRenderStep: %v", err)
	}
	condStep, err := steps.NewConditionalDecodeStep(coordGW, nil)
	if err != nil {
		t.Fatalf("NewConditionalDecodeStep: %v", err)
	}
	decodeStep, err := steps.NewDecodeStep(coordGW, map[string]any{"kv_connector": kv.SGLang})
	if err != nil {
		t.Fatalf("NewDecodeStep: %v", err)
	}

	addr := startCoordinatorH2C(t, config.ServerConfig{}, pipeline.New([]pipeline.Step{
		replaceStep,
		renderStep,
		condStep,
		decodeStep,
	}), coordGW)

	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("grpc.NewClient: %v", err)
	}
	defer conn.Close()

	stream, err := sglangapiv1.NewSglangServiceClient(conn).Generate(context.Background(), &sglangapiv1.GenerateRequest{
		Text: &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "skip json steps"}},
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	item, err := stream.Recv()
	if err != nil || item.GetFrame().GetText() != "stub: skip json steps" {
		t.Fatalf("Recv = (%v, %v), want stub: skip json steps", item, err)
	}
}

type stubSGLangServiceServer struct {
	sglangapiv1.UnimplementedSglangServiceServer
}

func (s *stubSGLangServiceServer) HealthCheck(_ context.Context, _ *sglangapiv1.HealthCheckRequest) (*sglangapiv1.HealthCheckResponse, error) {
	return &sglangapiv1.HealthCheckResponse{Healthy: true}, nil
}

func (s *stubSGLangServiceServer) Generate(req *sglangapiv1.GenerateRequest, stream grpc.ServerStreamingServer[sglangapiv1.GenerateStreamItem]) error {
	if req.GetBootstrapHost() != nil || req.GetBootstrapRoom() != nil {
		return status.Error(codes.InvalidArgument, "unexpected bootstrap fields in decode-only pipeline")
	}
	return stream.Send(&sglangapiv1.GenerateStreamItem{
		Item: &sglangapiv1.GenerateStreamItem_Frame{
			Frame: &sglangapiv1.GenerateResponse{
				Text: "stub: " + req.GetText().GetOne(),
			},
		},
	})
}
