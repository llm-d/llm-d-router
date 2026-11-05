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
	"crypto/tls"
	"encoding/base64"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/encoding/gzip"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/protoadapt"

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
	"github.com/llm-d/llm-d-router/pkg/coordinator/steps/asyncbroker"
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
	if got := items[0].GetMetaInfo().GetCachedTokens(); got != 0 {
		t.Fatalf("non-terminal cached_tokens = %d, want 0 from decode", got)
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
		resp, err := gwClient.StreamGRPC(r.Context(), r.URL.Path, nil, body)
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
	// The frame header declares more than the 1 MB limit.
	oversized, err := grpcframe.FramePayload(bytes.Repeat([]byte("x"), config.BytesPerMB+16))
	if err != nil {
		t.Fatalf("FramePayload: %v", err)
	}
	if got := sendRaw(sglangapiv1.SglangService_Generate_FullMethodName, oversized); got != "8" {
		t.Fatalf("oversized body status = %q, want 8 (ResourceExhausted)", got)
	}
	if got := sendRaw(sglangapiv1.SglangService_Generate_FullMethodName, []byte{0x01, 0x00, 0x00, 0x00, 0x00}); got != "12" {
		t.Fatalf("compressed frame flag status = %q, want 12 (Unimplemented)", got)
	}

	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("grpc.NewClient: %v", err)
	}
	defer conn.Close()
	stream, err := sglangapiv1.NewSglangServiceClient(conn).Generate(
		context.Background(),
		&sglangapiv1.GenerateRequest{
			Text: &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "gzip"}},
		},
		grpc.UseCompressor(gzip.Name),
	)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if _, recvErr := stream.Recv(); status.Code(recvErr) != codes.Unimplemented {
		t.Fatalf("gzip compressed request code = %v (%v), want Unimplemented", status.Code(recvErr), recvErr)
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
		resp, err := gwClient.StreamGRPC(r.Context(), r.URL.Path, nil, body)
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
	encodeStep, err := steps.NewEncodeStep(coordGW, nil)
	if err != nil {
		t.Fatalf("NewEncodeStep: %v", err)
	}
	asyncStep, err := asyncbroker.New(coordGW, map[string]any{
		"redis_url":   "redis://" + miniredis.RunT(t).Addr(),
		"wakeup_mode": "poll",
	})
	if err != nil {
		t.Fatalf("asyncbroker.New: %v", err)
	}
	decodeStep, err := steps.NewDecodeStep(coordGW, map[string]any{"kv_connector": kv.SGLang})
	if err != nil {
		t.Fatalf("NewDecodeStep: %v", err)
	}

	addr := startCoordinatorH2C(t, config.ServerConfig{}, pipeline.New([]pipeline.Step{
		asyncStep,
		replaceStep,
		renderStep,
		encodeStep,
		condStep,
		decodeStep,
	}), coordGW)

	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("grpc.NewClient: %v", err)
	}
	defer conn.Close()

	// The async mode header would make async-broker answer with a JSON
	// error if it ran on this request.
	callCtx := metadata.NewOutgoingContext(context.Background(), metadata.Pairs("x-llm-d-async-mode", "enqueue"))
	stream, err := sglangapiv1.NewSglangServiceClient(conn).Generate(callCtx, &sglangapiv1.GenerateRequest{
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

func encodeStatusDetailsBin(t *testing.T, code codes.Code, msg string, detail proto.Message) string {
	t.Helper()
	st, err := status.New(code, msg).WithDetails(protoadapt.MessageV1Of(detail))
	if err != nil {
		t.Fatalf("status.WithDetails: %v", err)
	}
	raw, err := proto.Marshal(st.Proto())
	if err != nil {
		t.Fatalf("proto.Marshal status: %v", err)
	}
	return base64.RawStdEncoding.EncodeToString(raw)
}

func assertErrorBodyDetail(t *testing.T, err error, wantCode uint32, wantMsg string) {
	t.Helper()
	st, ok := status.FromError(err)
	if !ok {
		t.Fatalf("expected gRPC status error, got %v", err)
	}
	details := st.Details()
	if len(details) != 1 {
		t.Fatalf("got %d status details, want 1 (%v)", len(details), details)
	}
	eb, ok := details[0].(*sglangapiv1.ErrorBody)
	if !ok {
		t.Fatalf("detail type = %T, want *sglangapiv1.ErrorBody", details[0])
	}
	if eb.GetCode() != wantCode || eb.GetMessage() != wantMsg {
		t.Fatalf("ErrorBody = (%d, %q), want (%d, %q)", eb.GetCode(), eb.GetMessage(), wantCode, wantMsg)
	}
}

func TestSGLangGRPC_MissingPrefillEndpointHeaderReturnsInternal(t *testing.T) {
	var decodeCalled atomic.Bool
	gw := startH2CTestGateway(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Header.Get(reqcommon.EPPProfileHeaderKey) {
		case gateway.PhasePrefill:
			w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
			w.WriteHeader(http.StatusOK)
			w.Header().Set(http.TrailerPrefix+grpcframe.HeaderStatus, grpcframe.StatusOK)
		case gateway.PhaseDecode:
			decodeCalled.Store(true)
			w.WriteHeader(http.StatusInternalServerError)
		}
	}))

	gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
	addr := startCoordinatorH2C(t, config.ServerConfig{}, newSGLangPDPipeline(t, gwClient), gwClient)

	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("grpc.NewClient: %v", err)
	}
	defer conn.Close()

	stream, err := sglangapiv1.NewSglangServiceClient(conn).Generate(context.Background(), &sglangapiv1.GenerateRequest{
		Text: &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "missing header"}},
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	_, recvErr := stream.Recv()
	if status.Code(recvErr) != codes.Internal {
		t.Fatalf("Recv code = %v (%v), want Internal", status.Code(recvErr), recvErr)
	}
	if decodeCalled.Load() {
		t.Fatal("decode must not be called when prefill omits x-llm-d-prefiller-host-port")
	}
}

func TestSGLangGRPC_TrailersOnlyOKAndErrorsPreserveDetails(t *testing.T) {
	t.Run("decode trailers-only OK returns EOF", func(t *testing.T) {
		gw := startH2CTestGateway(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
			w.Header().Set(grpcframe.HeaderStatus, grpcframe.StatusOK)
			w.WriteHeader(http.StatusOK)
		}))
		gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
		decodeStep, err := steps.NewDecodeStep(gwClient, map[string]any{"kv_connector": kv.SGLang})
		if err != nil {
			t.Fatalf("NewDecodeStep: %v", err)
		}
		addr := startCoordinatorH2C(t, config.ServerConfig{}, pipeline.New([]pipeline.Step{decodeStep}), gwClient)

		conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			t.Fatalf("grpc.NewClient: %v", err)
		}
		defer conn.Close()

		stream, err := sglangapiv1.NewSglangServiceClient(conn).Generate(context.Background(), &sglangapiv1.GenerateRequest{
			Text: &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "empty ok"}},
		})
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		if _, recvErr := stream.Recv(); !errors.Is(recvErr, io.EOF) {
			t.Fatalf("Recv = %v, want io.EOF", recvErr)
		}
	})

	t.Run("prefill trailers-only error preserves ErrorBody details", func(t *testing.T) {
		detailsBin := encodeStatusDetailsBin(t, codes.InvalidArgument, "invalid lora", &sglangapiv1.ErrorBody{
			Code:    http.StatusBadRequest,
			Message: "unknown lora adapter",
		})
		gw := startH2CTestGateway(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Header.Get(reqcommon.EPPProfileHeaderKey) != gateway.PhasePrefill {
				t.Errorf("unexpected phase %q", r.Header.Get(reqcommon.EPPProfileHeaderKey))
			}
			w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
			w.Header().Set(grpcframe.HeaderStatus, strconv.Itoa(int(codes.InvalidArgument)))
			w.Header().Set(grpcframe.HeaderMessage, grpcframe.EncodeStatusMessage("invalid lora"))
			w.Header().Set(grpcframe.HeaderStatusDetailsBin, detailsBin)
			w.WriteHeader(http.StatusOK)
		}))
		gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
		addr := startCoordinatorH2C(t, config.ServerConfig{}, newSGLangPDPipeline(t, gwClient), gwClient)

		conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			t.Fatalf("grpc.NewClient: %v", err)
		}
		defer conn.Close()

		stream, err := sglangapiv1.NewSglangServiceClient(conn).Generate(context.Background(), &sglangapiv1.GenerateRequest{
			Text: &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "bad lora"}},
		})
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		_, recvErr := stream.Recv()
		if status.Code(recvErr) != codes.InvalidArgument {
			t.Fatalf("Recv code = %v (%v), want InvalidArgument", status.Code(recvErr), recvErr)
		}
		assertErrorBodyDetail(t, recvErr, http.StatusBadRequest, "unknown lora adapter")
	})

	t.Run("decode trailers-only error preserves ErrorBody details", func(t *testing.T) {
		detailsBin := encodeStatusDetailsBin(t, codes.ResourceExhausted, "decode full", &sglangapiv1.ErrorBody{
			Code:    http.StatusTooManyRequests,
			Message: "decode queue full",
		})
		gw := startH2CTestGateway(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
			w.Header().Set(grpcframe.HeaderStatus, strconv.Itoa(int(codes.ResourceExhausted)))
			w.Header().Set(grpcframe.HeaderMessage, grpcframe.EncodeStatusMessage("decode full"))
			w.Header().Set(grpcframe.HeaderStatusDetailsBin, detailsBin)
			w.WriteHeader(http.StatusOK)
		}))
		gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
		decodeStep, err := steps.NewDecodeStep(gwClient, map[string]any{"kv_connector": kv.SGLang})
		if err != nil {
			t.Fatalf("NewDecodeStep: %v", err)
		}
		addr := startCoordinatorH2C(t, config.ServerConfig{}, pipeline.New([]pipeline.Step{decodeStep}), gwClient)

		conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			t.Fatalf("grpc.NewClient: %v", err)
		}
		defer conn.Close()

		stream, err := sglangapiv1.NewSglangServiceClient(conn).Generate(context.Background(), &sglangapiv1.GenerateRequest{
			Text: &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "decode err"}},
		})
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		_, recvErr := stream.Recv()
		if status.Code(recvErr) != codes.ResourceExhausted {
			t.Fatalf("Recv code = %v (%v), want ResourceExhausted", status.Code(recvErr), recvErr)
		}
		assertErrorBodyDetail(t, recvErr, http.StatusTooManyRequests, "decode queue full")
	})
}

func TestSGLangGRPC_MissingGRPCStatusReturnsInternal(t *testing.T) {
	t.Run("decode empty 200 without grpc-status", func(t *testing.T) {
		gw := startH2CTestGateway(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
			w.WriteHeader(http.StatusOK)
		}))
		gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
		decodeStep, err := steps.NewDecodeStep(gwClient, map[string]any{"kv_connector": kv.SGLang})
		if err != nil {
			t.Fatalf("NewDecodeStep: %v", err)
		}
		addr := startCoordinatorH2C(t, config.ServerConfig{}, pipeline.New([]pipeline.Step{decodeStep}), gwClient)

		conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			t.Fatalf("grpc.NewClient: %v", err)
		}
		defer conn.Close()

		stream, err := sglangapiv1.NewSglangServiceClient(conn).Generate(context.Background(), &sglangapiv1.GenerateRequest{
			Text: &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "no status"}},
		})
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		if _, recvErr := stream.Recv(); status.Code(recvErr) != codes.Internal {
			t.Fatalf("Recv code = %v (%v), want Internal", status.Code(recvErr), recvErr)
		}
	})

	t.Run("decode stream cut off after a frame header is not a clean end", func(t *testing.T) {
		// The upstream sends one whole frame, then a 5-byte header that
		// declares a second frame, then ends the stream with grpc-status 0.
		// The truncated frame must surface as an error, not be dropped with
		// the OK status forwarded.
		gw := startH2CTestGateway(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
			w.WriteHeader(http.StatusOK)
			frameBytes, err := grpcframe.MarshalProto(&sglangapiv1.GenerateStreamItem{
				Item: &sglangapiv1.GenerateStreamItem_Frame{Frame: &sglangapiv1.GenerateResponse{Text: "chunk-1"}},
			})
			if err != nil {
				t.Errorf("MarshalProto: %v", err)
				return
			}
			_, _ = w.Write(frameBytes)
			_, _ = w.Write([]byte{0, 0, 0, 0, 9})
			w.Header().Set(http.TrailerPrefix+grpcframe.HeaderStatus, grpcframe.StatusOK)
		}))
		gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
		decodeStep, err := steps.NewDecodeStep(gwClient, map[string]any{"kv_connector": kv.SGLang})
		if err != nil {
			t.Fatalf("NewDecodeStep: %v", err)
		}
		addr := startCoordinatorH2C(t, config.ServerConfig{}, pipeline.New([]pipeline.Step{decodeStep}), gwClient)

		conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			t.Fatalf("grpc.NewClient: %v", err)
		}
		defer conn.Close()

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		stream, err := sglangapiv1.NewSglangServiceClient(conn).Generate(ctx, &sglangapiv1.GenerateRequest{
			Text: &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "truncated"}},
		})
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		item, err := stream.Recv()
		if err != nil || item.GetFrame().GetText() != "chunk-1" {
			t.Fatalf("first Recv = (%v, %v), want chunk-1", item, err)
		}
		if _, recvErr := stream.Recv(); status.Code(recvErr) != codes.Unavailable {
			t.Fatalf("Recv after truncated frame code = %v (%v), want Unavailable", status.Code(recvErr), recvErr)
		}
	})

	t.Run("decode stream after frame without grpc-status trailer", func(t *testing.T) {
		gw := startH2CTestGateway(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
			w.WriteHeader(http.StatusOK)
			frameBytes, err := grpcframe.MarshalProto(&sglangapiv1.GenerateStreamItem{
				Item: &sglangapiv1.GenerateStreamItem_Frame{
					Frame: &sglangapiv1.GenerateResponse{Text: "chunk-1"},
				},
			})
			if err != nil {
				t.Errorf("MarshalProto: %v", err)
				return
			}
			_, _ = w.Write(frameBytes)
		}))
		gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
		decodeStep, err := steps.NewDecodeStep(gwClient, map[string]any{"kv_connector": kv.SGLang})
		if err != nil {
			t.Fatalf("NewDecodeStep: %v", err)
		}
		addr := startCoordinatorH2C(t, config.ServerConfig{}, pipeline.New([]pipeline.Step{decodeStep}), gwClient)

		conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			t.Fatalf("grpc.NewClient: %v", err)
		}
		defer conn.Close()

		stream, err := sglangapiv1.NewSglangServiceClient(conn).Generate(context.Background(), &sglangapiv1.GenerateRequest{
			Text: &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "dropped trailer"}},
		})
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		item, err := stream.Recv()
		if err != nil || item.GetFrame().GetText() != "chunk-1" {
			t.Fatalf("first Recv = (%v, %v), want chunk-1", item, err)
		}
		if _, recvErr := stream.Recv(); status.Code(recvErr) != codes.Internal {
			t.Fatalf("second Recv code = %v (%v), want Internal", status.Code(recvErr), recvErr)
		}
	})

	t.Run("prefill stream without grpc-status fails decode", func(t *testing.T) {
		gw := startH2CTestGateway(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.Header.Get(reqcommon.EPPProfileHeaderKey) {
			case gateway.PhasePrefill:
				w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
				w.Header().Set(routing.PrefillEndpointHeader, "10.2.3.4:30000")
				w.WriteHeader(http.StatusOK)
				// End stream cleanly with no grpc-status trailer.
			case gateway.PhaseDecode:
				w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
				w.WriteHeader(http.StatusOK)
				_ = http.NewResponseController(w).Flush()
				<-r.Context().Done()
			}
		}))
		gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
		addr := startCoordinatorH2C(t, config.ServerConfig{}, newSGLangPDPipeline(t, gwClient), gwClient)

		conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			t.Fatalf("grpc.NewClient: %v", err)
		}
		defer conn.Close()

		stream, err := sglangapiv1.NewSglangServiceClient(conn).Generate(context.Background(), &sglangapiv1.GenerateRequest{
			Text: &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "prefill no status"}},
		})
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		if _, recvErr := stream.Recv(); status.Code(recvErr) != codes.Internal {
			t.Fatalf("Recv code = %v (%v), want Internal", status.Code(recvErr), recvErr)
		}
	})
}

func TestSGLangGRPC_PrefillFailureAfterDecodeStartedWritesErrorTrailer(t *testing.T) {
	firstFrameReadByClient := make(chan struct{})
	detailsBin := encodeStatusDetailsBin(t, codes.ResourceExhausted, "kv transfer aborted", &sglangapiv1.ErrorBody{
		Code:    http.StatusTooManyRequests,
		Message: "kv transfer aborted mid-stream",
	})

	gw := startH2CTestGateway(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Header.Get(reqcommon.EPPProfileHeaderKey) {
		case gateway.PhasePrefill:
			w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
			w.Header().Set(routing.PrefillEndpointHeader, "10.2.3.4:30000")
			w.WriteHeader(http.StatusOK)
			_ = http.NewResponseController(w).Flush()

			<-firstFrameReadByClient
			w.Header().Set(http.TrailerPrefix+grpcframe.HeaderStatus, strconv.Itoa(int(codes.ResourceExhausted)))
			w.Header().Set(http.TrailerPrefix+grpcframe.HeaderMessage, grpcframe.EncodeStatusMessage("kv transfer aborted"))
			w.Header().Set(http.TrailerPrefix+grpcframe.HeaderStatusDetailsBin, detailsBin)

		case gateway.PhaseDecode:
			w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
			w.WriteHeader(http.StatusOK)
			frameBytes, err := grpcframe.MarshalProto(&sglangapiv1.GenerateStreamItem{
				Item: &sglangapiv1.GenerateStreamItem_Frame{
					Frame: &sglangapiv1.GenerateResponse{Text: "partial token"},
				},
			})
			if err != nil {
				t.Errorf("MarshalProto: %v", err)
				return
			}
			_, _ = w.Write(frameBytes)
			_ = http.NewResponseController(w).Flush()
			<-r.Context().Done()
		}
	}))

	gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
	addr := startCoordinatorH2C(t, config.ServerConfig{}, newSGLangPDPipeline(t, gwClient), gwClient)

	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("grpc.NewClient: %v", err)
	}
	defer conn.Close()

	stream, err := sglangapiv1.NewSglangServiceClient(conn).Generate(context.Background(), &sglangapiv1.GenerateRequest{
		Text: &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "midstream prefill fail"}},
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	item, err := stream.Recv()
	if err != nil || item.GetFrame().GetText() != "partial token" {
		t.Fatalf("first Recv = (%v, %v), want partial token", item, err)
	}
	close(firstFrameReadByClient)

	_, recvErr := stream.Recv()
	if status.Code(recvErr) != codes.ResourceExhausted {
		t.Fatalf("second Recv code = %v (%v), want ResourceExhausted", status.Code(recvErr), recvErr)
	}
	assertErrorBodyDetail(t, recvErr, http.StatusTooManyRequests, "kv transfer aborted mid-stream")
}

func TestSGLangGRPC_PrefillInStreamErrorEncodesMessageAndMapsCode(t *testing.T) {
	const rawErrMsg = "prompt rejected: 50% over limit\ndetail line"
	decodeReady := make(chan struct{})

	gw := startH2CTestGateway(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Header.Get(reqcommon.EPPProfileHeaderKey) {
		case gateway.PhasePrefill:
			w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
			w.Header().Set(routing.PrefillEndpointHeader, "10.2.3.4:30000")
			w.WriteHeader(http.StatusOK)
			_ = http.NewResponseController(w).Flush()

			<-decodeReady
			errFrame, err := grpcframe.MarshalProto(&sglangapiv1.GenerateStreamItem{
				Item: &sglangapiv1.GenerateStreamItem_Error{
					Error: &sglangapiv1.GenerateStreamError{
						Error: &sglangapiv1.ErrorBody{
							Code:    http.StatusBadRequest,
							Message: rawErrMsg,
						},
					},
				},
			})
			if err != nil {
				t.Errorf("MarshalProto: %v", err)
				return
			}
			_, _ = w.Write(errFrame)
			w.Header().Set(http.TrailerPrefix+grpcframe.HeaderStatus, grpcframe.StatusOK)

		case gateway.PhaseDecode:
			w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
			w.WriteHeader(http.StatusOK)
			_ = http.NewResponseController(w).Flush()
			close(decodeReady)
			<-r.Context().Done()
		}
	}))

	gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
	addr := startCoordinatorH2C(t, config.ServerConfig{}, newSGLangPDPipeline(t, gwClient), gwClient)

	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("grpc.NewClient: %v", err)
	}
	defer conn.Close()

	stream, err := sglangapiv1.NewSglangServiceClient(conn).Generate(context.Background(), &sglangapiv1.GenerateRequest{
		Text: &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "in-stream error"}},
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	_, recvErr := stream.Recv()
	st, ok := status.FromError(recvErr)
	if !ok || st.Code() != codes.InvalidArgument {
		t.Fatalf("Recv status = (%v, %v), want InvalidArgument", st, recvErr)
	}
	if st.Message() != rawErrMsg {
		t.Fatalf("Recv message = %q, want %q", st.Message(), rawErrMsg)
	}
}

func TestSGLangGRPC_PerIndexCachedTokensAndTerminalWait(t *testing.T) {
	decodeSentTerminal := make(chan struct{})

	gw := startH2CTestGateway(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Header.Get(reqcommon.EPPProfileHeaderKey) {
		case gateway.PhasePrefill:
			w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
			w.Header().Set(routing.PrefillEndpointHeader, "10.2.3.4:30000")
			w.WriteHeader(http.StatusOK)
			_ = http.NewResponseController(w).Flush()

			// Delay prefill frames until decode has already emitted its terminal
			// frames upstream, exercising sglangPrefillWaitOnTerminal as well as
			// per-index cached_tokens tracking.
			<-decodeSentTerminal
			time.Sleep(20 * time.Millisecond)

			for _, item := range []struct {
				idx    uint32
				cached uint32
			}{{idx: 0, cached: 40}, {idx: 1, cached: 16}} {
				frameBytes, err := grpcframe.MarshalProto(&sglangapiv1.GenerateStreamItem{
					Item: &sglangapiv1.GenerateStreamItem_Frame{
						Frame: &sglangapiv1.GenerateResponse{
							Index: proto.Uint32(item.idx),
							MetaInfo: &sglangapiv1.GenerateMetaInfo{
								PromptTokens: 64,
								CachedTokens: item.cached,
								FinishReason: &sglangapiv1.FinishReason{
									Kind: &sglangapiv1.FinishReason_Length{Length: &sglangapiv1.FinishLength{Length: proto.Uint64(1)}},
								},
							},
						},
					},
				})
				if err != nil {
					t.Errorf("MarshalProto prefill: %v", err)
					return
				}
				_, _ = w.Write(frameBytes)
			}
			w.Header().Set(http.TrailerPrefix+grpcframe.HeaderStatus, grpcframe.StatusOK)

		case gateway.PhaseDecode:
			w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
			w.WriteHeader(http.StatusOK)

			for _, idx := range []uint32{0, 1} {
				frameBytes, err := grpcframe.MarshalProto(&sglangapiv1.GenerateStreamItem{
					Item: &sglangapiv1.GenerateStreamItem_Frame{
						Frame: &sglangapiv1.GenerateResponse{
							Index: proto.Uint32(idx),
							Text:  "out",
							MetaInfo: &sglangapiv1.GenerateMetaInfo{
								PromptTokens:     64,
								CompletionTokens: 1,
								CachedTokens:     0,
								FinishReason: &sglangapiv1.FinishReason{
									Kind: &sglangapiv1.FinishReason_Stop{Stop: &sglangapiv1.FinishStop{}},
								},
							},
						},
					},
				})
				if err != nil {
					t.Errorf("MarshalProto decode: %v", err)
					return
				}
				_, _ = w.Write(frameBytes)
			}
			_ = http.NewResponseController(w).Flush()
			close(decodeSentTerminal)
			w.Header().Set(http.TrailerPrefix+grpcframe.HeaderStatus, grpcframe.StatusOK)
		}
	}))

	gwClient := gateway.New(config.GatewayConfig{Address: gw.URL})
	addr := startCoordinatorH2C(t, config.ServerConfig{}, newSGLangPDPipeline(t, gwClient), gwClient)

	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("grpc.NewClient: %v", err)
	}
	defer conn.Close()

	stream, err := sglangapiv1.NewSglangServiceClient(conn).Generate(context.Background(), &sglangapiv1.GenerateRequest{
		Text: &sglangapiv1.StringOrList{
			Value: &sglangapiv1.StringOrList_Many{Many: &sglangapiv1.StringList{Items: []string{"p0", "p1"}}},
		},
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}

	gotCached := map[uint32]uint32{}
	for {
		item, recvErr := stream.Recv()
		if errors.Is(recvErr, io.EOF) {
			break
		}
		if recvErr != nil {
			t.Fatalf("stream.Recv: %v", recvErr)
		}
		gotCached[item.GetFrame().GetIndex()] = item.GetFrame().GetMetaInfo().GetCachedTokens()
	}
	if gotCached[0] != 40 || gotCached[1] != 16 {
		t.Fatalf("got per-index cached_tokens = %v, want {0:40, 1:16}", gotCached)
	}
}

func TestSGLangGRPC_ClientCancellationAndTLSServing(t *testing.T) {
	t.Run("client cancellation cancels prefill and decode", func(t *testing.T) {
		var prefillCancelled, decodeCancelled atomic.Bool
		gw := startH2CTestGateway(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.Header.Get(reqcommon.EPPProfileHeaderKey) {
			case gateway.PhasePrefill:
				w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
				w.Header().Set(routing.PrefillEndpointHeader, "10.2.3.4:30000")
				w.WriteHeader(http.StatusOK)
				_ = http.NewResponseController(w).Flush()
				<-r.Context().Done()
				prefillCancelled.Store(true)
			case gateway.PhaseDecode:
				w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
				w.WriteHeader(http.StatusOK)
				frameBytes, _ := grpcframe.MarshalProto(&sglangapiv1.GenerateStreamItem{
					Item: &sglangapiv1.GenerateStreamItem_Frame{
						Frame: &sglangapiv1.GenerateResponse{Text: "tok-1"},
					},
				})
				_, _ = w.Write(frameBytes)
				_ = http.NewResponseController(w).Flush()
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

		callCtx, cancelCall := context.WithCancel(context.Background())
		stream, err := sglangapiv1.NewSglangServiceClient(conn).Generate(callCtx, &sglangapiv1.GenerateRequest{
			Text: &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "cancel me"}},
		})
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		if _, err := stream.Recv(); err != nil {
			t.Fatalf("first Recv: %v", err)
		}
		cancelCall()

		deadline := time.Now().Add(2 * time.Second)
		for (!prefillCancelled.Load() || !decodeCancelled.Load()) && time.Now().Before(deadline) {
			time.Sleep(10 * time.Millisecond)
		}
		if !prefillCancelled.Load() || !decodeCancelled.Load() {
			t.Fatalf("expected both legs cancelled, got prefill=%v decode=%v", prefillCancelled.Load(), decodeCancelled.Load())
		}
	})

	t.Run("TLS HTTP/2 serving", func(t *testing.T) {
		backendSrv := grpc.NewServer()
		sglangapiv1.RegisterSglangServiceServer(backendSrv, &stubSGLangServiceServer{})
		backendLn, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatalf("net.Listen: %v", err)
		}
		go func() { _ = backendSrv.Serve(backendLn) }()
		t.Cleanup(backendSrv.Stop)

		gwClient := gateway.New(config.GatewayConfig{Address: "http://" + backendLn.Addr().String()})
		decodeStep, err := steps.NewDecodeStep(gwClient, map[string]any{"kv_connector": kv.SGLang})
		if err != nil {
			t.Fatalf("NewDecodeStep: %v", err)
		}

		srv, err := server.New(config.ServerConfig{SecureServing: true}, pipeline.New([]pipeline.Step{decodeStep}), gwClient)
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

		tlsCreds := credentials.NewTLS(&tls.Config{InsecureSkipVerify: true}) //#nosec G402 -- self-signed test cert
		conn, err := grpc.NewClient(ln.Addr().String(), grpc.WithTransportCredentials(tlsCreds))
		if err != nil {
			t.Fatalf("grpc.NewClient TLS: %v", err)
		}
		defer conn.Close()

		stream, err := sglangapiv1.NewSglangServiceClient(conn).Generate(context.Background(), &sglangapiv1.GenerateRequest{
			Text: &sglangapiv1.StringOrList{Value: &sglangapiv1.StringOrList_One{One: "over tls"}},
		})
		if err != nil {
			t.Fatalf("Generate TLS: %v", err)
		}
		item, err := stream.Recv()
		if err != nil || item.GetFrame().GetText() != "stub: over tls" {
			t.Fatalf("Recv TLS = (%v, %v), want stub: over tls", item, err)
		}
		if _, err := stream.Recv(); !errors.Is(err, io.EOF) {
			t.Fatalf("expected EOF over TLS, got %v", err)
		}
	})
}
