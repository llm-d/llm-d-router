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

package steps

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/protobuf/proto"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpctest"
	sglangapiv1 "github.com/llm-d/llm-d-router/pkg/common/grpc/sglang/gen/apiv1"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/kv"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/metrics/metricstest"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

type sglangRerunGateway struct {
	mu   sync.Mutex
	reqs map[string][]*sglangapiv1.GenerateRequest
}

func newSGLangRerunGateway(t *testing.T, respond func(w http.ResponseWriter, r *http.Request, req *sglangapiv1.GenerateRequest, phase string, attempt int)) (*gateway.Client, *sglangRerunGateway) {
	t.Helper()
	gw := &sglangRerunGateway{reqs: map[string][]*sglangapiv1.GenerateRequest{}}
	srv := grpctest.NewH2CServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		frames := grpcframe.NewReader(r.Body, maxGRPCFrameBytes)
		defer frames.Release()
		frame, err := frames.Next()
		if err != nil {
			// A concurrent peer failure can cancel the leg's HTTP/2 stream
			// while the server is still reading the request body.
			if r.Context().Err() != nil {
				return
			}
			t.Errorf("read gRPC request frame: %v", err)
			return
		}
		req := &sglangapiv1.GenerateRequest{}
		if err := proto.Unmarshal(frame[grpcframe.HeaderLen:], req); err != nil {
			t.Errorf("unmarshal GenerateRequest: %v", err)
			return
		}
		phase := r.Header.Get(reqcommon.EPPProfileHeaderKey)
		gw.mu.Lock()
		gw.reqs[phase] = append(gw.reqs[phase], req)
		attempt := len(gw.reqs[phase])
		gw.mu.Unlock()
		respond(w, r, req, phase, attempt)
	}))
	return gateway.New(config.GatewayConfig{Address: srv.URL}), gw
}

func (g *sglangRerunGateway) requests(phase string) []*sglangapiv1.GenerateRequest {
	g.mu.Lock()
	defer g.mu.Unlock()
	return slices.Clone(g.reqs[phase])
}

func sglangBootstrapRoom(t *testing.T, req *sglangapiv1.GenerateRequest) int64 {
	t.Helper()
	require.NotNil(t, req.GetBootstrapRoom().GetOne(), "expected bootstrap_room.one")
	return req.GetBootstrapRoom().GetOne().GetValue()
}

func sglangBootstrapHost(t *testing.T, req *sglangapiv1.GenerateRequest) string {
	t.Helper()
	require.NotNil(t, req.GetBootstrapHost().GetOne(), "expected bootstrap_host.one")
	return req.GetBootstrapHost().GetOne().GetValue()
}

func sglangStreamErrorItem(code uint32, msg string, retryable *bool) *sglangapiv1.GenerateStreamItem {
	return &sglangapiv1.GenerateStreamItem{
		Item: &sglangapiv1.GenerateStreamItem_Error{
			Error: &sglangapiv1.GenerateStreamError{
				Error: &sglangapiv1.ErrorBody{
					Code:      code,
					Message:   msg,
					Retryable: retryable,
				},
			},
		},
	}
}

func sglangTokenFrameItem(text string, cachedTokens uint32, finished bool) *sglangapiv1.GenerateStreamItem {
	var finish *sglangapiv1.FinishReason
	if finished {
		finish = &sglangapiv1.FinishReason{
			Kind: &sglangapiv1.FinishReason_Stop{Stop: &sglangapiv1.FinishStop{}},
		}
	}
	return &sglangapiv1.GenerateStreamItem{
		Item: &sglangapiv1.GenerateStreamItem_Frame{
			Frame: &sglangapiv1.GenerateResponse{
				Text: text,
				MetaInfo: &sglangapiv1.GenerateMetaInfo{
					PromptTokens: 4,
					CachedTokens: cachedTokens,
					FinishReason: finish,
				},
			},
		},
	}
}

func sglangPayloadBytes(t *testing.T) []byte {
	t.Helper()
	raw, err := proto.Marshal(&sglangapiv1.GenerateRequest{
		InputIds: &sglangapiv1.TokenIdsOrList{
			Value: &sglangapiv1.TokenIdsOrList_One{
				One: &sglangapiv1.TokenIds{Ids: []int64{1, 2, 3, 4}},
			},
		},
		Stream: proto.Bool(true),
	})
	require.NoError(t, err)
	return raw
}

func runSGLangPipeline(ctx context.Context, t *testing.T, gwClient *gateway.Client, withPrefill bool, maxReruns int) (*httptest.ResponseRecorder, error) {
	t.Helper()
	rec := httptest.NewRecorder()
	err := runSGLangPipelineWithWriter(ctx, t, gwClient, withPrefill, maxReruns, rec)
	return rec, err
}

func runSGLangPipelineWithWriter(ctx context.Context, t *testing.T, gwClient *gateway.Client, withPrefill bool, maxReruns int, w http.ResponseWriter) error {
	t.Helper()
	params := map[string]any{ParamKVConnector: kv.SGLang}
	var pipeSteps []pipeline.Step
	if withPrefill {
		prefill, err := NewPrefillStep(gwClient, params)
		require.NoError(t, err)
		pipeSteps = append(pipeSteps, prefill)
	}
	decode, err := NewDecodeStep(gwClient, params)
	require.NoError(t, err)
	pipeSteps = append(pipeSteps, decode)
	p := pipeline.New(pipeSteps)
	p.SetMaxStepReruns(maxReruns)

	reqCtx := &pipeline.RequestContext{
		RequestID:        "req-sglang-rerun",
		Model:            "test-model",
		Stream:           true,
		OriginalPath:     sglangapiv1.SglangService_Generate_FullMethodName,
		OriginalHeaders:  http.Header{},
		KVTransferParams: map[string]any{},
		ResponseWriter:   w,
		GRPC: &pipeline.SGLangGRPCRequest{
			Payload: sglangPayloadBytes(t),
		},
	}
	err = p.Execute(ctx, reqCtx)
	reqCtx.ClosePrefillStream()
	return err
}

type notifyFirstWriteRecorder struct {
	*httptest.ResponseRecorder
	once sync.Once
	ch   chan struct{}
}

func (n *notifyFirstWriteRecorder) Write(p []byte) (int, error) {
	written, err := n.ResponseRecorder.Write(p)
	n.once.Do(func() { close(n.ch) })
	return written, err
}

func readSGLangStreamItems(t *testing.T, body []byte) []*sglangapiv1.GenerateStreamItem {
	t.Helper()
	r := grpcframe.NewReader(bytes.NewReader(body), maxGRPCFrameBytes)
	defer r.Release()
	var items []*sglangapiv1.GenerateStreamItem
	for {
		frame, err := r.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		require.NoError(t, err)
		item := &sglangapiv1.GenerateStreamItem{}
		require.NoError(t, proto.Unmarshal(frame[grpcframe.HeaderLen:], item))
		items = append(items, item)
	}
	return items
}

func writeSGLangPrefillHeaders(w http.ResponseWriter, host string) {
	w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
	w.Header().Set(routing.PrefillEndpointHeader, host+":8000")
	w.Header().Set("Trailer", grpcframe.HeaderStatus)
	w.WriteHeader(http.StatusOK)
	_ = http.NewResponseController(w).Flush()
}

func writeSGLangFrame(t *testing.T, w http.ResponseWriter, item *sglangapiv1.GenerateStreamItem) {
	t.Helper()
	framed, err := grpcframe.MarshalProto(item)
	require.NoError(t, err)
	_, err = w.Write(framed)
	require.NoError(t, err)
	_ = http.NewResponseController(w).Flush()
}

func TestSGLangGRPCRerun_DecodeRetryableErrorBeforeFirstToken(t *testing.T) {
	reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)
	prefill1Canceled := make(chan struct{})

	gwClient, gw := newSGLangRerunGateway(t, func(w http.ResponseWriter, r *http.Request, _ *sglangapiv1.GenerateRequest, phase string, attempt int) {
		switch phase {
		case gateway.PhasePrefill:
			writeSGLangPrefillHeaders(w, fmt.Sprintf("prefill-%d", attempt))
			if attempt == 1 {
				<-r.Context().Done()
				close(prefill1Canceled)
				return
			}
			writeSGLangFrame(t, w, sglangTokenFrameItem("", 48, true))
			w.Header().Set(grpcframe.HeaderStatus, grpcframe.StatusOK)
		case gateway.PhaseDecode:
			if attempt == 1 {
				grpctest.WriteFrames(t, w, sglangStreamErrorItem(http.StatusInternalServerError, "kv transfer failed", proto.Bool(true)))
				return
			}
			grpctest.WriteFrames(t, w, sglangTokenFrameItem("ok-2", 0, true))
		default:
			t.Errorf("unexpected phase %q", phase)
		}
	})

	rec, err := runSGLangPipeline(rerunTestContext(t), t, gwClient, true, 1)
	require.NoError(t, err)

	select {
	case <-prefill1Canceled:
	case <-time.After(3 * time.Second):
		t.Fatal("attempt-1 prefill stream was not canceled after retryable decode error")
	}

	prefills := gw.requests(gateway.PhasePrefill)
	decodes := gw.requests(gateway.PhaseDecode)
	require.Len(t, prefills, 2)
	require.Len(t, decodes, 2)

	room1 := sglangBootstrapRoom(t, prefills[0])
	room2 := sglangBootstrapRoom(t, prefills[1])
	require.NotEqual(t, room1, room2, "re-run must allocate a fresh bootstrap_room")
	require.Equal(t, room1, sglangBootstrapRoom(t, decodes[0]))
	require.Equal(t, room2, sglangBootstrapRoom(t, decodes[1]))
	require.Equal(t, "prefill-1", sglangBootstrapHost(t, decodes[0]))
	require.Equal(t, "prefill-2", sglangBootstrapHost(t, decodes[1]))

	items := readSGLangStreamItems(t, rec.Body.Bytes())
	require.Len(t, items, 1)
	require.Equal(t, "ok-2", items[0].GetFrame().GetText())
	require.Equal(t, uint32(48), items[0].GetFrame().GetMetaInfo().GetCachedTokens())
	require.Equal(t, grpcframe.StatusOK, rec.Header().Get(http.TrailerPrefix+grpcframe.HeaderStatus))

	require.InDelta(t, 1.0, counterValue(t, reg, stepRerunMetric, map[string]string{"step": DecodeStepName}), 1e-9)
	require.Zero(t, counterValue(t, reg, stepErrorMetric, map[string]string{"step": DecodeStepName}))
}

func TestSGLangGRPCRerun_PrefillRetryableErrorBeforeDecodeHeaders(t *testing.T) {
	t.Run("prefill fails while decode dial is in flight", func(t *testing.T) {
		reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)
		decode1Dialed := make(chan struct{})
		decode1Canceled := make(chan struct{})

		gwClient, gw := newSGLangRerunGateway(t, func(w http.ResponseWriter, r *http.Request, req *sglangapiv1.GenerateRequest, phase string, attempt int) {
			switch phase {
			case gateway.PhasePrefill:
				writeSGLangPrefillHeaders(w, fmt.Sprintf("prefill-%d", attempt))
				if attempt == 1 {
					<-decode1Dialed
					writeSGLangFrame(t, w, sglangStreamErrorItem(http.StatusInternalServerError, "kv transfer aborted", proto.Bool(true)))
					w.Header().Set(grpcframe.HeaderStatus, grpcframe.StatusOK)
					return
				}
				writeSGLangFrame(t, w, sglangTokenFrameItem("", 32, true))
				w.Header().Set(grpcframe.HeaderStatus, grpcframe.StatusOK)
			case gateway.PhaseDecode:
				if sglangBootstrapHost(t, req) == "prefill-1" {
					close(decode1Dialed)
					<-r.Context().Done()
					close(decode1Canceled)
					return
				}
				grpctest.WriteFrames(t, w, sglangTokenFrameItem("after-prefill-rerun", 0, true))
			default:
				t.Errorf("unexpected phase %q", phase)
			}
		})

		rec, err := runSGLangPipeline(rerunTestContext(t), t, gwClient, true, 1)
		require.NoError(t, err)

		select {
		case <-decode1Canceled:
		case <-time.After(3 * time.Second):
			t.Fatal("attempt-1 decode was not canceled after retryable prefill error")
		}

		prefills := gw.requests(gateway.PhasePrefill)
		decodes := gw.requests(gateway.PhaseDecode)
		require.Len(t, prefills, 2)
		require.Len(t, decodes, 2)
		require.NotEqual(t, sglangBootstrapRoom(t, prefills[0]), sglangBootstrapRoom(t, prefills[1]))
		require.Equal(t, "prefill-2", sglangBootstrapHost(t, decodes[1]))

		items := readSGLangStreamItems(t, rec.Body.Bytes())
		require.Len(t, items, 1)
		require.Equal(t, "after-prefill-rerun", items[0].GetFrame().GetText())
		require.Equal(t, uint32(32), items[0].GetFrame().GetMetaInfo().GetCachedTokens())

		require.InDelta(t, 1.0, counterValue(t, reg, stepRerunMetric, map[string]string{"step": PrefillStepName}), 1e-9)
	})

	t.Run("prefill fails immediately", func(t *testing.T) {
		reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)

		gwClient, gw := newSGLangRerunGateway(t, func(w http.ResponseWriter, r *http.Request, req *sglangapiv1.GenerateRequest, phase string, attempt int) {
			switch phase {
			case gateway.PhasePrefill:
				writeSGLangPrefillHeaders(w, fmt.Sprintf("prefill-%d", attempt))
				if attempt == 1 {
					writeSGLangFrame(t, w, sglangStreamErrorItem(http.StatusInternalServerError, "kv transfer aborted immediately", proto.Bool(true)))
					w.Header().Set(grpcframe.HeaderStatus, grpcframe.StatusOK)
					return
				}
				writeSGLangFrame(t, w, sglangTokenFrameItem("", 24, true))
				w.Header().Set(grpcframe.HeaderStatus, grpcframe.StatusOK)
			case gateway.PhaseDecode:
				if sglangBootstrapHost(t, req) == "prefill-1" {
					<-r.Context().Done()
					return
				}
				grpctest.WriteFrames(t, w, sglangTokenFrameItem("after-immediate-prefill-rerun", 0, true))
			default:
				t.Errorf("unexpected phase %q", phase)
			}
		})

		rec, err := runSGLangPipeline(rerunTestContext(t), t, gwClient, true, 1)
		require.NoError(t, err)

		prefills := gw.requests(gateway.PhasePrefill)
		decodes := gw.requests(gateway.PhaseDecode)
		require.Len(t, prefills, 2)
		require.NotEmpty(t, decodes)
		require.NotEqual(t, sglangBootstrapRoom(t, prefills[0]), sglangBootstrapRoom(t, prefills[1]))
		require.Equal(t, "prefill-2", sglangBootstrapHost(t, decodes[len(decodes)-1]))

		items := readSGLangStreamItems(t, rec.Body.Bytes())
		require.Len(t, items, 1)
		require.Equal(t, "after-immediate-prefill-rerun", items[0].GetFrame().GetText())
		require.Equal(t, uint32(24), items[0].GetFrame().GetMetaInfo().GetCachedTokens())

		require.InDelta(t, 1.0, counterValue(t, reg, stepRerunMetric, map[string]string{"step": PrefillStepName}), 1e-9)
	})
}

func TestSGLangGRPCRerun_PrefillRetryableErrorAfterDecodeHeadersBeforeFirstFrame(t *testing.T) {
	reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)
	decode1HeadersSent := make(chan struct{})
	decode1Canceled := make(chan struct{})

	gwClient, gw := newSGLangRerunGateway(t, func(w http.ResponseWriter, r *http.Request, _ *sglangapiv1.GenerateRequest, phase string, attempt int) {
		switch phase {
		case gateway.PhasePrefill:
			writeSGLangPrefillHeaders(w, fmt.Sprintf("prefill-%d", attempt))
			if attempt == 1 {
				<-decode1HeadersSent
				writeSGLangFrame(t, w, sglangStreamErrorItem(http.StatusInternalServerError, "kv transfer aborted mid-wait", proto.Bool(true)))
				w.Header().Set(grpcframe.HeaderStatus, grpcframe.StatusOK)
				return
			}
			writeSGLangFrame(t, w, sglangTokenFrameItem("", 16, true))
			w.Header().Set(grpcframe.HeaderStatus, grpcframe.StatusOK)
		case gateway.PhaseDecode:
			if attempt == 1 {
				w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
				w.Header().Set("Trailer", grpcframe.HeaderStatus)
				w.WriteHeader(http.StatusOK)
				_ = http.NewResponseController(w).Flush()
				close(decode1HeadersSent)
				<-r.Context().Done()
				close(decode1Canceled)
				return
			}
			grpctest.WriteFrames(t, w, sglangTokenFrameItem("ok-after-headers", 0, true))
		default:
			t.Errorf("unexpected phase %q", phase)
		}
	})

	rec, err := runSGLangPipeline(rerunTestContext(t), t, gwClient, true, 1)
	require.NoError(t, err)

	select {
	case <-decode1Canceled:
	case <-time.After(3 * time.Second):
		t.Fatal("attempt-1 decode stream was not canceled after retryable prefill error")
	}

	prefills := gw.requests(gateway.PhasePrefill)
	decodes := gw.requests(gateway.PhaseDecode)
	require.Len(t, prefills, 2)
	require.Len(t, decodes, 2)
	require.NotEqual(t, sglangBootstrapRoom(t, prefills[0]), sglangBootstrapRoom(t, prefills[1]))
	require.Equal(t, "prefill-2", sglangBootstrapHost(t, decodes[1]))

	items := readSGLangStreamItems(t, rec.Body.Bytes())
	require.Len(t, items, 1)
	require.Equal(t, "ok-after-headers", items[0].GetFrame().GetText())

	require.InDelta(t, 1.0, counterValue(t, reg, stepRerunMetric, map[string]string{"step": PrefillStepName}), 1e-9)
}

func TestSGLangGRPCRerun_PrefillRetryableErrorOnDecodeEmptyEOF(t *testing.T) {
	prefill1Done := make(chan struct{})

	gwClient, gw := newSGLangRerunGateway(t, func(w http.ResponseWriter, _ *http.Request, _ *sglangapiv1.GenerateRequest, phase string, attempt int) {
		switch phase {
		case gateway.PhasePrefill:
			writeSGLangPrefillHeaders(w, fmt.Sprintf("prefill-%d", attempt))
			if attempt == 1 {
				<-prefill1Done
				writeSGLangFrame(t, w, sglangStreamErrorItem(http.StatusInternalServerError, "prefill error at eof", proto.Bool(true)))
				w.Header().Set(grpcframe.HeaderStatus, grpcframe.StatusOK)
				return
			}
			writeSGLangFrame(t, w, sglangTokenFrameItem("", 0, true))
			w.Header().Set(grpcframe.HeaderStatus, grpcframe.StatusOK)
		case gateway.PhaseDecode:
			if attempt == 1 {
				// Send HTTP 200 OK + zero frames + OK trailer while prefill fails right after.
				w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
				w.Header().Set("Trailer", grpcframe.HeaderStatus)
				w.WriteHeader(http.StatusOK)
				_ = http.NewResponseController(w).Flush()
				close(prefill1Done)
				w.Header().Set(grpcframe.HeaderStatus, grpcframe.StatusOK)
				return
			}
			grpctest.WriteFrames(t, w, sglangTokenFrameItem("ok-eof-rerun", 0, true))
		default:
			t.Errorf("unexpected phase %q", phase)
		}
	})

	rec, err := runSGLangPipeline(rerunTestContext(t), t, gwClient, true, 1)
	require.NoError(t, err)
	require.Len(t, gw.requests(gateway.PhasePrefill), 2)
	items := readSGLangStreamItems(t, rec.Body.Bytes())
	require.Len(t, items, 1)
	require.Equal(t, "ok-eof-rerun", items[0].GetFrame().GetText())
}

func TestSGLangGRPCRerun_DecodeOnly(t *testing.T) {
	reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)

	gwClient, gw := newSGLangRerunGateway(t, func(w http.ResponseWriter, _ *http.Request, _ *sglangapiv1.GenerateRequest, phase string, attempt int) {
		require.Equal(t, gateway.PhaseDecode, phase)
		if attempt == 1 {
			grpctest.WriteFrames(t, w, sglangStreamErrorItem(http.StatusInternalServerError, "transient decode error", proto.Bool(true)))
			return
		}
		grpctest.WriteFrames(t, w, sglangTokenFrameItem("decode-only-ok", 0, true))
	})

	rec, err := runSGLangPipeline(rerunTestContext(t), t, gwClient, false, 1)
	require.NoError(t, err)
	require.Len(t, gw.requests(gateway.PhaseDecode), 2)

	items := readSGLangStreamItems(t, rec.Body.Bytes())
	require.Len(t, items, 1)
	require.Equal(t, "decode-only-ok", items[0].GetFrame().GetText())

	require.InDelta(t, 1.0, counterValue(t, reg, stepRerunMetric, map[string]string{"step": DecodeStepName}), 1e-9)
}

func TestSGLangGRPCRerun_BoundedAttemptsAndExhaustion(t *testing.T) {
	reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)
	prefill2Canceled := make(chan struct{})

	gwClient, gw := newSGLangRerunGateway(t, func(w http.ResponseWriter, r *http.Request, _ *sglangapiv1.GenerateRequest, phase string, attempt int) {
		switch phase {
		case gateway.PhasePrefill:
			writeSGLangPrefillHeaders(w, fmt.Sprintf("prefill-%d", attempt))
			switch attempt {
			case 1:
				writeSGLangFrame(t, w, sglangStreamErrorItem(http.StatusInternalServerError, "prefill fail 1", proto.Bool(true)))
				w.Header().Set(grpcframe.HeaderStatus, grpcframe.StatusOK)
			case 2:
				<-r.Context().Done()
				close(prefill2Canceled)
			default:
				writeSGLangFrame(t, w, sglangTokenFrameItem("", 0, true))
				w.Header().Set(grpcframe.HeaderStatus, grpcframe.StatusOK)
			}
		case gateway.PhaseDecode:
			grpctest.WriteFrames(t, w, sglangStreamErrorItem(http.StatusInternalServerError, fmt.Sprintf("decode fail %d", attempt), proto.Bool(true)))
		}
	})

	rec, err := runSGLangPipeline(rerunTestContext(t), t, gwClient, true, 2)
	require.NoError(t, err)
	select {
	case <-prefill2Canceled:
	case <-time.After(3 * time.Second):
		t.Fatal("attempt-2 prefill stream was not canceled after retryable decode error")
	}
	require.Len(t, gw.requests(gateway.PhasePrefill), 3, "1 initial attempt + 2 reruns")

	items := readSGLangStreamItems(t, rec.Body.Bytes())
	require.Len(t, items, 1)
	require.NotNil(t, items[0].GetError())
	require.True(t, items[0].GetError().GetError().GetRetryable())
	require.Contains(t, items[0].GetError().GetError().GetMessage(), "decode fail")
	require.InDelta(t, 2.0, counterValue(t, reg, stepRerunMetric, nil), 1e-9)
}

func TestSGLangGRPCRerun_DisabledByDefault(t *testing.T) {
	reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)

	gwClient, gw := newSGLangRerunGateway(t, func(w http.ResponseWriter, _ *http.Request, _ *sglangapiv1.GenerateRequest, phase string, _ int) {
		switch phase {
		case gateway.PhasePrefill:
			writeSGLangPrefillHeaders(w, "prefill-1")
			writeSGLangFrame(t, w, sglangTokenFrameItem("", 0, true))
			w.Header().Set(grpcframe.HeaderStatus, grpcframe.StatusOK)
		case gateway.PhaseDecode:
			grpctest.WriteFrames(t, w, sglangStreamErrorItem(http.StatusInternalServerError, "no rerun when 0", proto.Bool(true)))
		}
	})

	rec, err := runSGLangPipeline(rerunTestContext(t), t, gwClient, true, 0)
	require.NoError(t, err)
	require.Len(t, gw.requests(gateway.PhasePrefill), 1)
	require.Len(t, gw.requests(gateway.PhaseDecode), 1)

	items := readSGLangStreamItems(t, rec.Body.Bytes())
	require.Len(t, items, 1)
	require.Equal(t, "no rerun when 0", items[0].GetError().GetError().GetMessage())
	require.Zero(t, counterValue(t, reg, stepRerunMetric, nil))
}

func TestSGLangGRPCRerun_NonRetryableErrorDoesNotRerun(t *testing.T) {
	cases := []struct {
		name      string
		failPhase string
		retryable *bool
	}{
		{name: "decode retryable=false", failPhase: gateway.PhaseDecode, retryable: proto.Bool(false)},
		{name: "decode retryable=nil", failPhase: gateway.PhaseDecode, retryable: nil},
		{name: "prefill retryable=false", failPhase: gateway.PhasePrefill, retryable: proto.Bool(false)},
		{name: "prefill retryable=nil", failPhase: gateway.PhasePrefill, retryable: nil},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)
			gwClient, gw := newSGLangRerunGateway(t, func(w http.ResponseWriter, r *http.Request, _ *sglangapiv1.GenerateRequest, phase string, _ int) {
				switch phase {
				case gateway.PhasePrefill:
					writeSGLangPrefillHeaders(w, "prefill-1")
					if tc.failPhase == gateway.PhasePrefill {
						writeSGLangFrame(t, w, sglangStreamErrorItem(http.StatusBadRequest, "bad input", tc.retryable))
						w.Header().Set(grpcframe.HeaderStatus, grpcframe.StatusOK)
						return
					}
					writeSGLangFrame(t, w, sglangTokenFrameItem("", 0, true))
					w.Header().Set(grpcframe.HeaderStatus, grpcframe.StatusOK)
				case gateway.PhaseDecode:
					if tc.failPhase == gateway.PhasePrefill {
						<-r.Context().Done()
						return
					}
					grpctest.WriteFrames(t, w, sglangStreamErrorItem(http.StatusBadRequest, "bad input", tc.retryable))
				}
			})

			rec, err := runSGLangPipeline(rerunTestContext(t), t, gwClient, true, 2)
			require.Len(t, gw.requests(gateway.PhasePrefill), 1)
			require.Zero(t, counterValue(t, reg, stepRerunMetric, nil))
			if tc.failPhase == gateway.PhaseDecode {
				require.NoError(t, err)
				require.Len(t, gw.requests(gateway.PhaseDecode), 1)
				items := readSGLangStreamItems(t, rec.Body.Bytes())
				require.Len(t, items, 1)
				require.Equal(t, "bad input", items[0].GetError().GetError().GetMessage())
			} else {
				require.Error(t, err)
				require.Equal(t, strconv.Itoa(int(codes.InvalidArgument)), rec.Header().Get(grpcframe.HeaderStatus))
			}
		})
	}
}

func TestSGLangGRPCRerun_NoRerunAfterFirstOutputFrame(t *testing.T) {
	t.Run("decode error after first frame", func(t *testing.T) {
		reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)
		gwClient, gw := newSGLangRerunGateway(t, func(w http.ResponseWriter, _ *http.Request, _ *sglangapiv1.GenerateRequest, phase string, _ int) {
			switch phase {
			case gateway.PhasePrefill:
				writeSGLangPrefillHeaders(w, "prefill-1")
				writeSGLangFrame(t, w, sglangTokenFrameItem("", 0, true))
				w.Header().Set(grpcframe.HeaderStatus, grpcframe.StatusOK)
			case gateway.PhaseDecode:
				grpctest.WriteFrames(t, w,
					sglangTokenFrameItem("first-tok", 0, false),
					sglangStreamErrorItem(http.StatusInternalServerError, "late decode error", proto.Bool(true)),
				)
			}
		})

		rec, err := runSGLangPipeline(rerunTestContext(t), t, gwClient, true, 2)
		require.NoError(t, err)
		require.Len(t, gw.requests(gateway.PhasePrefill), 1)
		require.Len(t, gw.requests(gateway.PhaseDecode), 1)
		require.Zero(t, counterValue(t, reg, stepRerunMetric, nil))

		items := readSGLangStreamItems(t, rec.Body.Bytes())
		require.Len(t, items, 2)
		require.Equal(t, "first-tok", items[0].GetFrame().GetText())
		require.Equal(t, "late decode error", items[1].GetError().GetError().GetMessage())
	})

	t.Run("prefill error after first decode frame", func(t *testing.T) {
		reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)
		firstFrameCommitted := make(chan struct{})
		gwClient, gw := newSGLangRerunGateway(t, func(w http.ResponseWriter, r *http.Request, _ *sglangapiv1.GenerateRequest, phase string, _ int) {
			switch phase {
			case gateway.PhasePrefill:
				writeSGLangPrefillHeaders(w, "prefill-1")
				<-firstFrameCommitted
				writeSGLangFrame(t, w, sglangStreamErrorItem(http.StatusInternalServerError, "late prefill error", proto.Bool(true)))
				w.Header().Set(grpcframe.HeaderStatus, grpcframe.StatusOK)
			case gateway.PhaseDecode:
				w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
				w.Header().Set("Trailer", grpcframe.HeaderStatus)
				w.WriteHeader(http.StatusOK)
				writeSGLangFrame(t, w, sglangTokenFrameItem("first-tok", 0, false))
				_ = http.NewResponseController(w).Flush()
				<-r.Context().Done()
			}
		})

		rec := &notifyFirstWriteRecorder{
			ResponseRecorder: httptest.NewRecorder(),
			ch:               firstFrameCommitted,
		}
		err := runSGLangPipelineWithWriter(rerunTestContext(t), t, gwClient, true, 2, rec)
		require.Error(t, err)
		require.Len(t, gw.requests(gateway.PhasePrefill), 1)
		require.Len(t, gw.requests(gateway.PhaseDecode), 1)
		require.Zero(t, counterValue(t, reg, stepRerunMetric, nil))

		items := readSGLangStreamItems(t, rec.Body.Bytes())
		require.Len(t, items, 1)
		require.Equal(t, "first-tok", items[0].GetFrame().GetText())
		require.Equal(t, strconv.Itoa(int(codes.Internal)), rec.Header().Get(http.TrailerPrefix+grpcframe.HeaderStatus))
	})
}

func TestSGLangGRPCRerun_ClientDisconnectPreventsRerun(t *testing.T) {
	reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)
	ctx, cancel := context.WithCancel(rerunTestContext(t))
	defer cancel()

	gwClient, gw := newSGLangRerunGateway(t, func(w http.ResponseWriter, r *http.Request, _ *sglangapiv1.GenerateRequest, phase string, _ int) {
		switch phase {
		case gateway.PhasePrefill:
			writeSGLangPrefillHeaders(w, "prefill-1")
			<-r.Context().Done()
		case gateway.PhaseDecode:
			cancel()
			grpctest.WriteFrames(t, w, sglangStreamErrorItem(http.StatusInternalServerError, "after disconnect", proto.Bool(true)))
		}
	})

	_, err := runSGLangPipeline(ctx, t, gwClient, true, 2)
	require.Error(t, err)
	require.Len(t, gw.requests(gateway.PhasePrefill), 1)
	require.Len(t, gw.requests(gateway.PhaseDecode), 1)
	require.Zero(t, counterValue(t, reg, stepRerunMetric, nil))
}
