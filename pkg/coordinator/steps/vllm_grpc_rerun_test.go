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
	"context"
	"fmt"
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
	"google.golang.org/protobuf/types/known/structpb"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpctest"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/gen/vllmpb"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/ec"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/kv"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/metrics/metricstest"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

const (
	stepRerunMetric = "llm_d_coordinator_step_reruns_total"
	rerunMediaHash  = "hash-img-1"
)

var unavailableWire = grpcStatusWire{status: unavailableStatus}

// rerunGateway is a fake gateway that records the request of each worker leg.
type rerunGateway struct {
	mu   sync.Mutex
	reqs map[string][]*vllmpb.GenerateRequest
}

// newRerunGateway starts a fake gateway. For each worker leg it records the
// request, then calls respond with the phase and the attempt number of that
// phase. The attempt number starts at 1.
func newRerunGateway(t *testing.T, respond func(w http.ResponseWriter, phase string, attempt int)) (*gateway.Client, *rerunGateway) {
	t.Helper()
	gw := &rerunGateway{reqs: map[string][]*vllmpb.GenerateRequest{}}
	srv := grpctest.NewH2CServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		req := &vllmpb.GenerateRequest{}
		grpctest.ReadRequest(t, r, req)
		phase := r.Header.Get(reqcommon.EPPProfileHeaderKey)
		gw.mu.Lock()
		gw.reqs[phase] = append(gw.reqs[phase], req)
		attempt := len(gw.reqs[phase])
		gw.mu.Unlock()
		respond(w, phase, attempt)
	}))
	return gateway.New(config.GatewayConfig{Address: srv.URL}), gw
}

func (g *rerunGateway) requests(phase string) []*vllmpb.GenerateRequest {
	g.mu.Lock()
	defer g.mu.Unlock()
	return slices.Clone(g.reqs[phase])
}

// attemptHost returns the host that a successful leg reports for its attempt,
// so a test can tell which attempt produced the transfer params of a later leg.
func attemptHost(phase string, attempt int) string {
	return fmt.Sprintf("%s-%d", phase, attempt)
}

// writeLegOK answers a leg with success. An encode leg and a prefill leg
// report transfer params that name the attempt.
func writeLegOK(t *testing.T, w http.ResponseWriter, phase string, attempt int) {
	t.Helper()
	finish := &vllmpb.FinishInfo{FinishReason: vllmpb.FinishInfo_LENGTH, NumOutputTokens: 1}
	out := &vllmpb.SequenceOutput{FinishInfo: finish}
	host := attemptHost(phase, attempt)
	var err error
	switch phase {
	case gateway.PhaseEncode:
		finish.EcTransferParams, err = structpb.NewStruct(map[string]any{
			rerunMediaHash: map[string]any{"remote_host": host},
		})
	case gateway.PhasePrefill:
		finish.KvTransferParams, err = structpb.NewStruct(map[string]any{
			"do_remote_prefill": true,
			"remote_block_ids":  []any{[]any{1}},
			"remote_host":       host,
		})
	case gateway.PhaseDecode:
		finish.FinishReason = vllmpb.FinishInfo_STOP
		out.TokenIds = []uint32{42}
	default:
		t.Errorf("unexpected phase %q", phase)
	}
	require.NoError(t, err)
	grpctest.WriteFrames(t, w, &vllmpb.GenerateResponse{
		PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 4},
		Outputs:    out,
	})
}

// writeStatusInHeaders answers a leg with a trailers-only response.
func writeStatusInHeaders(w http.ResponseWriter, code codes.Code) {
	grpcframe.WriteError(w, code, "leg failed")
}

// writeStatusInTrailers answers a leg with response headers, no message, and
// the status in the trailers.
func writeStatusInTrailers(w http.ResponseWriter, code codes.Code) {
	w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
	w.WriteHeader(http.StatusOK)
	_ = http.NewResponseController(w).Flush()
	grpctest.WriteStatusTrailer(w, code, "leg failed")
}

// failFirst answers the first attempt of the failing phase with fail, and
// every other leg with success.
func failFirst(t *testing.T, failing string, fail func(http.ResponseWriter)) func(http.ResponseWriter, string, int) {
	return func(w http.ResponseWriter, phase string, attempt int) {
		if phase == failing && attempt == 1 {
			fail(w)
			return
		}
		writeLegOK(t, w, phase, attempt)
	}
}

// newRerunPipeline builds a pipeline of the named steps with the NIXL
// connectors, or of encode, prefill, and decode when it gets no name. The
// pipeline honors maxReruns re-runs for each request.
func newRerunPipeline(t *testing.T, gwClient *gateway.Client, maxReruns int, names ...string) *pipeline.Pipeline {
	t.Helper()
	if len(names) == 0 {
		names = []string{EncodeStepName, PrefillStepName, DecodeStepName}
	}
	connectors := map[string]any{ParamKVConnector: kv.NIXL, ParamECConnector: ec.NIXL}
	steps := make([]pipeline.Step, 0, len(names))
	for _, name := range names {
		var step pipeline.Step
		var err error
		switch name {
		case EncodeStepName:
			step, err = NewEncodeStep(gwClient, map[string]any{ParamECConnector: ec.NIXL})
		case PrefillStepName:
			step, err = NewPrefillStep(gwClient, connectors)
		case DecodeStepName:
			step, err = NewDecodeStep(gwClient, connectors)
		default:
			t.Fatalf("unexpected step %q", name)
		}
		require.NoError(t, err)
		steps = append(steps, step)
	}
	p := pipeline.New(steps)
	p.SetMaxStepReruns(maxReruns)
	return p
}

// newRerunRequest returns a vLLM gRPC request with one image, so the encode
// step runs. A non-nil httpReply makes it a translated HTTP request.
func newRerunRequest(t *testing.T, httpReply *pipeline.HTTPGenerateReply, stream bool) (*pipeline.RequestContext, *httptest.ResponseRecorder) {
	t.Helper()
	rec := httptest.NewRecorder()
	return &pipeline.RequestContext{
		RequestID:        "req-rerun",
		KVTransferParams: map[string]any{},
		ResponseWriter:   rec,
		Stream:           stream,
		GRPC: &pipeline.VLLMGRPCRequest{
			HTTP: httpReply,
			Req: scanTestVLLMRequest(t, &vllmpb.GenerateRequest{
				Prompt:   &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{1, 32000, 2}}},
				Stopping: &vllmpb.StoppingCriteria{MaxNewTokens: 16},
				Media: []*vllmpb.MediaItem{{
					Modality: vllmpb.Modality_MODALITY_IMAGE,
					Uuid:     rerunMediaHash,
					Source:   &vllmpb.MediaItem_DataUri{DataUri: "data:image/png;base64,AAAA"},
				}},
			}),
		},
	}, rec
}

func rerunTestContext(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	t.Cleanup(cancel)
	return ctx
}

func remoteHost(params *structpb.Struct) any {
	return params.AsMap()["remote_host"]
}

func TestVLLMGRPCRerun_DecodeUnavailableBeforeFirstToken(t *testing.T) {
	unavailable := strconv.Itoa(int(codes.Unavailable))
	modes := []struct {
		name      string
		httpReply *pipeline.HTTPGenerateReply
		stream    bool
		// wantClient checks that the client got a successful response.
		wantClient func(t *testing.T, rec *httptest.ResponseRecorder)
	}{
		{
			name:   "native gRPC stream",
			stream: true,
			wantClient: func(t *testing.T, rec *httptest.ResponseRecorder) {
				require.Empty(t, rec.Header().Get(grpcframe.HeaderStatus))
				require.Equal(t, grpcframe.StatusOK, rec.Header().Get(http.TrailerPrefix+grpcframe.HeaderStatus))
				require.Len(t, grpctest.ReadFrames(t, rec.Body), 1)
			},
		},
		{
			name: "native gRPC unary",
			wantClient: func(t *testing.T, rec *httptest.ResponseRecorder) {
				require.Empty(t, rec.Header().Get(grpcframe.HeaderStatus))
				require.Equal(t, grpcframe.StatusOK, rec.Header().Get(http.TrailerPrefix+grpcframe.HeaderStatus))
				require.Len(t, grpctest.ReadFrames(t, rec.Body), 1)
			},
		},
		{
			name:      "translated HTTP unary",
			httpReply: &pipeline.HTTPGenerateReply{},
			wantClient: func(t *testing.T, rec *httptest.ResponseRecorder) {
				require.Equal(t, http.StatusOK, rec.Code)
				require.Contains(t, rec.Body.String(), `"token_ids":[42]`)
			},
		},
		{
			name:      "translated HTTP stream",
			httpReply: &pipeline.HTTPGenerateReply{},
			stream:    true,
			wantClient: func(t *testing.T, rec *httptest.ResponseRecorder) {
				require.Equal(t, http.StatusOK, rec.Code)
				require.Contains(t, rec.Body.String(), `"token_ids":[42]`)
				require.Contains(t, rec.Body.String(), "data: [DONE]")
			},
		},
	}
	failures := []struct {
		name string
		fail func(http.ResponseWriter)
	}{
		{name: "status in headers", fail: func(w http.ResponseWriter) { writeStatusInHeaders(w, codes.Unavailable) }},
		{name: "status in trailers", fail: func(w http.ResponseWriter) { writeStatusInTrailers(w, codes.Unavailable) }},
	}
	for _, mode := range modes {
		for _, failure := range failures {
			t.Run(mode.name+", "+failure.name, func(t *testing.T) {
				reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)
				gwClient, gw := newRerunGateway(t, failFirst(t, gateway.PhaseDecode, failure.fail))
				reqCtx, rec := newRerunRequest(t, mode.httpReply, mode.stream)

				require.NoError(t, newRerunPipeline(t, gwClient, 1).Execute(rerunTestContext(t), reqCtx))

				require.Len(t, gw.requests(gateway.PhaseEncode), 1, "the re-run starts at prefill")
				require.Len(t, gw.requests(gateway.PhasePrefill), 2)
				decodes := gw.requests(gateway.PhaseDecode)
				require.Len(t, decodes, 2)
				require.Equal(t, attemptHost(gateway.PhasePrefill, 2), remoteHost(decodes[1].GetKv().GetKvTransferParams()),
					"the second decode loads the KV blocks of the second prefill")

				mode.wantClient(t, rec)
				require.NotContains(t, rec.Header().Values(grpcframe.HeaderStatus), unavailable)
				require.InDelta(t, 1.0, counterValue(t, reg, stepRerunMetric, map[string]string{"step": DecodeStepName}), 1e-9)
				require.Zero(t, counterValue(t, reg, stepErrorMetric, map[string]string{"step": DecodeStepName}))
			})
		}
	}
}

func TestVLLMGRPCRerun_PrefillUnavailable(t *testing.T) {
	failures := []struct {
		name string
		fail func(http.ResponseWriter)
	}{
		{name: "status in headers", fail: func(w http.ResponseWriter) { writeStatusInHeaders(w, codes.Unavailable) }},
		{name: "status in trailers", fail: func(w http.ResponseWriter) { writeStatusInTrailers(w, codes.Unavailable) }},
	}
	for _, failure := range failures {
		t.Run(failure.name, func(t *testing.T) {
			reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)
			gwClient, gw := newRerunGateway(t, failFirst(t, gateway.PhasePrefill, failure.fail))
			reqCtx, rec := newRerunRequest(t, nil, true)

			require.NoError(t, newRerunPipeline(t, gwClient, 1).Execute(rerunTestContext(t), reqCtx))

			require.Len(t, gw.requests(gateway.PhaseEncode), 2, "the re-run starts at encode")
			prefills := gw.requests(gateway.PhasePrefill)
			require.Len(t, prefills, 2)
			require.Equal(t,
				map[string]any{rerunMediaHash: map[string]any{"remote_host": attemptHost(gateway.PhaseEncode, 2)}},
				prefills[1].GetKv().GetEcTransferParams().AsMap(),
				"the second prefill loads only the encoder cache of the second encode")
			decodes := gw.requests(gateway.PhaseDecode)
			require.Len(t, decodes, 1)
			require.Equal(t, attemptHost(gateway.PhasePrefill, 2), remoteHost(decodes[0].GetKv().GetKvTransferParams()))

			require.Equal(t, grpcframe.StatusOK, rec.Header().Get(http.TrailerPrefix+grpcframe.HeaderStatus))
			require.InDelta(t, 1.0, counterValue(t, reg, stepRerunMetric, map[string]string{"step": PrefillStepName}), 1e-9)
		})
	}
}

// In a pipeline without prefill, decode loads the encoder cache of encode.
func TestVLLMGRPCRerun_DecodeUnavailableWithoutPrefill(t *testing.T) {
	gwClient, gw := newRerunGateway(t, failFirst(t, gateway.PhaseDecode, func(w http.ResponseWriter) {
		writeStatusInHeaders(w, codes.Unavailable)
	}))
	reqCtx, rec := newRerunRequest(t, nil, true)

	p := newRerunPipeline(t, gwClient, 1, EncodeStepName, DecodeStepName)
	require.NoError(t, p.Execute(rerunTestContext(t), reqCtx))

	require.Len(t, gw.requests(gateway.PhaseEncode), 2, "the re-run starts at encode")
	decodes := gw.requests(gateway.PhaseDecode)
	require.Len(t, decodes, 2)
	require.Equal(t,
		map[string]any{rerunMediaHash: map[string]any{"remote_host": attemptHost(gateway.PhaseEncode, 2)}},
		decodes[1].GetKv().GetEcTransferParams().AsMap(),
		"the second decode loads only the encoder cache of the second encode")
	require.Equal(t, grpcframe.StatusOK, rec.Header().Get(http.TrailerPrefix+grpcframe.HeaderStatus))
}

// Without max_step_reruns, and after the last re-run, the client gets the
// UNAVAILABLE status of the worker as before.
func TestVLLMGRPCRerun_ReturnsUnavailableWithoutRerunLeft(t *testing.T) {
	cases := []struct {
		name        string
		maxReruns   int
		wantDecodes int
	}{
		{name: "re-runs disabled", maxReruns: 0, wantDecodes: 1},
		{name: "last re-run used", maxReruns: 1, wantDecodes: 2},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)
			gwClient, gw := newRerunGateway(t, func(w http.ResponseWriter, phase string, attempt int) {
				if phase == gateway.PhaseDecode {
					writeStatusInHeaders(w, codes.Unavailable)
					return
				}
				writeLegOK(t, w, phase, attempt)
			})
			reqCtx, rec := newRerunRequest(t, nil, true)

			err := newRerunPipeline(t, gwClient, tc.maxReruns).Execute(rerunTestContext(t), reqCtx)

			var streamed *pipeline.UpstreamStreamedError
			require.ErrorAs(t, err, &streamed, "the decode step wrote the status to the client")
			require.Equal(t, strconv.Itoa(int(codes.Unavailable)), rec.Header().Get(grpcframe.HeaderStatus))
			require.Len(t, gw.requests(gateway.PhaseDecode), tc.wantDecodes)
			require.Len(t, gw.requests(gateway.PhasePrefill), tc.wantDecodes)
			require.InDelta(t, float64(tc.maxReruns), counterValue(t, reg, stepRerunMetric, map[string]string{"step": DecodeStepName}), 1e-9)
			require.InDelta(t, 1.0, counterValue(t, reg, stepErrorMetric, map[string]string{"step": DecodeStepName}), 1e-9)
		})
	}
}

func TestVLLMGRPCRerun_NoRerunForOtherFailures(t *testing.T) {
	cases := []struct {
		name       string
		fail       func(t *testing.T, w http.ResponseWriter)
		wantHeader string // grpc-status in the client headers
		wantTrail  string // grpc-status in the client trailers
	}{
		{
			name:       "status other than UNAVAILABLE",
			fail:       func(_ *testing.T, w http.ResponseWriter) { writeStatusInHeaders(w, codes.Internal) },
			wantHeader: strconv.Itoa(int(codes.Internal)),
		},
		{
			name: "UNAVAILABLE after the first output token",
			fail: func(t *testing.T, w http.ResponseWriter) {
				w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
				w.WriteHeader(http.StatusOK)
				_, err := w.Write(grpctest.Frame(t, &vllmpb.GenerateResponse{
					Outputs: &vllmpb.SequenceOutput{TokenIds: []uint32{42}},
				}))
				require.NoError(t, err)
				grpctest.WriteStatusTrailer(w, codes.Unavailable, "leg failed")
			},
			wantTrail: strconv.Itoa(int(codes.Unavailable)),
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			reg := metricstest.NewRegistry(t, coordmetrics.Register, coordmetrics.Reset)
			gwClient, gw := newRerunGateway(t, failFirst(t, gateway.PhaseDecode, func(w http.ResponseWriter) { tc.fail(t, w) }))
			reqCtx, rec := newRerunRequest(t, nil, true)

			err := newRerunPipeline(t, gwClient, 1).Execute(rerunTestContext(t), reqCtx)

			var streamed *pipeline.UpstreamStreamedError
			require.ErrorAs(t, err, &streamed)
			require.Len(t, gw.requests(gateway.PhasePrefill), 1)
			require.Len(t, gw.requests(gateway.PhaseDecode), 1)
			require.Equal(t, tc.wantHeader, rec.Header().Get(grpcframe.HeaderStatus))
			require.Equal(t, tc.wantTrail, rec.Header().Get(http.TrailerPrefix+grpcframe.HeaderStatus))
			require.Zero(t, counterValue(t, reg, stepRerunMetric, nil))
		})
	}
}

// A prefill that runs again must not leave the counts of its first attempt,
// or decode reports them for a prefill that did not measure them.
func TestVLLMGRPCRerun_PrefillRerunReplacesCounts(t *testing.T) {
	kvParams, err := structpb.NewStruct(map[string]any{"do_remote_prefill": true})
	require.NoError(t, err)
	gwClient, gw := newRerunGateway(t, func(w http.ResponseWriter, phase string, attempt int) {
		switch {
		case phase == gateway.PhasePrefill && attempt == 1:
			grpctest.WriteFrames(t, w, &vllmpb.GenerateResponse{
				PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 4},
				Outputs: &vllmpb.SequenceOutput{FinishInfo: &vllmpb.FinishInfo{
					NumOutputTokens: 1, NumCachedTokens: proto.Uint32(3), KvTransferParams: kvParams,
				}},
			})
		case phase == gateway.PhasePrefill:
			grpctest.WriteFrames(t, w, &vllmpb.GenerateResponse{
				Outputs: &vllmpb.SequenceOutput{FinishInfo: &vllmpb.FinishInfo{NumOutputTokens: 1, KvTransferParams: kvParams}},
			})
		case phase == gateway.PhaseDecode && attempt == 1:
			writeStatusInHeaders(w, codes.Unavailable)
		case phase == gateway.PhaseDecode:
			grpctest.WriteFrames(t, w, &vllmpb.GenerateResponse{
				Outputs: &vllmpb.SequenceOutput{TokenIds: []uint32{42}, FinishInfo: &vllmpb.FinishInfo{
					FinishReason: vllmpb.FinishInfo_STOP, NumOutputTokens: 1, NumCachedTokens: proto.Uint32(7),
				}},
			})
		default:
			writeLegOK(t, w, phase, attempt)
		}
	})
	reqCtx, rec := newRerunRequest(t, nil, true)

	require.NoError(t, newRerunPipeline(t, gwClient, 1).Execute(rerunTestContext(t), reqCtx))

	require.Len(t, gw.requests(gateway.PhasePrefill), 2)
	require.Nil(t, reqCtx.PrefillCachedTokens)
	require.Nil(t, reqCtx.GRPC.(*pipeline.VLLMGRPCRequest).PrefillPromptTokens)
	frames := grpctest.ReadFrames(t, rec.Body)
	require.Len(t, frames, 1)
	var got vllmpb.GenerateResponse
	grpctest.Unmarshal(t, frames[0], &got)
	require.Equal(t, uint32(7), got.GetOutputs().GetFinishInfo().GetNumCachedTokens(),
		"the client gets the count of decode, not the count of the first prefill")
}

// funcStep runs fn as a pipeline step.
type funcStep func(reqCtx *pipeline.RequestContext)

func (funcStep) Name() string { return "func" }

func (f funcStep) Execute(_ context.Context, reqCtx *pipeline.RequestContext) error {
	f(reqCtx)
	return nil
}

func TestRerunOnUnavailable(t *testing.T) {
	vllm := func(encodeRan, prefillRan bool) pipeline.GRPCRequest {
		return &pipeline.VLLMGRPCRequest{EncodeRan: encodeRan, PrefillRan: prefillRan}
	}
	cases := []struct {
		name      string
		step      string
		status    grpcStatusWire
		grpc      pipeline.GRPCRequest
		maxReruns int
		wantFrom  string // "" expects no re-run
	}{
		{name: "decode restarts at prefill", step: DecodeStepName, status: unavailableWire, grpc: vllm(true, true), maxReruns: 1, wantFrom: PrefillStepName},
		{name: "decode without prefill restarts at encode", step: DecodeStepName, status: unavailableWire, grpc: vllm(true, false), maxReruns: 1, wantFrom: EncodeStepName},
		{name: "decode alone runs again", step: DecodeStepName, status: unavailableWire, grpc: vllm(false, false), maxReruns: 1, wantFrom: DecodeStepName},
		{name: "prefill restarts at encode", step: PrefillStepName, status: unavailableWire, grpc: vllm(true, false), maxReruns: 1, wantFrom: EncodeStepName},
		{name: "second prefill restarts at encode", step: PrefillStepName, status: unavailableWire, grpc: vllm(true, true), maxReruns: 1, wantFrom: EncodeStepName},
		{name: "prefill alone runs again", step: PrefillStepName, status: unavailableWire, grpc: vllm(false, false), maxReruns: 1, wantFrom: PrefillStepName},
		{name: "encode does not run again", step: EncodeStepName, status: unavailableWire, grpc: vllm(false, false), maxReruns: 1},
		{name: "conditional-decode does not run again", step: ConditionalDecodeStepName, status: unavailableWire, grpc: vllm(false, false), maxReruns: 1},
		{name: "other status", step: DecodeStepName, status: grpcStatusWire{status: failedPreconditionStatus}, grpc: vllm(true, true), maxReruns: 1},
		{name: "no re-run left", step: DecodeStepName, status: unavailableWire, grpc: vllm(true, true)},
		{name: "SGLang request", step: DecodeStepName, status: unavailableWire, grpc: &pipeline.SGLangGRPCRequest{}, maxReruns: 1},
		{name: "HTTP worker protocol", step: DecodeStepName, status: unavailableWire, maxReruns: 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var got error
			// Only a request inside Pipeline.Execute has re-runs left.
			p := pipeline.New([]pipeline.Step{funcStep(func(reqCtx *pipeline.RequestContext) {
				got = rerunOnUnavailable(tc.step, reqCtx, tc.status)
			})})
			p.SetMaxStepReruns(tc.maxReruns)
			require.NoError(t, p.Execute(context.Background(), &pipeline.RequestContext{GRPC: tc.grpc}))

			if tc.wantFrom == "" {
				require.NoError(t, got)
				return
			}
			var rerun *pipeline.StepRerunError
			require.ErrorAs(t, got, &rerun)
			require.Equal(t, tc.step, rerun.Step)
			require.Equal(t, tc.wantFrom, rerun.From)
			var upstream *pipeline.UpstreamError
			require.ErrorAs(t, got, &upstream)
			require.Equal(t, http.StatusBadGateway, upstream.StatusCode)
		})
	}
}
