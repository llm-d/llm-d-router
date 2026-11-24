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

package builder

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/kv"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
	"github.com/llm-d/llm-d-router/pkg/coordinator/steps"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/attribute"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/headerstamp"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/responsereceived/headercopy"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/headerprofile"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

const (
	testMooncakeAttribute = "mooncakeHandoff"
	mooncakeWorkerA       = "10.0.0.2:8000"
	mooncakeWorkerB       = "10.0.0.3:8000"
	mooncakeEngineA       = "engine-a"
)

func mooncakePipelineConfig() config.PipelineConfig {
	return config.PipelineConfig{
		KVConnector:     kv.Mooncake,
		WorkerProtocol:  config.WorkerProtocolHTTP,
		UseOpenAIFormat: true,
		Steps: []config.StepConfig{
			{Type: steps.PrefillStepName},
			{Type: steps.DecodeStepName},
		},
	}
}

// Run the configured endpoint attribute, selected-endpoint stamp and response
// copy plugins. The gateway and the engines themselves are fixture servers.
func mooncakePrefillHeaders(t *testing.T, endpoint, servedEndpoint, engineID string, port int, includeMetadata bool) http.Header {
	t.Helper()
	ctx := context.Background()
	producer, err := attribute.Factory("", json.NewDecoder(strings.NewReader(
		fmt.Sprintf(`{"attributeKeys":[%q]}`, testMooncakeAttribute))), nil)
	require.NoError(t, err)
	stamper, err := headerstamp.Factory("", json.NewDecoder(strings.NewReader(
		fmt.Sprintf(`{"stamps":[{"attributeKey":%q,"headerName":%q}]}`, testMooncakeAttribute, routing.MooncakePrefillHandoffHeader))), nil)
	require.NoError(t, err)
	copier, err := headercopy.Factory("", json.NewDecoder(strings.NewReader(
		fmt.Sprintf(`{"headers":[%q]}`, routing.MooncakePrefillHandoffHeader))), nil)
	require.NoError(t, err)

	host, servingPort, err := net.SplitHostPort(endpoint)
	require.NoError(t, err)
	meta := &fwkdl.EndpointMetadata{Address: host, Port: servingPort, Attributes: map[string]string{}}
	if includeMetadata {
		meta.Attributes[testMooncakeAttribute] = fmt.Sprintf(
			`{"endpoint":%q,"engine_id":%q,"bootstrap_port":%d}`, endpoint, engineID, port)
	}
	server := fwkdl.NewEndpoint(meta, nil)
	require.NoError(t, producer.(fwkdl.EndpointExtractor).Extract(ctx, fwkdl.EndpointEvent{
		Type: fwkdl.EventAddOrUpdate, Endpoint: server,
	}))
	ep := fwksched.NewEndpointFromSnapshot(server.GetMetadata(), server.GetMetrics(), server.GetAttributes())
	result := &fwksched.SchedulingResult{
		PrimaryProfileName: gateway.PhasePrefill,
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			gateway.PhasePrefill: {TargetEndpoints: []fwksched.Endpoint{ep}},
		},
	}
	r := &fwksched.InferenceRequest{Headers: map[string]string{}, SchedulingResult: result}
	require.NoError(t, stamper.(fwkrh.PreRequest).PreRequest(ctx, r, result))
	resp := &fwkrh.Response{
		Headers: map[string]string{},
		ReqMetadata: map[string]any{
			metadata.DestinationEndpointNamespace: map[string]any{metadata.DestinationEndpointServedKey: servedEndpoint},
		},
	}
	headerprofile.NewHeaderProfileHandler("", "").WithPrefillProfile(gateway.PhasePrefill).ResponseHeader(ctx, r, resp, meta)
	copier.(fwkrh.ResponseHeaderProcessor).ResponseHeader(ctx, r, resp, meta)
	headers := http.Header{}
	for key, value := range resp.Headers {
		headers.Set(key, value)
	}
	return headers
}

func TestMooncakeHTTPPipeline(t *testing.T) {
	seenIDs := map[string]bool{}
	for _, tc := range []struct {
		name, endpoint, servedEndpoint, engineID, wantBootstrap        string
		port, choices                                                  int
		stream, metadata, prefillFailure, invalidBody, cancel, wantErr bool
	}{
		{name: "unary", endpoint: mooncakeWorkerA, servedEndpoint: mooncakeWorkerA, engineID: mooncakeEngineA, port: 8998, wantBootstrap: "http://10.0.0.2:8998", metadata: true},
		{name: "streaming", endpoint: mooncakeWorkerB, servedEndpoint: mooncakeWorkerB, engineID: "engine-b", port: 8999, wantBootstrap: "http://10.0.0.3:8999", metadata: true, stream: true},
		{name: "missing metadata", endpoint: mooncakeWorkerA, servedEndpoint: mooncakeWorkerA, wantErr: true},
		{name: "different served endpoint", endpoint: mooncakeWorkerA, servedEndpoint: mooncakeWorkerB, engineID: mooncakeEngineA, port: 8998, metadata: true, wantErr: true},
		{name: "invalid bootstrap port", endpoint: mooncakeWorkerA, servedEndpoint: mooncakeWorkerA, engineID: mooncakeEngineA, metadata: true, wantErr: true},
		{name: "prefill failure", endpoint: mooncakeWorkerA, servedEndpoint: mooncakeWorkerA, engineID: mooncakeEngineA, port: 8998, metadata: true, prefillFailure: true, wantErr: true},
		{name: "invalid prefill body", endpoint: mooncakeWorkerA, servedEndpoint: mooncakeWorkerA, engineID: mooncakeEngineA, port: 8998, metadata: true, invalidBody: true, wantErr: true},
		{name: "prefill cancellation", endpoint: mooncakeWorkerA, servedEndpoint: mooncakeWorkerA, engineID: mooncakeEngineA, port: 8998, metadata: true, cancel: true, wantErr: true},
		{name: "multiple completions", endpoint: mooncakeWorkerA, servedEndpoint: mooncakeWorkerA, choices: 2, wantErr: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			headers := mooncakePrefillHeaders(t, tc.endpoint, tc.servedEndpoint, tc.engineID, tc.port, tc.metadata)
			var mu sync.Mutex
			var prefill, decode map[string]any
			var decodeCalls int
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Header.Get(routing.MooncakePrefillHandoffHeader) != "" {
					t.Error("coordinator forwarded the handoff response header to a worker")
				}
				var body map[string]any
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Errorf("read fixture request: %v", err)
					w.WriteHeader(http.StatusBadRequest)
					return
				}
				params, _ := body[reqcommon.FieldKVTransferParams].(map[string]any)
				mu.Lock()
				switch r.Header.Get(reqcommon.EPPProfileHeaderKey) {
				case gateway.PhasePrefill:
					prefill = params
					mu.Unlock()
					if tc.prefillFailure {
						w.WriteHeader(http.StatusBadGateway)
						return
					}
					if tc.cancel {
						cancel()
						return
					}
					for key, values := range headers {
						for _, value := range values {
							w.Header().Add(key, value)
						}
					}
					if tc.invalidBody {
						_, _ = fmt.Fprint(w, "invalid JSON")
						return
					}
					_, _ = fmt.Fprint(w, `{}`)
				case gateway.PhaseDecode:
					decode = params
					decodeCalls++
					mu.Unlock()
					w.Header().Set(routing.MooncakePrefillHandoffHeader, "internal")
					if tc.stream {
						w.Header().Set("Content-Type", "text/event-stream")
						_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"done\"}}]}\n\ndata: [DONE]\n\n")
					} else {
						w.Header().Set("Content-Type", "application/json")
						_, _ = fmt.Fprint(w, `{"choices":[{"index":0,"message":{"role":"assistant","content":"done"}}]}`)
					}
				default:
					mu.Unlock()
					w.WriteHeader(http.StatusNotFound)
				}
			}))
			defer srv.Close()
			cfg := &config.Config{Pipeline: mooncakePipelineConfig()}
			p, err := Build(cfg, gateway.New(config.GatewayConfig{Address: srv.URL}))
			require.NoError(t, err)
			rec := httptest.NewRecorder()
			req := &pipeline.RequestContext{
				RequestID:    "test-mooncake",
				OriginalPath: reqcommon.PathChatCompletions,
				Model:        "m",
				Stream:       tc.stream,
				Body: map[string]any{
					"model": "m", "stream": tc.stream,
					"messages": []any{map[string]any{"role": "user", "content": "hello"}},
				},
				OriginalHeaders: http.Header{http.CanonicalHeaderKey(routing.MooncakePrefillHandoffHeader): {"ignored"}},
				ResponseWriter:  rec,
			}
			if tc.choices > 0 {
				req.Body[reqcommon.FieldN] = tc.choices
			}
			err = p.Execute(ctx, req)
			mu.Lock()
			prefillParams, decodeParams, calls := prefill, decode, decodeCalls
			mu.Unlock()
			if tc.wantErr {
				require.Error(t, err)
				require.Zero(t, calls)
				require.Nil(t, req.MooncakeHandoff)
				if tc.choices > 1 {
					require.Nil(t, prefillParams)
				}
				return
			}
			require.NoError(t, err)
			id, _ := prefillParams[reqcommon.FieldTransferID].(string)
			require.NotEmpty(t, id)
			require.False(t, seenIDs[id], "transfer IDs must be unique per request")
			seenIDs[id] = true
			require.Equal(t, false, prefillParams[reqcommon.FieldDoRemotePrefill])
			require.Equal(t, true, prefillParams[reqcommon.FieldDoRemoteDecode])
			require.Equal(t, map[string]any{
				reqcommon.FieldDoRemotePrefill: true, reqcommon.FieldDoRemoteDecode: false,
				reqcommon.FieldTransferID: id, reqcommon.FieldRemoteEngineID: tc.engineID,
				reqcommon.FieldRemoteBootstrapAddr: tc.wantBootstrap,
			}, decodeParams)
			require.Equal(t, 1, calls)
			require.Contains(t, rec.Body.String(), "done")
			require.Empty(t, rec.Header().Get(routing.MooncakePrefillHandoffHeader))
			require.Error(t, p.Steps()[1].Execute(context.Background(), req), "handoff may not be prepared twice")
		})
	}
}

func TestMooncakePipelineConfig(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*config.PipelineConfig)
	}{
		{"decode only", func(p *config.PipelineConfig) { p.Steps = p.Steps[1:] }},
		{"different decode connector", func(p *config.PipelineConfig) {
			p.Steps[1].Params = map[string]any{steps.ParamKVConnector: kv.NIXL}
		}},
		{"conditional decode", func(p *config.PipelineConfig) {
			p.Steps = append([]config.StepConfig{{Type: steps.ConditionalDecodeStepName}}, p.Steps...)
		}},
		{"force-stream", func(p *config.PipelineConfig) {
			p.Steps[1].Params = map[string]any{steps.ParamForceStream: true}
		}},
		{"forward token IDs", func(p *config.PipelineConfig) {
			p.Steps[1].Params = map[string]any{steps.ParamForwardPromptTokenIDs: true}
		}},
		{"tokens-in", func(p *config.PipelineConfig) { p.UseOpenAIFormat = false }},
		{"grpc workers", func(p *config.PipelineConfig) { p.WorkerProtocol = config.WorkerProtocolGRPC }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := mooncakePipelineConfig()
			tc.change(&cfg)
			require.Error(t, validatePipeline(cfg))
		})
	}
}
