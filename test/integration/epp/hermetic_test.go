/*
Copyright 2025 The Kubernetes Authors.

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

// Package epp contains integration tests for the Endpoint Picker extension.
package epp

import (
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/testing/protocmp"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	fwkepp "github.com/llm-d/llm-d-router/test/framework/epp"
	eppharness "github.com/llm-d/llm-d-router/test/framework/epp/harness"
)

const (
	modelSheddable                  = "sql-lora-sheddable"
	modelSheddableTarget            = "sql-lora-1fdg3"
	modelDirect                     = "direct-model"
	modelToBeWritten                = "model-to-be-rewritten"
	modelAfterRewrite               = "rewritten-model"
	inferenceObjectiveWithPriority4 = "inference-objective-with-priority-4"
)

func TestMain(m *testing.M) { os.Exit(eppharness.Run(m)) }

func TestFullDuplexStreamed_KubeInferenceObjectiveRequest(t *testing.T) {
	// executionModes defines the permutations of EPP deployment modes to test.
	executionModes := []struct {
		name     string
		mode     eppharness.RunMode
		strategy eppharness.StandaloneStrategy
	}{
		{name: "Standard", mode: eppharness.ModeStandard},
		{name: "Standalone-NoCRD", mode: eppharness.ModeStandalone, strategy: eppharness.StrategyNoCRD},
		{name: "Standalone-WithCRD", mode: eppharness.ModeStandalone, strategy: eppharness.StrategyWithCRD},
	}

	for _, executionMode := range executionModes {
		t.Run(executionMode.name, func(t *testing.T) {
			// Determine if we are running in the standalone mode without CRDs
			isNoCRD := executionMode.mode == eppharness.ModeStandalone && executionMode.strategy == eppharness.StrategyNoCRD

			// Helper function to override priority to 0 when in NoCRD mode
			prio := func(p int) int {
				if isNoCRD {
					return 0
				}
				return p
			}

			hermeticTests := []testCase{
				{
					name:     "select lora despite higher kv cache (affinity)",
					requests: fwkepp.ReqLLM(eppharness.Logger(), "test3", modelSQLLora, modelSQLLoraTarget),
					pods: []eppharness.PodState{
						eppharness.P(0, 10, 0.2, "foo", "bar"),
						eppharness.P(1, 10, 0.4, "foo", modelSQLLoraTarget), // Winner (Affinity overrides KV)
						eppharness.P(2, 10, 0.3, "foo"),
					},
					wantResponses: ExpectRouteTo("192.168.1.2:8000", modelSQLLoraTarget, "test3"),
					wantMetrics: map[string]string{
						"inference_objective_request_total": eppharness.CleanMetric(eppharness.MetricReqTotal(modelSQLLora, modelSQLLoraTarget, prio(2))),
					},
				},
				{
					name: "passthrough parser success",
					configText: `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
  - type: queue-scorer
  - type: kv-cache-utilization-scorer
  - type: passthrough-parser
  - type: mock-metrics-source
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: queue-scorer
      - pluginRef: kv-cache-utilization-scorer
requestHandler:
  parsers:
  - pluginRef: passthrough-parser
dataLayer:
  sources:
  - pluginRef: mock-metrics-source
`,
					requests: fwkepp.ReqRaw(
						map[string]string{
							"hi":                         "mom",
							reqcommon.RequestIDHeaderKey: "test-request-id",
							metadata.ObjectiveKey:        modelMyModel, // With passthrough parser, the objective key can still be used to specify priority.
						},
						"passthrough-parser",
					),
					pods: []eppharness.PodState{
						eppharness.P(0, 3, 0.2),
						eppharness.P(1, 0, 0.1), // Winner
						eppharness.P(2, 10, 0.2),
					},
					wantResponses: ExpectPassthroughRouteTo("192.168.1.2:8000", []byte("passthrough-parser")),
					wantMetrics: map[string]string{
						"inference_objective_request_total": eppharness.CleanMetric(eppharness.MetricReqTotal("", "", prio(2))),
						"inference_pool_ready_pods":         eppharness.CleanMetric(eppharness.MetricReadyPods(3)),
					},
				},
				{
					name:     "do not shed requests by default",
					requests: fwkepp.ReqLLM(eppharness.Logger(), "test4", modelSQLLora, modelSQLLoraTarget),
					pods: []eppharness.PodState{
						eppharness.P(0, 6, 0.2, "foo", "bar", modelSQLLoraTarget), // Winner (Lowest saturated)
						eppharness.P(1, 0, 0.85, "foo"),
						eppharness.P(2, 10, 0.9, "foo"),
					},
					wantResponses: ExpectRouteTo("192.168.1.1:8000", modelSQLLoraTarget, "test4"),
					wantMetrics: map[string]string{
						"inference_objective_request_total": eppharness.CleanMetric(eppharness.MetricReqTotal(modelSQLLora, modelSQLLoraTarget, prio(2))),
					},
				},

				// --- Error Handling & Edge Cases ----
				{
					name: "invalid json body",
					requests: fwkepp.ReqRaw(
						map[string]string{"hi": "mom"},
						"no healthy upstream",
					),
					pods: []eppharness.PodState{
						eppharness.P(0, 0, 0.2, "foo", "bar"),
					},
					wantResponses: ExpectReject(
						envoyTypePb.StatusCode_BadRequest,
						"inference error: BadRequest - error extracting request body: invalid character 'o' in literal null (expecting 'u')",
					),
				},
				{
					name: "split body across chunks",
					requests: fwkepp.ReqRaw(
						map[string]string{
							"hi":                         "mom",
							metadata.ObjectiveKey:        modelSheddable,
							metadata.ModelNameRewriteKey: modelSheddableTarget,
							reqcommon.RequestIDHeaderKey: "test-request-id",
						},
						`{"max_tokens":100,"model":"sql-lo`,
						`ra-sheddable","prompt":"test6","temperature":0}`,
					),
					pods: []eppharness.PodState{
						eppharness.P(0, 4, 0.2, "foo", "bar", modelSheddableTarget),
						eppharness.P(1, 4, 0.85, "foo", modelSheddableTarget),
					},
					wantResponses: ExpectRouteTo("192.168.1.1:8000", modelSheddableTarget, "test6"),
					wantMetrics: map[string]string{
						"inference_objective_request_total": eppharness.CleanMetric(eppharness.MetricReqTotal(modelSheddable, modelSheddableTarget, prio(0))),
					},
				},
				{
					name: "images edits: multipart body split across chunks, routed unchanged",
					requests: fwkepp.ReqRaw(
						map[string]string{
							":path":                      "/v1/images/edits",
							"content-type":               "multipart/form-data; boundary=" + imagesEditsBoundary,
							reqcommon.RequestIDHeaderKey: "test-request-id",
						},
						imagesEditsBody[:40],
						imagesEditsBody[40:],
					),
					pods: []eppharness.PodState{
						eppharness.P(0, 3, 0.2),
						eppharness.P(1, 0, 0.1), // Winner (Low Queue, Low KV)
						eppharness.P(2, 10, 0.2),
					},
					wantResponses: expectImagesEditsRouteTo("192.168.1.2:8000"),
				},
				{
					name: "images edits: non-multipart content-type rejected",
					requests: fwkepp.ReqRaw(
						map[string]string{
							":path":        "/v1/images/edits",
							"content-type": "application/json",
						},
						`{"model":"my-model","prompt":"edit"}`,
					),
					wantResponses: ExpectReject(envoyTypePb.StatusCode_BadRequest,
						"inference error: BadRequest - images edits request must have a multipart/form-data content-type"),
				},
				{
					name:     "no backend pods available",
					requests: fwkepp.ReqHeaderOnly(map[string]string{"content-type": "application/json"}),
					pods:     nil,
					wantResponses: ExpectReject(envoyTypePb.StatusCode_InternalServerError,
						"inference error: Internal - no pods available in datastore"),
				},
				{
					name: "request missing model field",
					requests: fwkepp.ReqRaw(
						map[string]string{"content-type": "application/json"},
						`{"prompt":"hello world"}`,
					),
					wantResponses: ExpectReject(envoyTypePb.StatusCode_BadRequest,
						"inference error: BadRequest - model not found in request body"),
				},

				// --- Subsetting & Metadata ---
				{
					name: "subsetting: select best from subset",
					// Only pods in the subset list are eligible.
					requests: ReqSubset("test2", modelSQLLora, modelSQLLoraTarget,
						"192.168.1.1:8000", "192.168.1.2:8000", "192.168.1.3:8000"),
					pods: []eppharness.PodState{
						eppharness.P(0, 0, 0.2, "foo"),
						eppharness.P(1, 0, 0.1, "foo", modelSQLLoraTarget), // Winner (Low Queue + Matches Subset)
						eppharness.P(2, 10, 0.2, "foo"),
					},
					wantResponses: ExpectRouteTo("192.168.1.2:8000", modelSQLLoraTarget, "test2"),
				},
				{
					name:     "subsetting: partial match",
					requests: ReqSubset("test2", modelSQLLora, modelSQLLoraTarget, "192.168.1.3:8000"),
					pods: []eppharness.PodState{
						eppharness.P(0, 0, 0.2, "foo"),
						eppharness.P(1, 0, 0.1, "foo", modelSQLLoraTarget),
						eppharness.P(2, 10, 0.2, "foo"), // Winner (Matches Subset, despite load)
					},
					wantResponses: ExpectRouteTo("192.168.1.3:8000", modelSQLLoraTarget, "test2"),
				},
				{
					name:     "subsetting: no pods match",
					requests: ReqSubset("test2", modelSQLLora, modelSQLLoraTarget, "192.168.1.99:8000"),
					pods: []eppharness.PodState{
						eppharness.P(0, 0, 0.2, "foo"),
						eppharness.P(1, 0, 0.1, "foo", modelSQLLoraTarget),
					},
					wantResponses: ExpectRejectWithDropReason(envoyTypePb.StatusCode_ServiceUnavailable,
						"inference error: ServiceUnavailable - failed to find endpoint candidates for serving the request",
						errcommon.RequestDroppedReasonNoEndpoints),
				},

				// --- Request Modification (Passthrough & Rewrite) ---
				{
					name: "passthrough: model not in objectives",
					requests: fwkepp.ReqRaw(
						map[string]string{
							"hi":                         "mom",
							metadata.ObjectiveKey:        modelDirect,
							metadata.ModelNameRewriteKey: modelDirect,
							reqcommon.RequestIDHeaderKey: "test-request-id",
						},
						`{"max_tokens":100,"model":"direct-`,
						`model","prompt":"test6","temperature":0}`,
					),
					pods: []eppharness.PodState{
						eppharness.P(0, 4, 0.2, "foo", "bar", modelSheddableTarget),
					},
					wantResponses: ExpectRouteTo("192.168.1.1:8000", modelDirect, "test6"),
					wantMetrics: map[string]string{
						"inference_objective_request_total": eppharness.CleanMetric(eppharness.MetricReqTotal(modelDirect, modelDirect, prio(2))),
					},
				},
				{
					name:     "rewrite request model",
					requests: fwkepp.ReqLLM(eppharness.Logger(), "test-rewrite", modelToBeWritten, modelToBeWritten),
					pods: []eppharness.PodState{
						eppharness.P(0, 0, 0.1, "foo", modelAfterRewrite),
					},
					wantResponses: ExpectRouteTo("192.168.1.1:8000", modelAfterRewrite, "test-rewrite"),
					wantMetrics: map[string]string{
						"inference_objective_request_total": eppharness.CleanMetric(eppharness.MetricReqTotal(modelToBeWritten, modelAfterRewrite, prio(0))),
					},
					requiresCRDs: true,
				},
				{
					name: "protocol: simple GET (header only)",
					requests: fwkepp.ReqHeaderOnly(map[string]string{
						"content-type": "text/event-stream",
						"status":       "200",
					}),
					pods:          []eppharness.PodState{eppharness.P(0, 0, 0, "foo")},
					wantResponses: nil,
				},

				// --- Response Processing (Buffering & Streaming) ---
				{
					name: "response buffering: multi-chunk JSON",
					requests: ReqResponseOnly(
						map[string]string{"content-type": "application/json"},
						`{"max_tokens":100,"model":"sql-lo`,
						`ra-sheddable","prompt":"test6","temperature":0}`,
					),
					pods: []eppharness.PodState{eppharness.P(0, 4, 0.2, modelSheddableTarget)},
					wantResponses: ExpectBufferResp(
						fmt.Sprintf(`{"max_tokens":100,"model":%q,"prompt":"test6","temperature":0}`, modelSheddable),
						"application/json"),
				},
				{
					name: "response buffering: invalid JSON",
					requests: ReqResponseOnly(
						map[string]string{"content-type": "application/json"},
						"no healthy upstream",
					),
					pods:          []eppharness.PodState{eppharness.P(0, 4, 0.2, modelSheddableTarget)},
					wantResponses: ExpectBufferResp("no healthy upstream", "application/json"),
				},
				{
					name: "response buffering: empty EOS chunk (JSON)",
					requests: ReqResponseOnly(
						map[string]string{"content-type": "application/json"},
						`{"max_tokens":100,"model":"sql-lora-sheddable","prompt":"test6","temperature":0}`,
						"",
					),
					pods: []eppharness.PodState{eppharness.P(0, 4, 0.2, modelSheddableTarget)},
					wantResponses: ExpectBufferResp(
						fmt.Sprintf(`{"max_tokens":100,"model":%q,"prompt":"test6","temperature":0}`, modelSheddable),
						"application/json"),
				},
				{
					name: "response streaming: SSE token counting",
					requests: ReqResponseOnly(
						map[string]string{"content-type": "text/event-stream", "status": "200"},
						// Chunk 1: Simulate a standard data chunk.
						`data: {}`,
						// Chunk 2: Usage data + DONE signal.
						`data: {"usage":{"prompt_tokens":7,"total_tokens":17,"completion_tokens":10}}`+"\n"+`data: [DONE]`,
						"", // EndOfStream
					),
					pods:         []eppharness.PodState{eppharness.P(0, 4, 0.2, modelSheddableTarget)},
					waitForModel: modelSheddable,
					wantResponses: ExpectStreamResp(
						`data: {}`,
						`data: {"usage":{"prompt_tokens":7,"total_tokens":17,"completion_tokens":10}}`+"\n"+`data: [DONE]`,
						"",
					),
					// Labels are empty because we skipped the Request phase.
					wantMetrics: map[string]string{
						"inference_objective_input_tokens": eppharness.CleanMetric(`
              # HELP inference_objective_input_tokens [ALPHA] [Deprecated: Use llm_d_epp_request_input_tokens] Inference objective input token count distribution for requests in each model.
              # TYPE inference_objective_input_tokens histogram
              inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="1"} 0
              inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="8"} 1
              inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="16"} 1
              inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="32"} 1
              inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="64"} 1
              inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="128"} 1
              inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="256"} 1
              inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="512"} 1
              inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="1024"} 1
              inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="2048"} 1
              inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="4096"} 1
              inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="8192"} 1
              inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="16384"} 1
              inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="32768"} 1
              inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="65536"} 1
              inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="131072"} 1
              inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="262144"} 1
              inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="524288"} 1
              inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="1.048576e+06"} 1
              inference_objective_input_tokens_bucket{model_name="",target_model_name="",le="+Inf"} 1
              inference_objective_input_tokens_sum{model_name="",target_model_name=""} 7
              inference_objective_input_tokens_count{model_name="",target_model_name=""} 1
              `),
					},
				},
			}
			tests := append(commonTestCases(prio), hermeticTests...)

			for _, tc := range tests {
				t.Run(tc.name, func(t *testing.T) {
					if isNoCRD && tc.requiresCRDs {
						t.Skipf("Skipping test %q: requires CRDs, but running in standalone without crd executionMode", tc.name)
					}

					ctx := t.Context()

					var h *eppharness.TestHarness
					var harnessOpts []eppharness.HarnessOption

					if len(tc.wantSpans) > 0 {
						harnessOpts = append(harnessOpts, eppharness.WithTracing())
					}

					if executionMode.mode == eppharness.ModeStandalone {
						harnessOpts = append(harnessOpts, eppharness.WithStandaloneMode(executionMode.strategy))
					} else {
						harnessOpts = append(harnessOpts, eppharness.WithStandardMode())
					}

					if tc.configText != "" {
						harnessOpts = append(harnessOpts, eppharness.WithConfigText(tc.configText))
					}

					h = eppharness.NewTestHarness(ctx, t, harnessOpts...)

					if executionMode.mode == eppharness.ModeStandard || executionMode.strategy == eppharness.StrategyWithCRD {
						h = h.WithBaseResources()
					}

					// In standalone mode without crd, we cannot wait for an Objective CRD to sync as it doesn't exist.
					// We only wait for Pod discovery.
					modelToSync := tc.waitForModel
					if modelToSync == "" {
						modelToSync = modelMyModel
					}

					h.WithPods(tc.pods).WaitForSync(len(tc.pods), modelToSync)
					if len(tc.pods) > 0 {
						h.WaitForReadyPodsMetric(len(tc.pods))
					}

					responses, err := fwkepp.StreamedRequest(t, h.Client, tc.requests, len(tc.wantResponses))
					require.NoError(t, err)

					if diff := cmp.Diff(tc.wantResponses, responses,
						protocmp.Transform(),
						protocmp.SortRepeated(func(a, b *configPb.HeaderValueOption) bool {
							return a.GetHeader().GetKey() < b.GetHeader().GetKey()
						}),
					); diff != "" {
						t.Errorf("Response mismatch (-want +got): %v", diff)
					}

					if len(tc.wantMetrics) > 0 {
						h.ExpectMetrics(tc.wantMetrics)
					}
					if len(tc.wantSpans) > 0 {
						// Close the stream so the server finishes processing and ends the root span
						_ = h.Client.CloseSend()

						assert.Eventually(t, func() bool {
							spans := h.GetSpans()
							recordedSpans := make(map[string]bool)
							for _, s := range spans {
								recordedSpans[s.Name] = true
							}

							for _, want := range tc.wantSpans {
								if !recordedSpans[want] {
									return false
								}
							}
							return true
						}, 5*time.Second, 50*time.Millisecond, "Expected spans %v not found", tc.wantSpans)
					}
				})
			}
		})
	}
}

const imagesEditsBoundary = "imagesEditsBoundary"

// imagesEditsBody is a multipart /v1/images/edits request carrying form fields and a file part.
var imagesEditsBody = strings.Join([]string{
	"--" + imagesEditsBoundary,
	`Content-Disposition: form-data; name="model"`,
	"",
	modelMyModel,
	"--" + imagesEditsBoundary,
	`Content-Disposition: form-data; name="prompt"`,
	"",
	"make the sky blue",
	"--" + imagesEditsBoundary,
	`Content-Disposition: form-data; name="image"; filename="cat.png"`,
	"Content-Type: image/png",
	"",
	"raw-png-bytes",
	"--" + imagesEditsBoundary + "--",
	"",
}, "\r\n")

// expectImagesEditsRouteTo asserts the multipart request is routed to endpoint with its body
// forwarded unchanged: multipart payloads are raw bytes, so no model rewrite is applied.
func expectImagesEditsRouteTo(endpoint string) []*extProcPb.ProcessingResponse {
	return fwkepp.NewRequestBufferedResponse(
		endpoint,
		[]byte(imagesEditsBody),
		&configPb.HeaderValueOption{Header: &configPb.HeaderValue{
			Key: ":path", RawValue: []byte("/v1/images/edits"),
		}},
		&configPb.HeaderValueOption{Header: &configPb.HeaderValue{
			Key: "content-type", RawValue: []byte("multipart/form-data; boundary=" + imagesEditsBoundary),
		}},
		&configPb.HeaderValueOption{Header: &configPb.HeaderValue{
			Key: reqcommon.RequestIDHeaderKey, RawValue: []byte("test-request-id"),
		}},
	)
}
