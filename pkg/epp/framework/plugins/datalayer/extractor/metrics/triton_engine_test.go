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

package metrics_test

import (
	"context"
	"strings"
	"testing"

	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/metrics"
	sourcemetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/metrics"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/runningrequests"
)

// tritonScrape is the shape Triton serves on /metrics: nv_inference_exec_count is a
// cumulative counter (prometheus::BuildCounter in triton core src/metrics.cc), and
// nv_inference_pending_request_count is the only per-model in-flight gauge.
func tritonScrape(execCount string) string {
	return `# HELP nv_inference_exec_count Number of model executions performed (does not include cached requests)
# TYPE nv_inference_exec_count counter
nv_inference_exec_count{model="resnet50",version="1"} ` + execCount + `
# HELP nv_inference_pending_request_count Instantaneous number of pending requests awaiting execution per-model.
# TYPE nv_inference_pending_request_count gauge
nv_inference_pending_request_count{model="resnet50",version="1"} 0
`
}

func TestBuiltinTritonEngineDoesNotReadExecCountAsRunning(t *testing.T) {
	ctx := context.Background()
	plugin, err := metrics.CoreMetricsExtractorFactory("core", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	ext := plugin.(*metrics.Extractor)

	extract := func(name, execCount string) fwksched.Endpoint {
		parser := expfmt.NewTextParser(model.LegacyValidation)
		fams, err := parser.TextToMetricFamilies(strings.NewReader(tritonScrape(execCount)))
		if err != nil {
			t.Fatal(err)
		}
		meta := &fwkdl.EndpointMetadata{
			ID:     k8stypes.NamespacedName{Namespace: "default", Name: name},
			Labels: map[string]string{metrics.DefaultEngineTypeLabelKey: "triton"},
		}
		ep := fwkdl.NewEndpoint(meta, nil)
		if err := ext.Extract(ctx, fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{Payload: fams, Endpoint: ep}); err != nil {
			t.Fatalf("extract %s: %v", name, err)
		}
		return fwksched.NewEndpoint(meta, ep.GetMetrics(), nil)
	}

	// Both pods are idle: nothing pending, nothing executing. The long-lived pod has
	// simply executed more batches over its lifetime.
	longLived := extract("long-lived", "1000000")
	fresh := extract("fresh", "10")

	if got := longLived.GetMetrics().RunningRequestsSize; got != 0 {
		t.Errorf("idle Triton pod: RunningRequestsSize = %d, want 0 (README: \"Number of requests currently being processed\")", got)
	}

	scores := runningrequests.NewRunningRequestsSizeScorer().Score(ctx, &fwksched.InferenceRequest{}, []fwksched.Endpoint{longLived, fresh})
	if scores[longLived] != scores[fresh] {
		t.Errorf("two idle pods scored differently: long-lived=%.2f fresh=%.2f, want equal", scores[longLived], scores[fresh])
	}
}
