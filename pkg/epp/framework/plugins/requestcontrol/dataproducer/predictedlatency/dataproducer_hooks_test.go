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

package predictedlatency

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"

	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrlatency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/latency"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

func TestProducesConsumes(t *testing.T) {
	pl := NewPredictedLatency(DefaultConfig, nil)

	produces := pl.Produces()
	assert.Contains(t, produces, attrlatency.LatencyPredictionInfoKey)

	consumes := pl.Consumes()
	assert.Contains(t, consumes, attrprefix.PrefixCacheMatchInfoKey)
}

// TestProduce_CancelledContextDoesNotPublish verifies that when the
// director's Produce window has already closed (ctx cancelled), the plugin
// does not publish the SLO context into the ttlcache. If it did, ResponseBody
// would later find the context and issue an orphan decrement against counters
// PreRequest never incremented — draining prefillTokensInFlight negative.
func TestProduce_CancelledContextDoesNotPublish(t *testing.T) {
	cfg := DefaultConfig
	cfg.PredictInProduce = false // skip the prediction sidecar path
	pl := NewPredictedLatency(cfg, nil)

	request := createTestInferenceRequest("cancel-test", 0, 0)
	endpoint := createTestEndpoint("pod-a", 0.1, 0, 0)

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel before the plugin runs

	err := pl.Produce(ctx, request, []fwksched.Endpoint{endpoint})
	assert.ErrorIs(t, err, context.Canceled, "should propagate ctx.Err() on cancelled context")

	_, getErr := pl.getPredictedLatencyContextForRequest(request)
	assert.Error(t, getErr, "SLO context should NOT be stored when ctx is cancelled")
}

// TestProduce_LivesContextPublishes is the positive control for the
// cancellation test above: with a live context, the fast-path store still fires.
func TestProduce_LiveContextPublishes(t *testing.T) {
	cfg := DefaultConfig
	cfg.PredictInProduce = false
	pl := NewPredictedLatency(cfg, nil)

	request := createTestInferenceRequest("live-test", 0, 0)
	endpoint := createTestEndpoint("pod-a", 0.1, 0, 0)

	err := pl.Produce(context.Background(), request, []fwksched.Endpoint{endpoint})
	assert.NoError(t, err)

	_, getErr := pl.getPredictedLatencyContextForRequest(request)
	assert.NoError(t, getErr, "SLO context should be stored on the happy path")
}
