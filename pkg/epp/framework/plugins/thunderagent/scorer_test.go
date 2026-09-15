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

package thunderagent

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

func TestScoreAbstainsForAnonymousRequests(t *testing.T) {
	a := newTestAgent(testConfig())
	endpoints := newTestEndpoints("pod1", "pod2")

	assert.Nil(t, a.Score(context.Background(), &fwksched.InferenceRequest{}, endpoints))
	assert.Nil(t, a.Score(context.Background(), &fwksched.InferenceRequest{FairnessID: metadata.DefaultFairnessID}, endpoints))
}

func TestScoreStickyForBoundProgram(t *testing.T) {
	a := newTestAgent(testConfig())
	endpoints := newTestEndpoints("pod1", "pod2")

	seedProgram(t, a, "program-a", endpoints[0], 800)

	scores := a.Score(context.Background(), newRequest("program-a", 0), endpoints)
	assert.Equal(t, 1.0, scores[endpoints[0]], "bound pod wins regardless of load")
	assert.Equal(t, 0.0, scores[endpoints[1]])
}

func TestScoreUnboundProgramPrefersLeastLoaded(t *testing.T) {
	a := newTestAgent(testConfig())
	endpoints := newTestEndpoints("pod1", "pod2")

	seedProgram(t, a, "program-a", endpoints[0], 800)

	scores := a.Score(context.Background(), newRequest("program-b", 0), endpoints)
	assert.InDelta(t, 0.2, scores[endpoints[0]], 0.0001, "pod1 carries program-a's 800 tokens")
	assert.InDelta(t, 1.0, scores[endpoints[1]], 0.0001)
}

func TestScoreLoadCapsAtCapacity(t *testing.T) {
	a := newTestAgent(testConfig())
	endpoints := newTestEndpoints("pod1")

	seedProgram(t, a, "program-a", endpoints[0], 5000)

	scores := a.Score(context.Background(), newRequest("program-b", 0), endpoints)
	assert.InDelta(t, 0.0, scores[endpoints[0]], 0.0001)
}

// Re-placement after overload: a scheduling filter (e.g. utilization-detector)
// removes the bound pod from the candidates, the sticky branch misses, the
// load branch places the program elsewhere, and PreRequest rebinds it.
func TestScoreReplacesWhenBoundPodFiltered(t *testing.T) {
	a := newTestAgent(testConfig())
	all := newTestEndpoints("pod1", "pod2")

	seedProgram(t, a, "program-a", all[0], 400)

	candidates := all[1:]
	scores := a.Score(context.Background(), newRequest("program-a", 0), candidates)
	assert.InDelta(t, 1.0, scores[candidates[0]], 0.0001, "unloaded pod2 is the placement target")

	req := newRequest("program-a", 0)
	require.NoError(t, a.PreRequest(context.Background(), req, schedulingResultFor(candidates[0])))

	scores = a.Score(context.Background(), newRequest("program-a", 0), all)
	assert.Equal(t, 1.0, scores[all[1]], "the program is rebound to pod2")
	assert.Equal(t, 0.0, scores[all[0]])
}
