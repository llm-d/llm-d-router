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

	"github.com/stretchr/testify/require"

	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// A bound session's pod gets 1.0, every other candidate 0.0.
func TestScorePinsBoundSession(t *testing.T) {
	a := newTestAgent(testConfig())
	podA := schedEndpoint("pod-a", 0, 0)
	podB := schedEndpoint("pod-b", 0, 0)
	runTurn(t, a, "s1", podA, 400, 300)

	scores := a.Score(context.Background(), newRequest("s1", 400), []fwksched.Endpoint{podA, podB})
	require.Equal(t, 1.0, scores[podA])
	require.Equal(t, 0.0, scores[podB])
}

// Unknown sessions and anonymous requests get no scores.
func TestScoreAbstains(t *testing.T) {
	a := newTestAgent(testConfig())
	podA := schedEndpoint("pod-a", 0, 0)

	require.Nil(t, a.Score(context.Background(), newRequest("unknown", 400), []fwksched.Endpoint{podA}))
	require.Nil(t, a.Score(context.Background(), newRequest("", 400), []fwksched.Endpoint{podA}))
}

// When the bound pod is not among the candidates (it left the pool or a
// filter removed it), the scorer abstains rather than zeroing everything.
func TestScoreAbstainsWhenBoundPodMissing(t *testing.T) {
	a := newTestAgent(testConfig())
	podA := schedEndpoint("pod-a", 0, 0)
	podB := schedEndpoint("pod-b", 0, 0)
	runTurn(t, a, "s1", podA, 400, 300)

	require.Nil(t, a.Score(context.Background(), newRequest("s1", 400), []fwksched.Endpoint{podB}))
}

func TestScorerCategory(t *testing.T) {
	require.Equal(t, fwksched.Affinity, newTestAgent(testConfig()).Category())
}
