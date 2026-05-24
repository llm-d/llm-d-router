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

package scheduling

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

func TestScorerCategory_Values(t *testing.T) {
	assert.Equal(t, ScorerCategory("Affinity"), Affinity)
	assert.Equal(t, ScorerCategory("Distribution"), Distribution)
	assert.Equal(t, ScorerCategory("Balance"), Balance)
}

func TestScorerCategory_Distinct(t *testing.T) {
	cats := map[ScorerCategory]struct{}{
		Affinity:     {},
		Distribution: {},
		Balance:      {},
	}
	assert.Len(t, cats, 3, "scorer categories must be distinct")
}

// fakeBase satisfies plugin.Plugin so we can build mock filter/scorer/picker/profile-handler
// instances in tests below.
type fakeBase struct {
	tn plugin.TypedName
}

func (f *fakeBase) TypedName() plugin.TypedName { return f.tn }

type fakeFilter struct {
	fakeBase
	called bool
}

func (f *fakeFilter) Filter(_ context.Context, _ *CycleState, _ *InferenceRequest, pods []Endpoint) []Endpoint {
	f.called = true
	return pods
}

type fakeScorer struct {
	fakeBase
	cat ScorerCategory
}

func (s *fakeScorer) Category() ScorerCategory { return s.cat }
func (s *fakeScorer) Score(_ context.Context, _ *CycleState, _ *InferenceRequest, pods []Endpoint) map[Endpoint]float64 {
	out := make(map[Endpoint]float64, len(pods))
	for _, p := range pods {
		out[p] = 1
	}
	return out
}

type fakePicker struct {
	fakeBase
}

func (p *fakePicker) Pick(_ context.Context, _ *CycleState, scored []*ScoredEndpoint) *ProfileRunResult {
	res := &ProfileRunResult{}
	for _, s := range scored {
		res.TargetEndpoints = append(res.TargetEndpoints, s.Endpoint)
	}
	return res
}

type fakeProfileHandler struct {
	fakeBase
}

func (h *fakeProfileHandler) Pick(_ context.Context, _ *CycleState, _ *InferenceRequest,
	profiles map[string]SchedulerProfile, _ map[string]*ProfileRunResult) map[string]SchedulerProfile {
	return profiles
}

func (h *fakeProfileHandler) ProcessResults(_ context.Context, _ *CycleState, _ *InferenceRequest,
	profileResults map[string]*ProfileRunResult) (*SchedulingResult, error) {
	primary := ""
	for k := range profileResults {
		primary = k
		break
	}
	return &SchedulingResult{ProfileResults: profileResults, PrimaryProfileName: primary}, nil
}

func TestFilter_InterfaceCompliance(t *testing.T) {
	f := &fakeFilter{fakeBase: fakeBase{tn: plugin.TypedName{Type: "filter", Name: "f1"}}}
	var _ Filter = f
	var _ plugin.Plugin = f

	pods := []Endpoint{NewEndpoint(newTestMetadata("p1"), newTestMetrics(), nil)}
	got := f.Filter(context.Background(), NewCycleState(), &InferenceRequest{}, pods)
	assert.True(t, f.called)
	assert.Equal(t, pods, got)
}

func TestScorer_InterfaceCompliance(t *testing.T) {
	s := &fakeScorer{
		fakeBase: fakeBase{tn: plugin.TypedName{Type: "scorer", Name: "s1"}},
		cat:      Balance,
	}
	var _ Scorer = s
	var _ plugin.Plugin = s

	pods := []Endpoint{NewEndpoint(newTestMetadata("p1"), newTestMetrics(), nil)}
	scores := s.Score(context.Background(), NewCycleState(), &InferenceRequest{}, pods)
	assert.Len(t, scores, 1)
	assert.Equal(t, Balance, s.Category())
}

func TestPicker_InterfaceCompliance(t *testing.T) {
	p := &fakePicker{fakeBase: fakeBase{tn: plugin.TypedName{Type: "picker", Name: "p1"}}}
	var _ Picker = p
	var _ plugin.Plugin = p

	ep := NewEndpoint(newTestMetadata("p1"), newTestMetrics(), nil)
	res := p.Pick(context.Background(), NewCycleState(), []*ScoredEndpoint{{Endpoint: ep, Score: 1}})
	assert.NotNil(t, res)
	assert.Len(t, res.TargetEndpoints, 1)
}

func TestProfileHandler_InterfaceCompliance(t *testing.T) {
	h := &fakeProfileHandler{fakeBase: fakeBase{tn: plugin.TypedName{Type: "profile", Name: "ph1"}}}
	var _ ProfileHandler = h
	var _ plugin.Plugin = h

	profiles := map[string]SchedulerProfile{"primary": nil}
	picked := h.Pick(context.Background(), NewCycleState(), &InferenceRequest{}, profiles, nil)
	assert.Len(t, picked, 1)

	results := map[string]*ProfileRunResult{"primary": {}}
	sched, err := h.ProcessResults(context.Background(), NewCycleState(), &InferenceRequest{}, results)
	assert.NoError(t, err)
	assert.Equal(t, "primary", sched.PrimaryProfileName)
	assert.Same(t, results["primary"], sched.ProfileResults["primary"])
}
