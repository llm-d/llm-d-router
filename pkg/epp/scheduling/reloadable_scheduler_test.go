/*
Copyright 2026 The Kubernetes Authors.

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

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/single"
)

type fixedPicker struct {
	name    string
	started chan struct{}
	release chan struct{}
}

func (p *fixedPicker) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "fixed-picker", Name: p.name}
}

func (p *fixedPicker) Pick(_ context.Context, candidates []*fwksched.ScoredEndpoint) *fwksched.ProfileRunResult {
	if p.started != nil {
		close(p.started)
		<-p.release
	}
	for _, candidate := range candidates {
		if candidate.GetMetadata().ID.Name == p.name {
			return &fwksched.ProfileRunResult{TargetEndpoints: []fwksched.Endpoint{candidate.Endpoint}}
		}
	}
	return nil
}

func schedulerWithPicker(picker fwksched.Picker) *Scheduler {
	profile := NewSchedulerProfile().WithPicker(picker)
	return NewSchedulerWithConfig(NewSchedulerConfig(single.NewSingleProfileHandler(), map[string]fwksched.SchedulerProfile{"default": profile}))
}

func TestReloadableSchedulerPinsGenerationForSchedule(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	reloadable := NewReloadableScheduler(schedulerWithPicker(&fixedPicker{name: "old", started: started, release: release}))
	endpoints := []fwksched.Endpoint{
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: "old"}}, nil, nil),
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: "new"}}, nil, nil),
	}
	type outcome struct {
		result *fwksched.SchedulingResult
		err    error
	}
	result := make(chan outcome, 1)
	go func() {
		selected, err := reloadable.Schedule(context.Background(), &fwksched.InferenceRequest{RequestID: "request"}, endpoints)
		result <- outcome{result: selected, err: err}
	}()

	<-started
	require.Equal(t, uint64(2), reloadable.Replace(schedulerWithPicker(&fixedPicker{name: "new"})))
	close(release)
	previous := <-result
	require.NoError(t, previous.err)
	require.Equal(t, "old", previous.result.ProfileResults[previous.result.PrimaryProfileName].TargetEndpoints[0].GetMetadata().ID.Name)
	current, err := reloadable.Schedule(context.Background(), &fwksched.InferenceRequest{RequestID: "next"}, endpoints)
	require.NoError(t, err)
	require.Equal(t, "new", current.ProfileResults[current.PrimaryProfileName].TargetEndpoints[0].GetMetadata().ID.Name)
	require.Equal(t, uint64(2), reloadable.Generation())
}
