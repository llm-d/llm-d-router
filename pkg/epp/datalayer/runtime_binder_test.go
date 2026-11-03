// Copyright 2025 The llm-d Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package datalayer

import (
	"context"
	"errors"
	"testing"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"
)

type ownedEndpointSource struct {
	bound  []plugin.Plugin
	events []fwkdl.EventType
	err    error
}

func (*ownedEndpointSource) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: "owned-source", Name: "source"}
}
func (s *ownedEndpointSource) AppendExtractor(p plugin.Plugin) error {
	s.bound = append(s.bound, p)
	return s.err
}
func (s *ownedEndpointSource) NotifyEndpoint(_ context.Context, event fwkdl.EndpointEvent) (*fwkdl.EndpointEvent, error) {
	s.events = append(s.events, event.Type)
	return nil, nil //nolint:nilnil // EndpointSource uses nil to suppress lifecycle dispatch.
}

type customPayloadExtractor struct{}

func (*customPayloadExtractor) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: "custom-payload", Name: "consumer"}
}

func TestEndpointSourceOwnsCustomPayloadExtractors(t *testing.T) {
	for _, pending := range []bool{false, true} {
		t.Run(map[bool]string{false: "explicit", true: "registered"}[pending], func(t *testing.T) {
			r := NewRuntime(1)
			source := &ownedEndpointSource{}
			ext := &customPayloadExtractor{}
			cfg := &Config{Sources: []DataSourceConfig{{Plugin: source}}}
			if pending {
				require.NoError(t, r.Register(fwkdl.PendingRegistration{Owner: ext.TypedName(), SourceType: source.TypedName().Type, Extractor: ext}))
			} else {
				cfg.Sources[0].Extractors = []plugin.Plugin{ext}
			}
			require.NoError(t, r.Configure(cfg, newTestLogger(t)))
			require.Equal(t, []plugin.Plugin{ext}, source.bound)
			ep := r.NewEndpoint(t.Context(), &fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: "pod"}})
			r.ReleaseEndpoint(ep)
			require.Equal(t, []fwkdl.EventType{fwkdl.EventAddOrUpdate, fwkdl.EventDelete}, source.events)
		})
	}
}

func TestEndpointSourceBindingErrorFailsConfiguration(t *testing.T) {
	r := NewRuntime(1)
	source := &ownedEndpointSource{err: errors.New("wrong payload")}
	cfg := &Config{Sources: []DataSourceConfig{{Plugin: source, Extractors: []plugin.Plugin{&customPayloadExtractor{}}}}}
	require.ErrorContains(t, r.Configure(cfg, newTestLogger(t)), "wrong payload")
}
