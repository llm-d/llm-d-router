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

package sessionaffinity

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrsession "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/session"
)

// SessionAffinityType is the type of the SessionAffinity scorer.
const SessionAffinityType = "session-affinity-scorer"

var (
	_ scheduling.Scorer     = &SessionAffinity{}
	_ plugin.ConsumerPlugin = &SessionAffinity{}
)

// Config holds the scorer's configurable parameters.
type Config struct {
	// SessionIDProducerName names the session-id-producer instance whose
	// BoundEndpoint attribute this scorer consumes. Empty selects the default
	// producer.
	SessionIDProducerName string `json:"sessionIDProducerName,omitempty"`
}

// Factory builds a SessionAffinity scorer from raw plugin parameters.
func Factory(name string, parameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	config := Config{}
	if parameters != nil {
		if err := parameters.Decode(&config); err != nil {
			return nil, fmt.Errorf("invalid session affinity config: %w", err)
		}
	}
	// Production paths (auto-instantiation in
	// pkg/epp/datalayer/data_graph.go and explicit configloader) always
	// supply a non-nil handle; the nil branch is the test-only path that
	// constructs a plugin without registering metrics. In production,
	// handle.Metrics() may itself be nil when no recorder is configured,
	// in which case RegisterAffinityMetrics returns an error and factory
	// construction fails fast.
	if handle != nil {
		if err := attrsession.RegisterAffinityMetrics(handle.Metrics()); err != nil {
			return nil, err
		}
	}
	return NewSessionAffinity(config.SessionIDProducerName).WithName(name), nil
}

// NewSessionAffinity returns a scorer that consumes BoundEndpoint from the
// named session-id-producer (empty for the default).
func NewSessionAffinity(sessionIDProducerName string) *SessionAffinity {
	return &SessionAffinity{
		typedName: plugin.TypedName{Type: SessionAffinityType},
		bindingDK: attrsession.BoundEndpointDataKey.WithNonEmptyProducerName(sessionIDProducerName),
	}
}

// SessionAffinity is a routing scorer that gives the endpoint bound to the
// request's session a score of 1.0 and assigns 0.0 to every other candidate.
// When the request has no binding, every candidate receives 0.0.
type SessionAffinity struct {
	typedName plugin.TypedName
	bindingDK plugin.DataKey
}

// TypedName returns the typed name of the plugin.
func (s *SessionAffinity) TypedName() plugin.TypedName {
	return s.typedName
}

// WithName sets the name of the plugin.
func (s *SessionAffinity) WithName(name string) *SessionAffinity {
	s.typedName.Name = name
	return s
}

// Category returns the preference the scorer applies when scoring candidate endpoints.
func (s *SessionAffinity) Category() scheduling.ScorerCategory {
	return scheduling.Affinity
}

// Score gives the bound endpoint 1.0 and every other endpoint 0.0. With no
// binding present the result is all zeros, which lets other scorers decide.
func (s *SessionAffinity) Score(_ context.Context, request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) map[scheduling.Endpoint]float64 {
	scoredEndpoints := make(map[scheduling.Endpoint]float64, len(endpoints))
	target, hasTarget := s.readBinding(request)

	matched := false
	for _, endpoint := range endpoints {
		score := 0.0
		if hasTarget && attrsession.EndpointBoundForm(endpoint) == target {
			score = 1.0
			matched = true
		}
		scoredEndpoints[endpoint] = score
	}
	if hasTarget && !matched && len(endpoints) > 0 {
		attrsession.RecordStaleBinding(s.typedName.Name, s.typedName.Type)
	}
	return scoredEndpoints
}

// Consumes declares the BoundEndpoint attribute key read by this scorer.
// The key is Required: a soft scorer with no producer always returns zeros,
// which is a silent misconfiguration; the framework fails fast at init
// time so the operator notices.
func (s *SessionAffinity) Consumes() plugin.DataDependencies {
	return plugin.DataDependencies{
		Required: map[plugin.DataKey]any{
			s.bindingDK: attrsession.BoundEndpoint(""),
		},
	}
}

func (s *SessionAffinity) readBinding(request *scheduling.InferenceRequest) (attrsession.BoundEndpoint, bool) {
	if request == nil {
		return "", false
	}
	bound, ok := scheduling.ReadRequestAttribute[attrsession.BoundEndpoint](request, s.bindingDK.String())
	if !ok || bound == "" {
		return "", false
	}
	return bound, true
}
