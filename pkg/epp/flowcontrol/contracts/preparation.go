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

package contracts

import (
	"context"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// PreparedRequest owns the data used by both the dispatch check and scheduling.
type PreparedRequest struct {
	Request   *scheduling.InferenceRequest
	Endpoints []scheduling.Endpoint
	// Candidates records the located set before screeners select a subset.
	Candidates map[datalayer.ID]*datalayer.EndpointMetadata
	// WithoutPrefix bounds the cost when a cached-prefix estimate expires.
	WithoutPrefix    *PreparedRequest
	Err              error
	Filter           AdmissionFilter
	ProfileEndpoints map[string][]scheduling.Endpoint
	DataKeys         []plugin.DataKey
	Locate           func(context.Context) []datalayer.Endpoint
	ProducerErrors   map[plugin.TypedName]error
}

type AdmissionFilter func(context.Context, *scheduling.InferenceRequest, []scheduling.Endpoint,
	func(string, []scheduling.Endpoint) []scheduling.Endpoint) (bool, map[string][]scheduling.Endpoint)

// PrepareRequest runs outside the processor loop. Invocations for one queued
// request are serialized; cancellation may return the caller before it finishes.
type PrepareRequest func(context.Context, func(*PreparedRequest) bool, func(*PreparedRequest)) *PreparedRequest
