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

package datalayer

import (
	"context"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

// FakeEndpointFactory is a configurable EndpointFactory for tests. Each hook overrides the
// corresponding method. A nil NewEndpointFn creates a plain endpoint with fresh metrics; nil
// UpdateEndpointFn and ReleaseEndpointFn hooks are no-ops.
type FakeEndpointFactory struct {
	NewEndpointFn     func(ctx context.Context, meta *fwkdl.EndpointMetadata) fwkdl.Endpoint
	UpdateEndpointFn  func(ctx context.Context, ep fwkdl.Endpoint)
	ReleaseEndpointFn func(ep fwkdl.Endpoint)
}

var _ EndpointFactory = (*FakeEndpointFactory)(nil)

func (f *FakeEndpointFactory) NewEndpoint(ctx context.Context, meta *fwkdl.EndpointMetadata) fwkdl.Endpoint {
	if f.NewEndpointFn != nil {
		return f.NewEndpointFn(ctx, meta)
	}
	return fwkdl.NewEndpoint(meta, fwkdl.NewMetrics())
}

func (f *FakeEndpointFactory) UpdateEndpoint(ctx context.Context, ep fwkdl.Endpoint) {
	if f.UpdateEndpointFn != nil {
		f.UpdateEndpointFn(ctx, ep)
	}
}

func (f *FakeEndpointFactory) ReleaseEndpoint(ep fwkdl.Endpoint) {
	if f.ReleaseEndpointFn != nil {
		f.ReleaseEndpointFn(ep)
	}
}
