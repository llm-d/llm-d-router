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

package requestcontrol

import (
	"context"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/handlers"
)

var (
	scopeDeclaredKey   = fwkplugin.NewDataKey("declared", "some-producer")
	scopeUndeclaredKey = fwkplugin.NewDataKey("undeclared", "other-producer")
)

// declaringPlugin implements every director extension point by handing its
// arguments to visit, under the declarations it is constructed with.
type declaringPlugin struct {
	name     string
	produces map[fwkplugin.DataKey]any
	consumes map[fwkplugin.DataKey]any
	visit    func(request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint)
}

func (p *declaringPlugin) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "declaring-plugin", Name: p.name}
}

func (p *declaringPlugin) Produces() map[fwkplugin.DataKey]any { return p.produces }

func (p *declaringPlugin) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{Optional: p.consumes}
}

func (p *declaringPlugin) RequestHeader(_ context.Context, request *fwksched.InferenceRequest) error {
	p.visit(request, nil)
	return nil
}

func (p *declaringPlugin) Screen(_ context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	p.visit(request, endpoints)
	return endpoints
}

func (p *declaringPlugin) Admit(_ context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) error {
	p.visit(request, endpoints)
	return nil
}

func (p *declaringPlugin) ResponseHeader(_ context.Context, request *fwksched.InferenceRequest, _ *fwkrc.Response, _ *fwkdl.EndpointMetadata) {
	p.visit(request, nil)
}

func (p *declaringPlugin) ResponseBody(_ context.Context, request *fwksched.InferenceRequest, _ *fwkrc.Response, _ *fwkdl.EndpointMetadata) {
	p.visit(request, nil)
}

// scopeReads records what a declaringPlugin found on each call.
type scopeReads struct {
	mu                 sync.Mutex
	calls              int
	requestDeclared    []bool
	requestUndeclared  []bool
	endpointDeclared   []bool
	endpointUndeclared []bool
}

func (r *scopeReads) record(request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.calls++
	_, ok := request.GetAttribute(scopeDeclaredKey)
	r.requestDeclared = append(r.requestDeclared, ok)
	_, ok = request.GetAttribute(scopeUndeclaredKey)
	r.requestUndeclared = append(r.requestUndeclared, ok)
	for _, endpoint := range endpoints {
		_, ok = endpoint.Get(scopeDeclaredKey)
		r.endpointDeclared = append(r.endpointDeclared, ok)
		_, ok = endpoint.Get(scopeUndeclaredKey)
		r.endpointUndeclared = append(r.endpointUndeclared, ok)
	}
}

func scopeTestEndpoint(name string) fwksched.Endpoint {
	attrs := fwkdl.NewAttributes()
	attrs.Put(scopeDeclaredKey, testCloneable("declared"))
	attrs.Put(scopeUndeclaredKey, testCloneable("secret"))
	return fwksched.NewEndpoint(&fwkdl.EndpointMetadata{
		ID:   types.NamespacedName{Namespace: "default", Name: name},
		Name: name,
	}, &fwkdl.Metrics{}, attrs)
}

// Every director extension point confines a plugin to its declarations: a key
// the plugin consumes resolves, a key it does not reads as absent, on the
// request and on every endpoint the point hands it.
func TestDirector_ScopesPluginsToTheirDeclarations(t *testing.T) {
	newResponseContext := func(request *fwksched.InferenceRequest) *handlers.RequestContext {
		reqCtx := newResponseBodyTestRequestContext("scope-request")
		reqCtx.SchedulingRequest = request
		return reqCtx
	}

	tests := []struct {
		name          string
		configure     func(*Config, *declaringPlugin) *Config
		run           func(context.Context, *Director, *fwksched.InferenceRequest, []fwksched.Endpoint)
		wantCalls     int
		withEndpoints bool
	}{
		{
			name: fwkrc.RequestHeaderExtensionPoint,
			configure: func(c *Config, p *declaringPlugin) *Config {
				return c.WithRequestHeaderPlugins(p)
			},
			run: func(ctx context.Context, d *Director, request *fwksched.InferenceRequest, _ []fwksched.Endpoint) {
				require.NoError(t, d.runRequestHeaderProcessors(ctx, request))
			},
			wantCalls: 1,
		},
		{
			name: fwkrc.ScreenerExtensionPoint,
			configure: func(c *Config, p *declaringPlugin) *Config {
				return c.WithScreeners(p)
			},
			run: func(ctx context.Context, d *Director, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) {
				d.runScreeners(ctx, request, endpoints)
			},
			wantCalls:     1,
			withEndpoints: true,
		},
		{
			name: fwkrc.AdmissionExtensionPoint,
			configure: func(c *Config, p *declaringPlugin) *Config {
				return c.WithAdmissionPlugins(p)
			},
			run: func(ctx context.Context, d *Director, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) {
				require.NoError(t, d.runAdmissionPlugins(ctx, request, endpoints))
			},
			wantCalls:     1,
			withEndpoints: true,
		},
		{
			name: fwkrc.ResponseReceivedExtensionPoint,
			configure: func(c *Config, p *declaringPlugin) *Config {
				return c.WithResponseReceivedPlugins(p)
			},
			run: func(ctx context.Context, d *Director, request *fwksched.InferenceRequest, _ []fwksched.Endpoint) {
				reqCtx := newResponseContext(request)
				reqCtx.Response.Headers = map[string]string{}
				d.HandleResponseHeader(ctx, reqCtx)
			},
			wantCalls: 1,
		},
		{
			name: fwkrc.ResponseStreamingExtensionPoint + " streamed",
			configure: func(c *Config, p *declaringPlugin) *Config {
				return c.WithResponseStreamingPlugins(p)
			},
			run: func(ctx context.Context, d *Director, request *fwksched.InferenceRequest, _ []fwksched.Endpoint) {
				reqCtx := newResponseContext(request)
				d.HandleResponseBody(ctx, reqCtx, false)
				d.HandleResponseBody(ctx, reqCtx, false)
				d.HandleResponseBody(ctx, reqCtx, true)
			},
			wantCalls: 3,
		},
		{
			name: fwkrc.ResponseStreamingExtensionPoint + " single body",
			configure: func(c *Config, p *declaringPlugin) *Config {
				return c.WithResponseStreamingPlugins(p)
			},
			run: func(ctx context.Context, d *Director, request *fwksched.InferenceRequest, _ []fwksched.Endpoint) {
				d.HandleResponseBody(ctx, newResponseContext(request), true)
			},
			wantCalls: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			reads := &scopeReads{}
			plugin := &declaringPlugin{
				name:     tt.name,
				consumes: map[fwkplugin.DataKey]any{scopeDeclaredKey: nil},
				visit:    reads.record,
			}
			datalayer.RegisterScopeSpecs([]fwkplugin.Plugin{plugin})
			director := &Director{requestControlPlugins: *tt.configure(NewConfig(), plugin)}

			request := &fwksched.InferenceRequest{RequestID: "scope-request"}
			request.PutAttribute(scopeDeclaredKey, "declared")
			request.PutAttribute(scopeUndeclaredKey, "secret")
			endpoints := []fwksched.Endpoint{scopeTestEndpoint("a"), scopeTestEndpoint("b")}

			tt.run(ctx, director, request, endpoints)

			reads.mu.Lock()
			defer reads.mu.Unlock()
			require.Equal(t, tt.wantCalls, reads.calls)
			for call := range tt.wantCalls {
				assert.True(t, reads.requestDeclared[call], "call %d: a consumed request attribute must resolve", call)
				assert.False(t, reads.requestUndeclared[call], "call %d: an undeclared request attribute must read as absent", call)
			}
			if !tt.withEndpoints {
				return
			}
			require.Len(t, reads.endpointDeclared, len(endpoints))
			for i := range endpoints {
				assert.True(t, reads.endpointDeclared[i], "endpoint %d: a consumed attribute must resolve", i)
				assert.False(t, reads.endpointUndeclared[i], "endpoint %d: an undeclared attribute must read as absent", i)
			}
		})
	}
}

// RequestHeader has an error return, so a write outside Produces() fails the
// request, as it does at PreRequest, and a declared write reaches the store
// the director reads.
func TestRunRequestHeaderProcessors_EnforcesProducesDeclaration(t *testing.T) {
	newWriter := func(name string, writes fwkplugin.DataKey) *declaringPlugin {
		return &declaringPlugin{
			name:     name,
			produces: map[fwkplugin.DataKey]any{scopeDeclaredKey: ""},
			visit: func(request *fwksched.InferenceRequest, _ []fwksched.Endpoint) {
				request.PutAttribute(writes, "value")
			},
		}
	}

	t.Run("declared write reaches the request", func(t *testing.T) {
		plugin := newWriter("declared-writer", scopeDeclaredKey)
		datalayer.RegisterScopeSpecs([]fwkplugin.Plugin{plugin})
		director := &Director{requestControlPlugins: *NewConfig().WithRequestHeaderPlugins(plugin)}
		request := &fwksched.InferenceRequest{}

		require.NoError(t, director.runRequestHeaderProcessors(context.Background(), request))

		value, ok := request.GetAttribute(scopeDeclaredKey)
		assert.True(t, ok)
		assert.Equal(t, "value", value)
	})

	t.Run("undeclared write fails the request and is dropped", func(t *testing.T) {
		plugin := newWriter("undeclared-writer", scopeUndeclaredKey)
		datalayer.RegisterScopeSpecs([]fwkplugin.Plugin{plugin})
		director := &Director{requestControlPlugins: *NewConfig().WithRequestHeaderPlugins(plugin)}
		request := &fwksched.InferenceRequest{}

		err := director.runRequestHeaderProcessors(context.Background(), request)

		require.Error(t, err)
		assert.Equal(t, errcommon.Internal, errcommon.CanonicalCode(err), "an untyped error would reach Envoy as a stream error")
		assert.Contains(t, err.Error(), "add it to Produces()")
		_, ok := request.GetAttribute(scopeUndeclaredKey)
		assert.False(t, ok)
	})
}

// A screener hands back the endpoints it was given. The director intersects
// screener results by endpoint identity, so the scope must be removed before
// the intersection or every candidate would be dropped.
func TestRunScreeners_ReturnsUnderlyingEndpoints(t *testing.T) {
	endpoints := []fwksched.Endpoint{scopeTestEndpoint("a"), scopeTestEndpoint("b")}
	plugin := &declaringPlugin{name: "pass-through", visit: func(*fwksched.InferenceRequest, []fwksched.Endpoint) {}}
	datalayer.RegisterScopeSpecs([]fwkplugin.Plugin{plugin})
	director := &Director{requestControlPlugins: *NewConfig().WithScreeners(plugin)}

	result := director.runScreeners(context.Background(), &fwksched.InferenceRequest{}, endpoints)

	require.Len(t, result, len(endpoints))
	for i := range endpoints {
		assert.Same(t, endpoints[i], result[i])
	}
}
