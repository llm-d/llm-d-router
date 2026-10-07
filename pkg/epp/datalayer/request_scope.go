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

// Runtime confinement of a plugin's per-request attribute access, the other
// half of the endpoint confinement in endpoint_scope.go. Both read the same
// allowed-key sets, report through the same counter, and accumulate into the
// same Violations, so one invocation that misreaches on either store fails or
// is counted identically.
//
// An endpoint is confined by wrapping it, because plugins receive endpoints
// through an interface. A request is a struct, so it is confined by handing the
// plugin a shallow copy carrying the scope; the copy shares the backing store,
// so allowed writes are visible to the framework and to later plugins.

package datalayer

import (
	"fmt"

	"github.com/go-logr/logr"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// requestScope confines a plugin's request-attribute access to the keys it
// declares, on the same terms as ScopedEndpoint: a write outside Produces() is
// dropped and recorded, a read outside Consumes() resolves as absent.
type requestScope struct {
	allowedPut map[fwkplugin.DataKey]struct{}
	allowedGet map[fwkplugin.DataKey]struct{}
	reporter
}

var _ fwksched.AttributeScope = &requestScope{}

func (s *requestScope) AllowPut(key fwkplugin.DataKey) bool {
	if _, ok := s.allowedPut[key]; ok {
		return true
	}
	s.reject(metrics.DataScopeAccessWrite, fmt.Errorf(
		"plugin %q wrote undeclared request attribute %q; add it to Produces()", s.typedName.String(), key))
	return false
}

func (s *requestScope) AllowGet(key fwkplugin.DataKey) bool {
	if _, ok := s.allowedGet[key]; ok {
		return true
	}
	s.reject(metrics.DataScopeAccessRead, fmt.Errorf(
		"plugin %q read undeclared request attribute %q; add it to Consumes()", s.typedName.String(), key))
	return false
}

func (s *requestScope) Declares(key fwkplugin.DataKey) bool {
	_, ok := s.allowedGet[key]
	return ok
}

// ScopeRequest confines a plugin's request-attribute access for one invocation
// of extensionPoint. Use it where the plugin receives no endpoints; where it
// receives both, ScopeInvocation reports the two halves together.
func ScopeRequest(logger logr.Logger, extensionPoint string, plugin fwkplugin.Plugin,
	request *fwksched.InferenceRequest) (*fwksched.InferenceRequest, *Violations) {
	violations := &Violations{}
	return scopeRequest(logger, extensionPoint, plugin, request, violations), violations
}

// ScopeInvocation confines both stores a plugin reaches during one invocation
// under a single Violations, so a caller with an error return fails the request
// on the first rejected write wherever it happened.
func ScopeInvocation(logger logr.Logger, extensionPoint string, plugin fwkplugin.Plugin,
	request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) (*fwksched.InferenceRequest, []fwksched.Endpoint, *Violations) {
	violations := &Violations{}
	return scopeRequest(logger, extensionPoint, plugin, request, violations),
		scopeEndpoints(logger, extensionPoint, plugin, endpoints, violations),
		violations
}

func scopeRequest(logger logr.Logger, extensionPoint string, plugin fwkplugin.Plugin,
	request *fwksched.InferenceRequest, violations *Violations) *fwksched.InferenceRequest {
	if request == nil {
		return nil
	}
	typedName := plugin.TypedName()
	spec := scopeSpecFor(logger, plugin)
	return request.WithAttributeScope(&requestScope{
		allowedPut: spec.allowedPut,
		allowedGet: spec.allowedGet,
		reporter: reporter{
			typedName:      typedName,
			extensionPoint: extensionPoint,
			logger:         logger,
			violations:     violations,
		},
	})
}
