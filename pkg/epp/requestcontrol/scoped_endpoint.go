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

package requestcontrol

import (
	"github.com/go-logr/logr"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// scopedEndpoint wraps a scheduling.Endpoint and restricts Put/Get access to
// the DataKeys declared by the calling plugin. Unauthorized Put calls are
// skipped and logged; unauthorized Get calls return (nil, false) and are logged.
type scopedEndpoint struct {
	inner      fwksched.Endpoint
	allowedPut map[fwkplugin.DataKey]struct{}
	allowedGet map[fwkplugin.DataKey]struct{} // nil = all reads allowed
	pluginName string
	logger     logr.Logger
}

func (s *scopedEndpoint) GetMetadata() *fwkdl.EndpointMetadata {
	return s.inner.GetMetadata()
}

func (s *scopedEndpoint) GetMetrics() *fwkdl.Metrics {
	return s.inner.GetMetrics()
}

func (s *scopedEndpoint) String() string {
	return s.inner.String()
}

func (s *scopedEndpoint) Put(key fwkplugin.DataKey, value fwkdl.Cloneable) {
	if _, ok := s.allowedPut[key]; !ok {
		s.logger.Error(nil, "Plugin attempted Put with undeclared DataKey",
			"plugin", s.pluginName, "key", key.String())
		return
	}
	s.inner.Put(key, value)
}

func (s *scopedEndpoint) Get(key fwkplugin.DataKey) (fwkdl.Cloneable, bool) {
	if s.allowedGet != nil {
		if _, ok := s.allowedGet[key]; !ok {
			s.logger.Error(nil, "Plugin attempted Get with undeclared DataKey",
				"plugin", s.pluginName, "key", key.String())
			return nil, false
		}
	}
	return s.inner.Get(key)
}

func (s *scopedEndpoint) Keys() []fwkplugin.DataKey {
	return s.inner.Keys()
}

func (s *scopedEndpoint) Clone() fwkdl.AttributeMap {
	return s.inner.Clone()
}

// Inner returns the unwrapped endpoint.
func (s *scopedEndpoint) Inner() fwksched.Endpoint {
	return s.inner
}

// scopeEndpoints wraps each endpoint with access restrictions derived from the
// plugin's Produces() and Consumes() declarations.
func scopeEndpoints(logger logr.Logger, plugin fwkplugin.ProducerPlugin, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	produces := plugin.Produces()
	allowedPut := make(map[fwkplugin.DataKey]struct{}, len(produces))
	for key := range produces {
		allowedPut[key] = struct{}{}
	}

	var allowedGet map[fwkplugin.DataKey]struct{}
	if consumer, ok := plugin.(fwkplugin.ConsumerPlugin); ok {
		deps := consumer.Consumes()
		allowedGet = make(map[fwkplugin.DataKey]struct{}, len(deps.Required)+len(deps.Optional)+len(produces))
		for key := range deps.Required {
			allowedGet[key] = struct{}{}
		}
		for key := range deps.Optional {
			allowedGet[key] = struct{}{}
		}
		// A producer may read its own outputs.
		for key := range produces {
			allowedGet[key] = struct{}{}
		}
	}

	pluginName := plugin.TypedName().String()
	scoped := make([]fwksched.Endpoint, len(endpoints))
	for i, ep := range endpoints {
		scoped[i] = &scopedEndpoint{
			inner:      ep,
			allowedPut: allowedPut,
			allowedGet: allowedGet,
			pluginName: pluginName,
			logger:     logger,
		}
	}
	return scoped
}
