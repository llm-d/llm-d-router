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

package datalayer

import (
	"fmt"
	"testing"

	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// The scheduler runs every filter and scorer over the whole candidate set, so
// the per-plugin cost here is multiplied by the plugin count on each request.
func benchEndpoints(count int) []fwksched.Endpoint {
	endpoints := make([]fwksched.Endpoint, count)
	for i := range endpoints {
		attrs := fwkdl.NewAttributes()
		attrs.Put(consumedKey, cloneableStr("v"))
		endpoints[i] = fwksched.NewEndpoint(
			&fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: fmt.Sprintf("ep-%d", i)}},
			&fwkdl.Metrics{}, attrs)
	}
	return endpoints
}

func benchPlugin() fwkplugin.Plugin {
	p := &producerConsumerPlugin{}
	p.produces = map[fwkplugin.DataKey]any{producedKey: nil}
	p.consumes = &fwkplugin.DataDependencies{Optional: map[fwkplugin.DataKey]any{consumedKey: nil}}
	RegisterScopeSpecs([]fwkplugin.Plugin{p})
	return p
}

func BenchmarkScope(b *testing.B) {
	for _, count := range []int{10, 100} {
		b.Run(fmt.Sprintf("endpoints=%d", count), func(b *testing.B) {
			endpoints, plugin := benchEndpoints(count), benchPlugin()
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				scoped, _ := Scope(testLogger(), "test-extension-point", plugin, endpoints)
				_ = Unscope(scoped)
			}
		})
	}
}

func BenchmarkUnscopeScores(b *testing.B) {
	for _, count := range []int{10, 100} {
		b.Run(fmt.Sprintf("endpoints=%d", count), func(b *testing.B) {
			endpoints, plugin := benchEndpoints(count), benchPlugin()
			scoped, _ := Scope(testLogger(), "test-extension-point", plugin, endpoints)
			scores := make(map[fwksched.Endpoint]float64, len(scoped))
			for i, endpoint := range scoped {
				scores[endpoint] = float64(i)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = UnscopeScores(scores)
			}
		})
	}
}
