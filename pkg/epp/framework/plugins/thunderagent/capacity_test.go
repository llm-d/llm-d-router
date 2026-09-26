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

package thunderagent

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

func endpointWithCacheInfo(name string, blockSize, numBlocks int) fwkdl.Endpoint {
	return fwkdl.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: types.NamespacedName{Namespace: "default", Name: name}},
		&fwkdl.Metrics{CacheBlockSize: blockSize, CacheNumBlocks: numBlocks})
}

func TestEndpointCapacity(t *testing.T) {
	a := newTestAgent(testConfig())

	t.Run("real capacity from cache_config_info", func(t *testing.T) {
		capTokens, source := a.endpointCapacity(endpointWithCacheInfo("pod1", 16, 8192))
		assert.Equal(t, capacityReal, source)
		assert.Equal(t, float64(16*8192), capTokens)
	})

	t.Run("fallback when attributes are missing", func(t *testing.T) {
		capTokens, source := a.endpointCapacity(endpointWithCacheInfo("pod1", 0, 0))
		assert.Equal(t, capacityFallback, source)
		assert.Equal(t, float64(1000), capTokens)
	})

	t.Run("mixed pool resolves per pod", func(t *testing.T) {
		big, _ := a.endpointCapacity(endpointWithCacheInfo("big", 16, 16384))
		small, _ := a.endpointCapacity(endpointWithCacheInfo("small", 16, 4096))
		assert.Equal(t, 4.0, big/small)
	})

	t.Run("requireRealCapacity refuses the fallback", func(t *testing.T) {
		cfg := testConfig()
		cfg.RequireRealCapacity = true
		strict := newTestAgent(cfg)
		capTokens, source := strict.endpointCapacity(endpointWithCacheInfo("pod1", 0, 0))
		assert.Equal(t, capacityFallback, source)
		assert.Zero(t, capTokens)
	})
}
