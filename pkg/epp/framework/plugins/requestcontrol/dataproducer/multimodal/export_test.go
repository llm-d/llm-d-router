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

package multimodal

import (
	"maps"

	k8stypes "k8s.io/apimachinery/pkg/types"
)

func (p *Producer) cacheSnapshot() map[string]map[string]struct{} {
	p.mutex.RLock()
	defer p.mutex.RUnlock()
	snapshot := make(map[string]map[string]struct{}, p.cache.Len())
	for _, hash := range p.cache.Keys() {
		if pods, ok := p.cache.Get(hash); ok {
			snapshot[hash] = maps.Clone(pods)
		}
	}
	return snapshot
}

func (p *Producer) putCacheEntry(hash string, pods ...k8stypes.NamespacedName) {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	podSet := map[string]struct{}{}
	if existing, ok := p.cache.Get(hash); ok {
		podSet = maps.Clone(existing)
	}
	for _, pod := range pods {
		podSet[pod.String()] = struct{}{}
	}
	p.cache.Add(hash, podSet)
}
