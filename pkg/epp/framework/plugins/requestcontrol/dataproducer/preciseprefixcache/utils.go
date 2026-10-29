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

package preciseprefixcache

import (
	"fmt"
	"strconv"

	"k8s.io/apimachinery/pkg/util/sets"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// indexPodIdentifier returns the identifier the KV-block index stores md's blocks
// under. With pod discovery each data-parallel rank publishes to its own subscriber,
// so a rank endpoint, which shares its pod's address and port, adds its rank.
func (p *Producer) indexPodIdentifier(md *fwkdl.EndpointMetadata) string {
	id := fmt.Sprintf("%s:%s", md.Address, md.Port)
	if md.RanksPerPort > 1 && p.kvEventsConfig != nil && p.kvEventsConfig.DiscoverPods {
		id += "/rank-" + strconv.Itoa(md.RankIndex)
	}
	return id
}

// extractEndpointSet builds the identifier set used to filter kvblock.Index
// lookups to candidate endpoints. Endpoints without metadata are skipped.
func (p *Producer) extractEndpointSet(endpoints []scheduling.Endpoint) sets.Set[string] {
	endpointSet := sets.New[string]()
	for _, ep := range endpoints {
		if m := ep.GetMetadata(); m != nil {
			endpointSet.Insert(p.indexPodIdentifier(m))
		}
	}
	return endpointSet
}
