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

package approximateprefix

import (
	"strings"

	"k8s.io/apimachinery/pkg/types"
)

// GetPrefixMatch returns the endpoint IDs (NamespacedName strings) that have
// the given prefix hash cached. This exposes the internal indexer to the
// statestore Local prefix provider without exporting the unexported
// podSet/ServerID types.
func (p *dataProducer) GetPrefixMatch(hash uint64) []string {
	pods := p.indexerInst.Get(blockHash(hash))
	if pods == nil {
		return nil
	}
	result := make([]string, 0, len(pods))
	for pod := range pods {
		result = append(result, pod.String())
	}
	return result
}

// CommitPrefix records that the given prefix hashes are now cached on the
// endpoint identified by endpointID, as a byproduct of a routing decision. The
// endpoint's GPU block count is not known here, so the indexer falls back to
// its default LRU size (NumOfGPUBlocks <= 0 path in indexer.Add).
func (p *dataProducer) CommitPrefix(endpointID string, hashes []uint64) {
	blockHashes := make([]blockHash, 0, len(hashes))
	for _, h := range hashes {
		blockHashes = append(blockHashes, blockHash(h))
	}
	p.indexerInst.Add(blockHashes, server{ServerID: ServerID(parseNamespacedName(endpointID))})
}

// RemovePrefixEndpoint removes all prefix state for the given endpoint. This
// delegates to the indexer's RemovePod.
func (p *dataProducer) RemovePrefixEndpoint(endpointID string) {
	p.indexerInst.RemovePod(ServerID(parseNamespacedName(endpointID)))
}

// parseNamespacedName parses a NamespacedName string of the form "namespace/name"
// (as produced by NamespacedName.String()) into a types.NamespacedName. When
// there is no slash, the whole string is treated as the name with an empty
// namespace, matching the indexer's tolerance for unknown pods.
func parseNamespacedName(s string) types.NamespacedName {
	if idx := strings.Index(s, "/"); idx >= 0 {
		return types.NamespacedName{Namespace: s[:idx], Name: s[idx+1:]}
	}
	return types.NamespacedName{Name: s}
}
