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

package prefixhash

import (
	"strconv"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const (
	// RoutingMetadataNamespace is the Envoy dynamic-metadata namespace an upstream ext_proc (for example the
	// IPP) writes routing-relevant data to.
	RoutingMetadataNamespace = "llm-d.routing"
	// prefixHashesMetadataField is the field within that namespace holding the ordered block-hash chain, as
	// decimal-string uint64s (JSON cannot carry uint64 precisely as a number).
	prefixHashesMetadataField = "prefix_hashes"
)

// BlockHashesFromMetadata reads a precomputed prefix block-hash chain an upstream ext_proc placed in the
// request's Envoy dynamic metadata (namespace RoutingMetadataNamespace, field "prefix_hashes"), instead of
// hashing the tokenized body. The upstream stage MUST use the same block-hash scheme as GetBlockHashes for
// the hashes to match the per-endpoint index. Returns nil when the namespace or field is absent, so the
// caller treats a missing chain as "no prefix". maxPrefixBlocks (> 0) caps the chain length.
func BlockHashesFromMetadata(request *scheduling.InferenceRequest, maxPrefixBlocks int) [][]BlockHash {
	if request == nil || request.Metadata == nil {
		return nil
	}
	ns, ok := request.Metadata[RoutingMetadataNamespace].(map[string]any)
	if !ok {
		return nil
	}
	raw, ok := ns[prefixHashesMetadataField].([]any)
	if !ok {
		return nil
	}
	hashes := make([]BlockHash, 0, len(raw))
	for _, v := range raw {
		s, ok := v.(string)
		if !ok {
			continue
		}
		h, err := strconv.ParseUint(s, 10, 64)
		if err != nil {
			continue
		}
		hashes = append(hashes, BlockHash(h))
		if maxPrefixBlocks > 0 && len(hashes) >= maxPrefixBlocks {
			break
		}
	}
	if len(hashes) == 0 {
		return nil
	}
	return [][]BlockHash{hashes}
}
