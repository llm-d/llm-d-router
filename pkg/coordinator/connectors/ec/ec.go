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

// Package ec contains EC (encoder cache) transfer connector implementations
// selected at config time. Each connector controls how encoder pods hand off
// embeddings to the prefill consumer pod.
package ec

import (
	"context"
	"fmt"

	"github.com/llm-d/coordinator/pkg/pipeline"
)

// DefaultECConnectorName is the EC connector selected when an empty string is
// passed to Build. Defaults to ec-shared-storage (no-op on the wire).
const DefaultECConnectorName = SharedStorage

// loggerName is the WithName scope applied to the context logger in connector
// log lines.
const loggerName = "ec"

// Connector controls how encoder cache (vision encoder embeddings) is
// transferred from encoder pods to the prefill consumer pod. Two flavors:
//
//   - ec-nixl: encoder pods register embeddings in NIXL-mapped memory and
//     return {mm_hash: descriptor} per encoded image, where descriptor is an
//     opaque per-encoding map (fields such as peer_host, peer_port, size_bytes,
//     and nixl_agent_metadata_b64; the set varies by encoder). The coordinator
//     merges these by mm_hash and forwards them to the prefill request as
//     ec_transfer_params.
//   - ec-shared-storage: encoder pods write embeddings to shared storage keyed
//     by mm_hash. The consumer reads them back; no ec_transfer_params needed
//     on the wire.
//
// EC connector selection is independent of the KV connector — a deployment
// can pair ec-nixl with kv-shared-storage, etc.
type Connector interface {
	Name() string
	// MergeEncodeResponse incorporates one encoder response into
	// reqCtx.ECTransferParams. Callers must not call MergeEncodeResponse
	// concurrently; the encode step serializes calls after gathering parallel
	// responses.
	MergeEncodeResponse(ctx context.Context, reqCtx *pipeline.RequestContext, encResp map[string]any)
	// PreparePrefillECParams returns the ec_transfer_params map for the
	// prefill request body. A nil/empty return means no ec_transfer_params
	// field should be emitted. It errors when encoder responses carry
	// conflicting descriptors for the same mm_hash.
	PreparePrefillECParams(ctx context.Context, reqCtx *pipeline.RequestContext) (map[string]any, error)
}

// Build returns the named EC connector. An empty name selects DefaultECConnectorName.
func Build(name string) (Connector, error) {
	if name == "" {
		name = DefaultECConnectorName
	}
	switch name {
	case NIXL:
		return nixlEC{}, nil
	case SharedStorage:
		return sharedStorageEC{}, nil
	default:
		return nil, fmt.Errorf("unknown ec_connector: %q", name)
	}
}
