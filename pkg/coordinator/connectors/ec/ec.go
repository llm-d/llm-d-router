// Package ec contains EC (encoder cache) transfer connector implementations
// selected at config time. Each connector controls how encoder pods hand off
// embeddings to the prefill consumer pod.
package ec

import (
	"fmt"

	ctrl "sigs.k8s.io/controller-runtime"

	"github.com/llm-d/coordinator/pkg/pipeline"
)

// DefaultECConnectorName is the EC connector selected when an empty string is
// passed to Build. Defaults to ec-shared-storage (no-op on the wire).
const DefaultECConnectorName = SharedStorage

var logger = ctrl.Log.WithName("ec")

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
	MergeEncodeResponse(reqCtx *pipeline.RequestContext, encResp map[string]any)
	// PreparePrefillECParams returns the ec_transfer_params map for the
	// prefill request body. A nil/empty return means no ec_transfer_params
	// field should be emitted. It errors when encoder responses carry
	// conflicting descriptors for the same mm_hash.
	PreparePrefillECParams(reqCtx *pipeline.RequestContext) (map[string]any, error)
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
