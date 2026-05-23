// Package ec contains EC (encoder cache) transfer connector implementations
// selected at config time. Each connector controls how encoder pods hand off
// embeddings to the prefill consumer pod.
package ec

import (
	"fmt"

	ctrl "sigs.k8s.io/controller-runtime"

	"github.com/llm-d/coordinator/pkg/pipeline"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
)

// DefaultECConnectorName is the EC connector selected when an empty string is
// passed to Build. Defaults to shared_storage (no-op on the wire).
const DefaultECConnectorName = SharedStorage

var logger = ctrl.Log.WithName("ec")

// Connector controls how encoder cache (vision encoder embeddings) is
// transferred from encoder pods to the prefill consumer pod. Two flavors:
//
//   - nixl: encoder pods register embeddings in NIXL-mapped memory and return
//     {mm_hash: {peer_host, peer_port, size_bytes, nixl_agent_metadata_b64}}
//     per encoded image. The coordinator merges these by mm_hash and forwards
//     them to the prefill request as ec_transfer_params.
//   - shared_storage: encoder pods write embeddings to shared storage keyed
//     by mm_hash. The consumer reads them back; no ec_transfer_params needed
//     on the wire.
//
// EC connector selection is independent of the KV connector — a deployment
// can pair nixl-EC with shared_storage-KV, etc.
type Connector interface {
	Name() string
	// MergeEncodeResponse incorporates one encoder response into
	// reqCtx.ECTransferParams. Callers must not call MergeEncodeResponse
	// concurrently; the encode step serializes calls after gathering parallel
	// responses.
	MergeEncodeResponse(reqCtx *pipeline.RequestContext, encResp map[string]any)
	// PreparePrefillECParams returns the ec_transfer_params map for the
	// prefill request body. A nil/empty return means no ec_transfer_params
	// field should be emitted.
	PreparePrefillECParams(reqCtx *pipeline.RequestContext) map[string]any
}

// Build returns the named EC connector. An empty name selects DefaultECConnectorName.
func Build(name string) (Connector, error) {
	if name == "" {
		name = DefaultECConnectorName
	}
	switch name {
	case NIXLv2:
		logger.V(logutil.DEFAULT).Info("using connector", "name", name)
		return nixlV2{}, nil
	case SharedStorage:
		logger.V(logutil.DEFAULT).Info("using connector", "name", name)
		return sharedStorage{}, nil
	default:
		return nil, fmt.Errorf("unknown ec_connector: %q", name)
	}
}
