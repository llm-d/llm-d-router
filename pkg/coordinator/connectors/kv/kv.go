// Package kv contains KV transfer connector implementations selected at
// config time. Each connector defines the kv_transfer_params shape sent to
// prefill and decode pods. Orchestration variants (shared-storage
// try-decode-first) are not implemented in this package, they require
// pipeline changes outside the per-step wire format.
package kv

import (
	"context"
	"fmt"

	"github.com/llm-d/coordinator/pkg/pipeline"
)

// DefaultKVConnectorName is the KV connector selected when an empty string is
// passed to Build. Defaults to kv-shared-storage (no-op on the wire).
const DefaultKVConnectorName = SharedStorage

// loggerName is the WithName scope applied to the context logger in connector
// log lines.
const loggerName = "kv"

// Connector controls the kv_transfer_params wire shape on the prefill and
// decode requests. Implementations are stateless and safe to share across
// requests.
type Connector interface {
	Name() string
	// PreparePrefillKVParams returns the kv_transfer_params map written into
	// the prefill request body.
	PreparePrefillKVParams(ctx context.Context, reqCtx *pipeline.RequestContext) map[string]any
	// PrepareDecodeKVParams returns the kv_transfer_params map written into
	// the decode request body. The prefill response's kv_transfer_params is
	// already populated into reqCtx.KVTransferParams by PrefillStep.
	PrepareDecodeKVParams(ctx context.Context, reqCtx *pipeline.RequestContext) map[string]any
}

// Build returns the KV connector for name. An empty name selects DefaultKVConnectorName.
func Build(name string) (Connector, error) {
	if name == "" {
		name = DefaultKVConnectorName
	}
	switch name {
	case NIXL:
		return nixlKV{}, nil
	case SharedStorage:
		return sharedStorageKV{}, nil
	case SGLang:
		return sglangKV{}, nil
	default:
		return nil, fmt.Errorf("unknown kv_connector: %q", name)
	}
}
