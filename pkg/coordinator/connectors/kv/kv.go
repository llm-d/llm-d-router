// Package kv contains KV transfer connector implementations selected at
// config time. Each connector defines the kv_transfer_params shape sent to
// prefill and decode pods. Orchestration variants (shared_storage
// try-decode-first) are not implemented in this package — they require
// pipeline changes outside the per-step wire format.
package kv

import (
	"fmt"

	ctrl "sigs.k8s.io/controller-runtime"

	"github.com/llm-d/coordinator/pkg/pipeline"
	logutil "github.com/llm-d/llm-d-inference-scheduler/pkg/common/observability/logging"
)

// DefaultKVConnectorName is the KV connector selected when an empty string is
// passed to Build.
const DefaultKVConnectorName = NIXLv2

var logger = ctrl.Log.WithName("kv")

// Connector controls the kv_transfer_params wire shape on the prefill and
// decode requests. Implementations are stateless and safe to share across
// requests.
type Connector interface {
	Name() string
	// PreparePrefillKVParams returns the kv_transfer_params map written into
	// the prefill request body.
	PreparePrefillKVParams(reqCtx *pipeline.RequestContext) map[string]any
	// PrepareDecodeKVParams returns the kv_transfer_params map written into
	// the decode request body. The prefill response's kv_transfer_params is
	// already populated into reqCtx.KVTransferParams by PrefillStep.
	PrepareDecodeKVParams(reqCtx *pipeline.RequestContext) map[string]any
}

// Build returns the KV connector for name. An empty name selects DefaultKVConnectorName.
func Build(name string) (Connector, error) {
	if name == "" {
		name = DefaultKVConnectorName
	}
	switch name {
	case NIXLv2:
		logger.V(logutil.DEFAULT).Info("using connector", "name", name)
		return nixlV2{}, nil
	case SharedStorage:
		logger.V(logutil.DEFAULT).Info("using connector", "name", name)
		return sharedStorage{}, nil
	case SGLang:
		logger.V(logutil.DEFAULT).Info("using connector", "name", name)
		return sglangKV{}, nil
	default:
		return nil, fmt.Errorf("unknown connector: %q", name)
	}
}
