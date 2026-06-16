package kv

import (
	"context"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/coordinator/pkg/pipeline"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
)

// sharedStorageKV uses a shared filesystem for KV transfer. No remote_* fields
// are needed because the consumer reads from the same storage the producer
// writes to.
type sharedStorageKV struct{}

func (sharedStorageKV) Name() string { return SharedStorage }

func (sharedStorageKV) PreparePrefillKVParams(ctx context.Context, _ *pipeline.RequestContext) map[string]any {
	params := map[string]any{"do_remote_decode": true, "do_remote_prefill": false}
	log.FromContext(ctx).WithName(loggerName).V(logutil.TRACE).Info("preparing prefill kv params", "params", params)
	return params
}

func (sharedStorageKV) PrepareDecodeKVParams(ctx context.Context, _ *pipeline.RequestContext) map[string]any {
	params := map[string]any{"do_remote_decode": false, "do_remote_prefill": true}
	log.FromContext(ctx).WithName(loggerName).V(logutil.TRACE).Info("preparing decode kv params", "params", params)
	return params
}
