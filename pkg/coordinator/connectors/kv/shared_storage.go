package kv

import (
	"github.com/llm-d/coordinator/pkg/logging"
	"github.com/llm-d/coordinator/pkg/pipeline"
)

// sharedStorage uses a shared filesystem for KV transfer. No remote_* fields
// are needed because the consumer reads from the same storage the producer
// writes to.
type sharedStorage struct{}

func (sharedStorage) Name() string { return SharedStorage }

func (sharedStorage) PreparePrefillKVParams(_ *pipeline.RequestContext) map[string]any {
	params := map[string]any{"do_remote_decode": true}
	logger.V(logging.TRACE).Info("preparing prefill kv params", "params", params)
	return params
}

func (sharedStorage) PrepareDecodeKVParams(_ *pipeline.RequestContext) map[string]any {
	params := map[string]any{"do_remote_prefill": true}
	logger.V(logging.TRACE).Info("preparing decode kv params", "params", params)
	return params
}
