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

package kv

import (
	"context"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
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
