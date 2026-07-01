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

package ec

import (
	"context"
	"fmt"
	"reflect"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
)

// nixlEC is the NIXL EC connector: each encoder response carries an
// ec_transfer_params object keyed by the encoded image's mm_hash. The
// coordinator merges them into a single flat map and forwards it on the
// prefill request: {"hash1": {...}, "hash2": {...}}.
type nixlEC struct{}

func (nixlEC) Name() string { return NIXL }

func (nixlEC) MergeEncodeResponse(ctx context.Context, reqCtx *pipeline.RequestContext, encResp map[string]any) {
	logger := log.FromContext(ctx).WithName(loggerName)
	if len(encResp) == 0 {
		logger.Info("warning: encoder returned no ec_transfer_params; no nixl descriptor will be forwarded for this image")
		return
	}
	reqCtx.ECTransferParams = append(reqCtx.ECTransferParams, encResp)
	logger.V(logutil.TRACE).Info("merged encode response", "total", len(reqCtx.ECTransferParams))
}

// PreparePrefillECParams flattens the per-image encode responses into a single
// map keyed by mm_hash for the prefill request body. The returned map and its
// descriptors are independent copies of reqCtx.ECTransferParams, so callers may
// mutate the result freely.
func (nixlEC) PreparePrefillECParams(ctx context.Context, reqCtx *pipeline.RequestContext) (map[string]any, error) {
	logger := log.FromContext(ctx).WithName(loggerName)
	if len(reqCtx.ECTransferParams) == 0 {
		return make(map[string]any), nil
	}
	params := make(map[string]any, len(reqCtx.ECTransferParams))
	for _, entry := range reqCtx.ECTransferParams {
		for k, v := range entry {
			if v == nil {
				// A hash with no descriptor carries nothing to transfer; drop it
				// so the prefill body never sends "<mm_hash>": null.
				logger.V(logutil.DEBUG).Info("dropping ec_transfer_params entry with no descriptor",
					"mmHash", k)
				continue
			}
			desc := copyDescriptor(v)
			if existing, exists := params[k]; exists {
				// Two encoder replicas answered for the same mm_hash. Identical
				// descriptors are harmless; conflicting ones are not. Picking
				// one (last-write-wins) would point the prefill pull at a peer
				// that may have rotated its buffers, so reject the request.
				if !reflect.DeepEqual(existing, desc) {
					return nil, fmt.Errorf("ec_transfer_params: conflicting descriptors for mm_hash %q across encoder responses", k)
				}
				continue
			}
			params[k] = desc
		}
	}
	logger.V(logutil.TRACE).Info("preparing prefill ec params", "entries", len(params))
	return params, nil
}

// copyDescriptor returns a shallow copy of a descriptor map so the prepared
// prefill params do not alias reqCtx.ECTransferParams. Non-map values carry no
// mutable aliasing risk and are returned unchanged.
func copyDescriptor(v any) any {
	m, ok := v.(map[string]any)
	if !ok {
		return v
	}
	cp := make(map[string]any, len(m))
	for key, val := range m {
		cp[key] = val
	}
	return cp
}
