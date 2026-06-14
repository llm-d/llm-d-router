package ec

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/llm-d/coordinator/pkg/pipeline"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
)

// nixlEC is the NIXL EC connector: each encoder response carries an
// ec_transfer_params object keyed by the encoded image's mm_hash. The
// coordinator merges them into a single flat map and forwards it on the
// prefill request: {"hash1": {...}, "hash2": {...}}.
type nixlEC struct{}

func (nixlEC) Name() string { return NIXL }

func (nixlEC) MergeEncodeResponse(reqCtx *pipeline.RequestContext, encResp map[string]any) {
	if len(encResp) == 0 {
		logger.Info("warning: encoder returned no ec_transfer_params; no nixl descriptor will be forwarded for this image",
			"requestID", reqCtx.RequestID)
		return
	}
	reqCtx.ECTransferParams = append(reqCtx.ECTransferParams, encResp)
	logger.V(logutil.TRACE).Info("merged encode response", "total", len(reqCtx.ECTransferParams))
}

// PreparePrefillECParams flattens the per-image encode responses into a single
// map keyed by mm_hash for the prefill request body. The returned map and its
// descriptors are independent copies of reqCtx.ECTransferParams, so callers may
// mutate the result freely.
func (nixlEC) PreparePrefillECParams(reqCtx *pipeline.RequestContext) (map[string]any, error) {
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
					"mmHash", k, "requestID", reqCtx.RequestID)
				continue
			}
			desc := copyDescriptor(v)
			if existing, exists := params[k]; exists {
				// Two encoder replicas answered for the same mm_hash. Identical
				// descriptors are harmless; conflicting ones are not. Picking
				// one (last-write-wins) would point the prefill pull at a peer
				// that may have rotated its buffers, so reject the request.
				equal, err := descriptorsEqual(existing, desc)
				if err != nil {
					return nil, fmt.Errorf("ec_transfer_params: comparing descriptors for mm_hash %q: %w", k, err)
				}
				if !equal {
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

// descriptorsEqual reports whether two ec_transfer_params descriptors are
// byte-equal under canonical JSON encoding. encoding/json sorts object keys,
// so the comparison is independent of map iteration order.
func descriptorsEqual(a, b any) (bool, error) {
	ab, err := json.Marshal(a)
	if err != nil {
		return false, err
	}
	bb, err := json.Marshal(b)
	if err != nil {
		return false, err
	}
	return bytes.Equal(ab, bb), nil
}
