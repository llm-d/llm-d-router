package kv

import (
	"context"
	"os"
	"strconv"

	"github.com/google/uuid"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/coordinator/pkg/pipeline"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
)

const (
	fieldBootstrapHost = "bootstrap_host"
	fieldBootstrapPort = "bootstrap_port"
	fieldBootstrapRoom = "bootstrap_room"
)

var sglangBootstrapPort = func() int {
	port := 8998
	if s := os.Getenv("SGLANG_BOOTSTRAP_PORT"); s != "" {
		if p, err := strconv.Atoi(s); err == nil {
			port = p
		}
	}
	return port
}()

// sglangKV implements the SGLang KV transfer protocol. Both prefill and decode
// receive bootstrap coordination fields (port and room ID). The prefill pod is
// expected to echo bootstrap fields back in its kv_transfer_params response;
// PrepareDecodeKVParams forwards those verbatim so the decode pod can open the
// bootstrap channel to the prefill pod.
type sglangKV struct{}

func (sglangKV) Name() string { return SGLang }

func (sglangKV) PreparePrefillKVParams(ctx context.Context, _ *pipeline.RequestContext) map[string]any {
	params := map[string]any{
		"do_remote_decode":  true,
		"do_remote_prefill": false,
		fieldBootstrapPort:  sglangBootstrapPort,
		fieldBootstrapRoom:  uuid.NewString(),
	}
	log.FromContext(ctx).WithName(loggerName).V(logutil.TRACE).Info("preparing prefill kv params", "params", params)
	return params
}

func (sglangKV) PrepareDecodeKVParams(ctx context.Context, reqCtx *pipeline.RequestContext) map[string]any {
	out := make(map[string]any, len(reqCtx.KVTransferParams))
	for k, v := range reqCtx.KVTransferParams {
		out[k] = v
	}
	out["do_remote_decode"] = false
	out["do_remote_prefill"] = true
	log.FromContext(ctx).WithName(loggerName).V(logutil.TRACE).Info("preparing decode kv params", "params", out)
	return out
}
