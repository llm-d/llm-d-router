package kv

import (
	"math/rand/v2"
	"os"
	"strconv"
	"time"

	"github.com/llm-d/coordinator/pkg/connector"
	"github.com/llm-d/coordinator/pkg/logging"
	"github.com/llm-d/coordinator/pkg/pipeline"
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

func (sglangKV) Name() string { return connector.NameSGLang }

func (sglangKV) PreparePrefillKVParams(_ *pipeline.RequestContext) map[string]any {
	roomID := time.Now().UnixNano() + int64(rand.IntN(1000))
	params := map[string]any{
		"do_remote_decode": true,
		"bootstrap_port":   sglangBootstrapPort,
		"bootstrap_room":   roomID,
	}
	logger.V(logging.TRACE).Info("preparing prefill kv params", "params", params)
	return params
}

func (sglangKV) PrepareDecodeKVParams(reqCtx *pipeline.RequestContext) map[string]any {
	out := make(map[string]any, len(reqCtx.KVTransferParams))
	for k, v := range reqCtx.KVTransferParams {
		out[k] = v
	}
	logger.V(logging.TRACE).Info("preparing decode kv params", "params", out)
	return out
}
