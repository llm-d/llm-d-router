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
	"fmt"
	"os"
	"strconv"

	"github.com/google/uuid"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// paramBootstrapPort is the kv_connector_params key that sets the bootstrap
// port advertised to prefill pods. When it is absent, envSGLangBootstrapPort is
// read instead; an invalid env value falls back to the default and is logged,
// so the fallback is observable.
const (
	paramBootstrapPort         = "bootstrap_port"
	envSGLangBootstrapPort     = "SGLANG_BOOTSTRAP_PORT"
	defaultSGLangBootstrapPort = 8998
)

// parseSGLangBootstrapPort resolves the bootstrap port from a raw value.
// An empty value selects the default. rejected is true when a non-empty value
// fails to parse or falls outside the valid TCP port range, in which case the
// default is returned.
func parseSGLangBootstrapPort(raw string) (port int, rejected bool) {
	if raw == "" {
		return defaultSGLangBootstrapPort, false
	}
	p, err := strconv.Atoi(raw)
	if err != nil || p < 1 || p > 65535 {
		return defaultSGLangBootstrapPort, true
	}
	return p, false
}

// newSGLangKV builds the SGLang connector from its kv_connector_params. An
// invalid parameter is a configuration error; only the env fallback degrades
// to the default.
func newSGLangKV(params map[string]any) (Connector, error) {
	for k := range params {
		if k != paramBootstrapPort {
			return nil, fmt.Errorf("kv_connector_params: unknown key %q for %s", k, SGLang)
		}
	}
	if v, ok := params[paramBootstrapPort]; ok {
		// fmt.Sprint renders every integral numeric type the config decoder
		// produces as plain digits, so one parser validates all of them.
		raw := fmt.Sprint(v)
		port, rejected := parseSGLangBootstrapPort(raw)
		if raw == "" || rejected {
			return nil, fmt.Errorf("kv_connector_params.%s: invalid port %v (want 1-65535)", paramBootstrapPort, v)
		}
		return sglangKV{bootstrapPort: port}, nil
	}
	raw := os.Getenv(envSGLangBootstrapPort)
	port, rejected := parseSGLangBootstrapPort(raw)
	if rejected {
		log.Log.WithName(loggerName).Error(
			fmt.Errorf("invalid %s %q", envSGLangBootstrapPort, raw),
			"using default SGLang bootstrap port", "default", defaultSGLangBootstrapPort)
	}
	return sglangKV{bootstrapPort: port}, nil
}

// sglangKV implements the SGLang KV transfer protocol. Both prefill and decode
// receive bootstrap coordination fields (port and room ID). The prefill pod is
// expected to echo bootstrap fields back in its kv_transfer_params response;
// PrepareDecodeKVParams forwards those verbatim so the decode pod can open the
// bootstrap channel to the prefill pod.
type sglangKV struct {
	bootstrapPort int
}

func (sglangKV) Name() string { return SGLang }

func (c sglangKV) PreparePrefillKVParams(ctx context.Context, _ *pipeline.RequestContext) map[string]any {
	params := map[string]any{
		reqcommon.FieldDoRemoteDecode:  true,
		reqcommon.FieldDoRemotePrefill: false,
		reqcommon.FieldBootstrapPort:   c.bootstrapPort,
		reqcommon.FieldBootstrapRoom:   uuid.NewString(),
	}
	log.FromContext(ctx).WithName(loggerName).V(logutil.TRACE).Info("preparing prefill kv params", "params", params)
	return params
}

func (sglangKV) PrepareDecodeKVParams(ctx context.Context, reqCtx *pipeline.RequestContext) map[string]any {
	out := make(map[string]any, len(reqCtx.KVTransferParams))
	for k, v := range reqCtx.KVTransferParams {
		out[k] = v
	}
	out[reqcommon.FieldDoRemoteDecode] = false
	out[reqcommon.FieldDoRemotePrefill] = true
	log.FromContext(ctx).WithName(loggerName).V(logutil.TRACE).Info("preparing decode kv params", "params", out)
	return out
}
