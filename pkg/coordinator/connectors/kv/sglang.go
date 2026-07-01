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
	"sync"

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

// envSGLangBootstrapPort optionally overrides the bootstrap port advertised to
// prefill pods. A value that is not a valid integer is rejected in favor of the
// default and logged, so the fallback is observable.
const (
	envSGLangBootstrapPort     = "SGLANG_BOOTSTRAP_PORT"
	defaultSGLangBootstrapPort = 8998
)

var (
	sglangBootstrapPortOnce sync.Once
	sglangBootstrapPort     int
)

// parseSGLangBootstrapPort resolves the bootstrap port from the raw env value.
// An empty value selects the default. rejected is true only when a non-empty
// value fails to parse, in which case the default is returned.
func parseSGLangBootstrapPort(raw string) (port int, rejected bool) {
	if raw == "" {
		return defaultSGLangBootstrapPort, false
	}
	p, err := strconv.Atoi(raw)
	if err != nil {
		return defaultSGLangBootstrapPort, true
	}
	return p, false
}

// resolveSGLangBootstrapPort reads SGLANG_BOOTSTRAP_PORT once on first use,
// where a configured context logger is available to report a rejected value.
func resolveSGLangBootstrapPort(ctx context.Context) int {
	sglangBootstrapPortOnce.Do(func() {
		raw := os.Getenv(envSGLangBootstrapPort)
		port, rejected := parseSGLangBootstrapPort(raw)
		if rejected {
			log.FromContext(ctx).WithName(loggerName).Error(
				fmt.Errorf("invalid %s %q", envSGLangBootstrapPort, raw),
				"using default SGLang bootstrap port", "default", defaultSGLangBootstrapPort)
		}
		sglangBootstrapPort = port
	})
	return sglangBootstrapPort
}

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
		fieldBootstrapPort:  resolveSGLangBootstrapPort(ctx),
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
