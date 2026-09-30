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
	"math/rand/v2"
	"net"
	"os"
	"strconv"
	"sync"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

const (
	fieldBootstrapHost = "bootstrap_host"
	fieldBootstrapPort = "bootstrap_port"
	fieldBootstrapRoom = "bootstrap_room"
)

// envSGLangBootstrapPort optionally overrides the bootstrap port written into
// the prefill and decode bodies. A value that is not a valid integer is rejected in favor of the
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

// sglangRankFields are the SGLang data-parallel rank fields. A rank a client
// names can differ from the rank SGLang derives from bootstrap_room, and SGLang
// then fails the request, so the fields are removed from client bodies.
var sglangRankFields = []string{"routed_dp_rank", "data_parallel_rank", "disagg_prefill_dp_rank"}

// sglangKV implements the SGLang KV transfer protocol. The decode pod joins a
// room on the prefill pod's bootstrap server, named by bootstrap_host,
// bootstrap_port and bootstrap_room at the top level of the body, and the
// prefill request completes only after that. SGLang reads no
// kv_transfer_params, so the Prepare methods return nil and the fields are set
// by ApplyBootstrapFields.
type sglangKV struct{}

var _ ConcurrentConnector = sglangKV{}

func (sglangKV) Name() string { return SGLang }

func (sglangKV) PreparePrefillKVParams(context.Context, *pipeline.RequestContext) map[string]any {
	return nil
}

func (sglangKV) PrepareDecodeKVParams(context.Context, *pipeline.RequestContext) map[string]any {
	return nil
}

// ApplyBootstrapFields writes bootstrap_host, bootstrap_port and one integer
// bootstrap_room into every body. The host goes on the prefill body too: with
// data-parallel size above 1 the prefill pod uses it to register its rank.
func (sglangKV) ApplyBootstrapFields(ctx context.Context, prefillHostPort string, bodies ...map[string]any) error {
	host, _, err := net.SplitHostPort(prefillHostPort)
	if err != nil {
		return fmt.Errorf("invalid prefill endpoint %q: %w", prefillHostPort, err)
	}
	port := resolveSGLangBootstrapPort(ctx)
	// SGLang's own balancer draws the room from [0, 2^63).
	room := rand.Int64()
	for _, body := range bodies {
		for _, field := range sglangRankFields {
			delete(body, field)
		}
		body[fieldBootstrapHost] = host
		body[fieldBootstrapPort] = port
		body[fieldBootstrapRoom] = room
	}
	log.FromContext(ctx).WithName(loggerName).V(logutil.TRACE).Info("applied bootstrap fields",
		fieldBootstrapHost, host, fieldBootstrapPort, port, fieldBootstrapRoom, room)
	return nil
}
