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

package gpuutilization

import (
	"context"
	"encoding/json"
	"fmt"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrgpu "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/gpu"
)

const (
	PluginType       = "gpu-utilization-filter"
	DefaultThreshold = 0.90
)

var _ fwksched.Filter = &Plugin{}

// Config holds configurable parameters for the GPU utilization filter.
type Config struct {
	// Threshold is the maximum GPU utilization (0.0-1.0) an endpoint may
	// have before it is filtered out. Default: 0.90 (90%).
	Threshold float64 `json:"threshold,omitempty"`
}

// Plugin filters endpoints whose GPU utilization exceeds a configurable
// threshold. If all endpoints exceed the threshold, all are kept (graceful
// fallback). Endpoints without GPU data pass through unconditionally.
type Plugin struct {
	typedName      fwkplugin.TypedName
	config         Config
	gpuUtilDataKey fwkplugin.DataKey
}

// Factory creates a gpu-utilization-filter plugin from configuration.
func Factory(name string, rawParameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	config := Config{Threshold: DefaultThreshold}
	if rawParameters != nil {
		if err := rawParameters.Decode(&config); err != nil {
			return nil, fmt.Errorf("failed to unmarshal config: %w", err)
		}
	}
	if config.Threshold < 0 || config.Threshold > 1.0 {
		return nil, fmt.Errorf("threshold must be in [0, 1], got %f", config.Threshold)
	}
	return &Plugin{
		typedName:      fwkplugin.TypedName{Type: PluginType, Name: name},
		config:         config,
		gpuUtilDataKey: attrgpu.GPUUtilizationDataKey,
	}, nil
}

func (p *Plugin) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// Filter keeps endpoints whose GPU utilization is at or below the configured
// threshold. Endpoints without GPU utilization data pass through (the DCGM
// poller may not have scraped yet). If every endpoint with data exceeds the
// threshold, all original endpoints are returned as a graceful fallback.
func (p *Plugin) Filter(ctx context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	logger := log.FromContext(ctx)

	if len(endpoints) <= 1 {
		return endpoints
	}

	var passed []fwksched.Endpoint
	for _, ep := range endpoints {
		raw, ok := ep.Get(p.gpuUtilDataKey.String())
		if !ok {
			passed = append(passed, ep)
			continue
		}
		util := raw.(attrgpu.GPUUtilization)
		if float64(util) <= p.config.Threshold {
			passed = append(passed, ep)
		}
	}

	if len(passed) == 0 {
		logger.V(logutil.DEBUG).Info("GPUUtilizationFilter: all endpoints above threshold, keeping all",
			"threshold", p.config.Threshold, "total", len(endpoints))
		return endpoints
	}

	logger.V(logutil.DEBUG).Info("GPUUtilizationFilter: filtered endpoints",
		"threshold", p.config.Threshold, "passed", len(passed), "total", len(endpoints))
	return passed
}

// Consumes declares a Required dependency on GPUUtilization so the framework
// validates at init that a producer (dcgm-extractor) is configured.
func (p *Plugin) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{p.gpuUtilDataKey: attrgpu.GPUUtilization(0)},
	}
}
