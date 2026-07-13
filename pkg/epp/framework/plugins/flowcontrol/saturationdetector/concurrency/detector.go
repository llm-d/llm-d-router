/*
Copyright 2025 The Kubernetes Authors.

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
// Package concurrency implements a synchronous saturation detector and scheduling filter for LLM
// routing. It consumes in-flight requests and tokens data from the Endpoint's AttributeMap
// to provide instantaneous backpressure and protect endpoints from sudden traffic bursts.
//
// For detailed architectural trade-offs and configuration, see the package README.
package concurrency

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
)

const (
	ConcurrencyDetectorType = "concurrency-detector"
)

// ConcurrencyDetectorFactory instantiates the detector plugin using the provided JSON parameters.
func ConcurrencyDetectorFactory(
	name string,
	params *json.Decoder,
	handle fwkplugin.Handle,
) (fwkplugin.Plugin, error) {
	var apiCfg apiConfig
	if params != nil {
		if err := params.Decode(&apiCfg); err != nil {
			return nil, fmt.Errorf("failed to unmarshal concurrency detector config: %w", err)
		}
	}
	cfg, err := buildConfig(&apiCfg)
	if err != nil {
		return nil, err
	}
	return newDetector(name, *cfg, log.FromContext(handle.Context())), nil
}

var (
	_ fwksched.Filter                = &detector{}
	_ flowcontrol.SaturationDetector = &detector{}
)

// detector implements a saturation detector and scheduling filter based on active request concurrency.
type detector struct {
	config              config
	typedName           fwkplugin.TypedName
	inFlightLoadDataKey fwkplugin.DataKey
}

// newDetector creates a new instance of the Concurrency Detector.
func newDetector(name string, cfg config, logger logr.Logger) *detector {
	typedName := fwkplugin.TypedName{
		Type: ConcurrencyDetectorType,
		Name: name,
	}

	pluginLogger := logger.WithName(typedName.String())
	pluginLogger.V(logutil.DEFAULT).Info("Creating new ConcurrencyDetector",
		"mode", cfg.mode,
		"maxConcurrency", cfg.maxConcurrency,
		"maxTokenConcurrency", cfg.maxTokenConcurrency,
		"headroom", cfg.headroom)

	if cfg.headroom > 1.0 {
		pluginLogger.Info("Unusually high headroom configured; verify value is a fraction, not a percentage",
			"headroom", cfg.headroom,
			"effectiveBurst", fmt.Sprintf("%.0f%%", cfg.headroom*100))
	}

	return &detector{
		config:              cfg,
		typedName:           typedName,
		inFlightLoadDataKey: attrconcurrency.InFlightLoadDataKey.WithNonEmptyProducerName(cfg.inFlightLoadProducerName),
	}
}

// TypedName returns the type and name tuple of this plugin instance.
func (d *detector) TypedName() fwkplugin.TypedName {
	return d.typedName
}

func (d *detector) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{d.inFlightLoadDataKey: attrconcurrency.InFlightLoad{}},
	}
}

func (d *detector) getLoad(m datalayer.AttributeMap) *attrconcurrency.InFlightLoad {
	if val, ok := m.Get(d.inFlightLoadDataKey.String()); ok {
		if load, ok := val.(*attrconcurrency.InFlightLoad); ok {
			return load
		}
	}

	return &attrconcurrency.InFlightLoad{}
}

// Saturation calculates the saturation level of the pool.
//
// It returns an aggregate saturation signal where each dimension is evaluated as:
//
//	Saturation = Total Inflight / Total Capacity.
//
// In "hybrid" mode both the request and token dimensions are computed and the larger
// (more saturated) of the two is returned, so the pool reports saturated as soon as
// either dimension is exhausted.
func (d *detector) Saturation(_ context.Context, endpoints []datalayer.Endpoint) float64 {
	if len(endpoints) == 0 {
		return 1.0
	}

	var reqInflight, reqCapacity, tokInflight, tokCapacity int64
	for _, e := range endpoints {
		if e == nil {
			continue
		}

		reqCapacity += d.config.maxConcurrency
		tokCapacity += d.config.maxTokenConcurrency

		if e.GetMetadata() == nil {
			continue
		}

		load := d.getLoad(e.GetAttributes())
		reqInflight += load.Requests
		tokInflight += load.Tokens
	}

	switch d.config.mode {
	case modeTokens:
		return ratio(tokInflight, tokCapacity)
	case modeHybrid:
		return max(ratio(reqInflight, reqCapacity), ratio(tokInflight, tokCapacity))
	default:
		return ratio(reqInflight, reqCapacity)
	}
}

// ratio computes inflight/capacity, failing closed (1.0) when capacity is zero.
func ratio(inflight, capacity int64) float64 {
	if capacity == 0 {
		return 1.0
	}
	return float64(inflight) / float64(capacity)
}

// Filter blocks traffic to specific endpoints that are physically saturated or exceeding their safety limits.
//
// It applies a relaxed limit (Capacity * (1 + Headroom)) to allow for scheduling flexibility and burst tolerance.
// In "hybrid" mode an endpoint is dropped when either its request load or its token load reaches the limit.
func (d *detector) Filter(
	_ context.Context,
	_ *fwksched.InferenceRequest,
	endpoints []fwksched.Endpoint,
) []fwksched.Endpoint {
	// Pre-allocate assuming most endpoints will pass the filter to minimize allocations.
	filtered := make([]fwksched.Endpoint, 0, len(endpoints))

	reqLimit := int64(float64(d.config.maxConcurrency) * (1.0 + d.config.headroom))
	tokLimit := int64(float64(d.config.maxTokenConcurrency) * (1.0 + d.config.headroom))

	for _, e := range endpoints {
		if e == nil {
			continue
		}
		load := d.getLoad(e)

		if d.admits(load, reqLimit, tokLimit) {
			filtered = append(filtered, e)
		}
	}
	return filtered
}

// admits reports whether an endpoint is below its safety limit for the active mode.
func (d *detector) admits(load *attrconcurrency.InFlightLoad, reqLimit, tokLimit int64) bool {
	switch d.config.mode {
	case modeTokens:
		return load.Tokens < tokLimit
	case modeHybrid:
		return load.Requests < reqLimit && load.Tokens < tokLimit
	default:
		return load.Requests < reqLimit
	}
}
