/*
Copyright 2026 The Kubernetes Authors.

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

// Package probabilisticadmitter implements a binary-tier probabilistic admission control plugin.
// Protected requests (priority >= 0) are always admitted; sheddable requests (priority < 0)
// are rejected with probability min(sat^power * k, 1.0) where sat is the cluster saturation.
package probabilisticadmitter

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const (
	// Type is the registered plugin type for the probabilistic admitter.
	Type = "probabilistic-admitter"

	defaultQueueDepthThreshold  = 5
	defaultKVCacheUtilThreshold = 0.8
	defaultPower                = 5.0
	defaultK                    = 300.0
)

// Parameters defines the JSON-configurable fields for the plugin.
type Parameters struct {
	QueueDepthThreshold  int     `json:"queueDepthThreshold"`
	KVCacheUtilThreshold float64 `json:"kvCacheUtilThreshold"`
	Power                float64 `json:"power"`
	K                    float64 `json:"k"`
}

// compile-time interface assertion
var _ requestcontrol.Admitter = &ProbabilisticAdmitter{}

// ProbabilisticAdmitter implements binary-tier probabilistic shedding.
type ProbabilisticAdmitter struct {
	typedName            fwkplugin.TypedName
	queueDepthThreshold  float64
	kvCacheUtilThreshold float64
	power                float64
	k                    float64
	randFn               func() float64
}

// Factory creates a ProbabilisticAdmitter from plugin configuration.
func Factory(name string, rawParameters json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params := Parameters{
		QueueDepthThreshold:  defaultQueueDepthThreshold,
		KVCacheUtilThreshold: defaultKVCacheUtilThreshold,
		Power:                defaultPower,
		K:                    defaultK,
	}
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &params); err != nil {
			return nil, fmt.Errorf("failed to parse parameters for '%s' plugin: %w", Type, err)
		}
	}
	if params.QueueDepthThreshold <= 0 {
		return nil, fmt.Errorf("plugin '%s': queueDepthThreshold must be > 0, got %d", Type, params.QueueDepthThreshold)
	}
	if params.KVCacheUtilThreshold <= 0.0 {
		return nil, fmt.Errorf("plugin '%s': kvCacheUtilThreshold must be > 0, got %g", Type, params.KVCacheUtilThreshold)
	}
	if params.Power <= 0 {
		return nil, fmt.Errorf("plugin '%s': power must be > 0, got %g", Type, params.Power)
	}
	if params.K <= 0 {
		return nil, fmt.Errorf("plugin '%s': k must be > 0, got %g", Type, params.K)
	}
	return newAdmitter(params).WithName(name), nil
}

// newAdmitter creates a ProbabilisticAdmitter with the given parameters.
func newAdmitter(params Parameters) *ProbabilisticAdmitter {
	return &ProbabilisticAdmitter{
		typedName:            fwkplugin.TypedName{Type: Type},
		queueDepthThreshold:  float64(params.QueueDepthThreshold),
		kvCacheUtilThreshold: params.KVCacheUtilThreshold,
		power:                params.Power,
		k:                    params.K,
		randFn:               rand.Float64,
	}
}

// WithName sets the plugin instance name.
func (p *ProbabilisticAdmitter) WithName(name string) *ProbabilisticAdmitter {
	p.typedName.Name = name
	return p
}

// WithRandFn replaces the random number generator (for deterministic testing).
func (p *ProbabilisticAdmitter) WithRandFn(fn func() float64) *ProbabilisticAdmitter {
	p.randFn = fn
	return p
}

// TypedName returns the plugin type and instance name.
func (p *ProbabilisticAdmitter) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// AdmitRequest implements requestcontrol.Admitter.
// Returns nil to admit, or a ResourceExhausted error to reject.
func (p *ProbabilisticAdmitter) AdmitRequest(_ context.Context, request *fwksched.InferenceRequest, pods []fwksched.Endpoint) error {
	if request == nil {
		return nil
	}
	if request.Objectives.Priority >= 0 {
		return nil
	}
	if len(pods) == 0 {
		return nil
	}

	saturation := p.computeSaturation(pods)
	prob := math.Min(math.Pow(saturation, p.power)*p.k, 1.0)

	if p.randFn() < prob {
		return errcommon.Error{
			Code: errcommon.ResourceExhausted,
			Msg:  fmt.Sprintf("probabilistic-admitter: rejected, saturation=%.3f prob=%.2f", saturation, prob),
		}
	}
	return nil
}

// computeSaturation calculates avg(max(QD/qdThreshold, KV/kvThreshold)) across all pods.
func (p *ProbabilisticAdmitter) computeSaturation(pods []fwksched.Endpoint) float64 {
	var total float64
	for _, pod := range pods {
		m := pod.GetMetrics()
		if m == nil {
			total += 1.0
			continue
		}
		qRatio := float64(m.WaitingQueueSize) / p.queueDepthThreshold
		kvRatio := m.KVCacheUsagePercent / p.kvCacheUtilThreshold
		total += math.Max(qRatio, kvRatio)
	}
	return total / float64(len(pods))
}
