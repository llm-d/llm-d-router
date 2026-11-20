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

package programaware

import (
	"fmt"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/sessionstate"
)

// Strategy is the fairness scheduling policy. All methods must be safe for
// concurrent use.
type Strategy interface {
	Name() string
	Pick(bandPriority int, queues map[string]QueueInfo) flowcontrol.FlowQueueAccessor
	OnPreRequest(metrics *ProgramMetrics, request *fwksched.InferenceRequest)
	OnCompleted(metrics *ProgramMetrics, request *fwksched.InferenceRequest, response *fwkrh.Response)
	EvictProgram(id string)
	Collectors() []prometheus.Collector
	// Consumes declares the request data the strategy reads, which the plugin
	// reports as its own data dependencies.
	Consumes() plugin.DataDependencies
}

type QueueInfo struct {
	Queue   flowcontrol.FlowQueueAccessor
	Metrics *ProgramMetrics
	Len     int
}

func newStrategy(cfg Config, name string) (Strategy, error) {
	switch cfg.Strategy {
	case "", "las":
		return &LASStrategy{
			weightService:         cfg.LASWeightService,
			weightHeadWait:        cfg.LASWeightHeadWait,
			halfLifeSeconds:       cfg.LASHalfLifeSeconds,
			attainedServiceTokens: newAttainedServiceCollector(name),
		}, nil
	case turnPriorityStrategyName:
		return &turnPriorityStrategy{
			timeWeight:        cfg.TurnPriorityTimeWeight,
			inactivitySeconds: cfg.TurnPriorityInactivitySeconds,
			sessionStateKey:   sessionstate.SessionStateDataKey.WithNonEmptyProducerName(cfg.SessionStateProducerName),
		}, nil
	default:
		return nil, fmt.Errorf("unknown scoring strategy %q: only \"las\" and \"turn-priority\" are supported", cfg.Strategy)
	}
}

func rangeNormalize(v, min, max float64) float64 {
	if max == min {
		return 0.5
	}
	return (v - min) / (max - min)
}
