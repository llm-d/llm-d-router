package programaware

import (
	"fmt"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// Strategy is the fairness scheduling policy. All methods must be safe for
// concurrent use.
type Strategy interface {
	Name() string
	Pick(bandPriority int, queues map[string]QueueInfo) flowcontrol.FlowQueueAccessor
	OnPreRequest(metrics *ProgramMetrics, request *fwksched.InferenceRequest)
	OnCompleted(metrics *ProgramMetrics, request *fwksched.InferenceRequest, response *fwkrc.Response)
	EvictProgram(id string)
	Collectors() []prometheus.Collector
}

type QueueInfo struct {
	Queue   flowcontrol.FlowQueueAccessor
	Metrics *ProgramMetrics
	Len     int
}

func newStrategy(cfg Config) (Strategy, error) {
	switch cfg.Strategy {
	case "", "las":
		return &LASStrategy{
			weightService:   cfg.LASWeightService,
			weightHeadWait:  cfg.LASWeightHeadWait,
			decayFactor:     cfg.LASDecayFactor,
			halfLifeSeconds: cfg.LASHalfLifeSeconds,
		}, nil
	default:
		return nil, fmt.Errorf("unknown scoring strategy %q: only \"las\" is supported", cfg.Strategy)
	}
}

func rangeNormalize(v, min, max float64) float64 {
	if max == min {
		return 0.5
	}
	return (v - min) / (max - min)
}
