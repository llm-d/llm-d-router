package loadaware

import (
	"context"
	"encoding/json"
	"fmt"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const (
	// LoadAwareType is the type of the LoadAware scorer
	LoadAwareType = "load-aware-scorer"

	// QueueThresholdDefault defines the default queue threshold value
	QueueThresholdDefault = 128
)

type loadAwareParameters struct {
	Threshold int `json:"threshold"`
}

// compile-time type assertion
var _ scheduling.Scorer = &LoadAware{}

// Factory defines the factory function for the LoadAware
func Factory(name string, rawParameters json.RawMessage, handle plugin.Handle) (plugin.Plugin, error) {
	parameters := loadAwareParameters{Threshold: QueueThresholdDefault}
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' scorer - %w", LoadAwareType, err)
		}
	}

	return NewLoadAware(handle.Context(), parameters.Threshold).WithName(name), nil
}

// NewLoadAware creates a new load based scorer
func NewLoadAware(ctx context.Context, queueThreshold int) *LoadAware {
	if queueThreshold <= 0 {
		queueThreshold = QueueThresholdDefault
		log.FromContext(ctx).V(logutil.DEFAULT).Info(fmt.Sprintf("queueThreshold %d should be positive, using default queue threshold %d", queueThreshold, QueueThresholdDefault))
	}

	return &LoadAware{
		typedName:      plugin.TypedName{Type: LoadAwareType},
		queueThreshold: float64(queueThreshold),
	}
}

// LoadAware scorer that is based on load
type LoadAware struct {
	typedName      plugin.TypedName
	queueThreshold float64
}

// TypedName returns the typed name of the plugin.
func (s *LoadAware) TypedName() plugin.TypedName {
	return s.typedName
}

// WithName sets the name of the plugin.
func (s *LoadAware) WithName(name string) *LoadAware {
	s.typedName.Name = name
	return s
}

// Category returns the preference the scorer applies when scoring candidate endpoints.
func (s *LoadAware) Category() scheduling.ScorerCategory {
	return scheduling.Distribution
}

// Score scores the given pod in range of 0-1
// Currently metrics contains number of requests waiting in the queue, there is no information about number of requests
// that can be processed in the given pod immediately.
// Pod with empty waiting requests queue is scored with 0.5
// Pod with requests in the queue will get score between 0.5 and 0.
// Score 0 will get pod with number of requests in the queue equal to the threshold used in load-based filter
// In the future, pods with additional capacity will get score higher than 0.5
func (s *LoadAware) Score(_ context.Context, _ *scheduling.CycleState, _ *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) map[scheduling.Endpoint]float64 {
	scoredEndpoints := make(map[scheduling.Endpoint]float64)

	for _, endpoint := range endpoints {
		waitingRequests := float64(endpoint.GetMetrics().WaitingQueueSize)

		if waitingRequests == 0 {
			scoredEndpoints[endpoint] = 0.5
		} else {
			if waitingRequests > s.queueThreshold {
				waitingRequests = s.queueThreshold
			}
			scoredEndpoints[endpoint] = 0.5 * (1.0 - (waitingRequests / s.queueThreshold))
		}
	}
	return scoredEndpoints
}
