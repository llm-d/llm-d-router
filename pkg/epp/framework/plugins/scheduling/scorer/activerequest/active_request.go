package activerequest

import (
	"context"
	"encoding/json"
	"fmt"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
)

const (
	// ActiveRequestType is the type of the ActiveRequest scorer.
	ActiveRequestType = "active-request-scorer"
)

// Parameters defines the parameters for ActiveRequest.
type Parameters struct {
	// Deprecated: RequestTimeout is ignored. In-flight request lifecycle is
	// tracked by the inflight-load-producer data producer.
	RequestTimeout string `json:"requestTimeout"`

	// IdleThreshold defines the maximum number of active requests for a pod
	// to be considered "idle". Pods with request count <= idleThreshold
	// will receive a score of 1.0.
	// Default: 0 (only pods with zero requests are considered idle)
	IdleThreshold int `json:"idleThreshold"`

	// MaxBusyScore defines the maximum score that can be assigned to busy pods
	// (pods with request count > idleThreshold). This creates a scoring gap
	// between idle and busy pods.
	// Range: 0.0 to 1.0
	// Default: 1.0 (no gap, current behavior)
	// Example: 0.5 means idle pods get 1.0, busiest pod gets 0.0, least busy gets 0.5
	MaxBusyScore float64 `json:"maxBusyScore"`
}

// endpointScores implements logr.Marshaler to lazily convert endpoint keys
// to strings only when the log line is actually written.
type endpointScores map[scheduling.Endpoint]float64

func (s endpointScores) MarshalLog() interface{} {
	result := make(map[string]float64, len(s))
	for ep, score := range s {
		result[ep.GetMetadata().NamespacedName.String()] = score
	}
	return result
}

// compile-time type assertion
var _ scheduling.Scorer = &ActiveRequest{}
var _ plugin.ConsumerPlugin = &ActiveRequest{}

// Factory defines the factory function for the ActiveRequest scorer.
func Factory(name string, rawParameters json.RawMessage, handle plugin.Handle) (plugin.Plugin, error) {
	parameters := Parameters{}
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' scorer - %w", ActiveRequestType, err)
		}
	}

	return NewActiveRequest(handle.Context(), &parameters).WithName(name), nil
}

// NewActiveRequest creates a new ActiveRequest scorer.
func NewActiveRequest(ctx context.Context, params *Parameters) *ActiveRequest {
	logger := log.FromContext(ctx)

	if params != nil && params.RequestTimeout != "" {
		logger.Info("DEPRECATED: requestTimeout is deprecated and ignored; inflight tracking is handled by inflight-load-producer",
			"requestTimeout", params.RequestTimeout)
	}

	// Set idle threshold (default: 0)
	idleThreshold := 0
	if params != nil && params.IdleThreshold >= 0 {
		idleThreshold = params.IdleThreshold
	}

	// Set max busy score (default: 1.0)
	maxBusyScore := 1.0
	if params != nil && params.MaxBusyScore >= 0 && params.MaxBusyScore <= 1.0 {
		maxBusyScore = params.MaxBusyScore
	}

	if idleThreshold != 0 || maxBusyScore != 1.0 {
		logger.Info("Active request scorer configured with idle preference",
			"idleThreshold", idleThreshold,
			"maxBusyScore", maxBusyScore)
	}

	return &ActiveRequest{
		typedName:     plugin.TypedName{Type: ActiveRequestType},
		idleThreshold: idleThreshold,
		maxBusyScore:  maxBusyScore,
	}
}

// ActiveRequest scores endpoints based on in-flight request counts produced by
// the inflight-load-producer data producer.
type ActiveRequest struct {
	typedName plugin.TypedName

	// idleThreshold defines the max request count to be considered idle
	idleThreshold int
	// maxBusyScore defines the maximum score for busy (non-idle) pods
	maxBusyScore float64
}

// TypedName returns the typed name of the plugin.
func (s *ActiveRequest) TypedName() plugin.TypedName {
	return s.typedName
}

// WithName sets the name of the plugin.
func (s *ActiveRequest) WithName(name string) *ActiveRequest {
	s.typedName.Name = name
	return s
}

// Category returns the preference the scorer applies when scoring candidate endpoints.
func (s *ActiveRequest) Category() scheduling.ScorerCategory {
	return scheduling.Distribution
}

// Consumes returns the in-flight load attribute required for scoring.
func (s *ActiveRequest) Consumes() map[string]any {
	return map[string]any{
		attrconcurrency.InFlightLoadKey: attrconcurrency.InFlightLoad{},
	}
}

// Score scores the given endpoints based on the number of active requests
// being served by each endpoint. The score is normalized to a range of 0-1.
func (s *ActiveRequest) Score(ctx context.Context, _ *scheduling.CycleState, _ *scheduling.InferenceRequest,
	endpoints []scheduling.Endpoint) map[scheduling.Endpoint]float64 {
	requestCounts := make(map[scheduling.Endpoint]int64, len(endpoints))
	logCounts := make(map[string]int64, len(endpoints))
	maxCount := int64(0)

	for _, endpoint := range endpoints {
		endpointName := endpoint.GetMetadata().NamespacedName.String()
		count := requestCount(ctx, endpoint)
		requestCounts[endpoint] = count
		logCounts[endpointName] = count
		if count > maxCount {
			maxCount = count
		}
	}

	log.FromContext(ctx).V(logutil.TRACE).Info("Active request counts", "endpointCounts", logCounts, "maxCount", maxCount)

	scoredEndpointsMap := make(map[scheduling.Endpoint]float64, len(endpoints))
	for _, endpoint := range endpoints {
		count := requestCounts[endpoint]
		// Check if pod is idle (count <= idleThreshold)
		if count <= int64(s.idleThreshold) {
			scoredEndpointsMap[endpoint] = 1.0 // Idle pods always get max score
			continue
		}

		// Busy pod: scale from 0 to maxBusyScore
		scoredEndpointsMap[endpoint] = float64(maxCount-count) / float64(maxCount) * s.maxBusyScore
	}

	log.FromContext(ctx).V(logutil.TRACE).Info("Scored endpoints", "scores", endpointScores(scoredEndpointsMap))
	return scoredEndpointsMap
}

func requestCount(ctx context.Context, endpoint scheduling.Endpoint) int64 {
	val, ok := endpoint.Get(attrconcurrency.InFlightLoadKey)
	if !ok {
		return 0
	}

	switch load := val.(type) {
	case *attrconcurrency.InFlightLoad:
		if load == nil {
			log.FromContext(ctx).V(logutil.TRACE).Info("Ignoring nil in-flight load attribute",
				"endpoint", endpoint.GetMetadata().NamespacedName.String())
			return 0
		}
		return load.Requests
	default:
		log.FromContext(ctx).V(logutil.TRACE).Info("Ignoring in-flight load attribute with unexpected type",
			"endpoint", endpoint.GetMetadata().NamespacedName.String(),
			"attributeType", fmt.Sprintf("%T", val))
		return 0
	}
}
