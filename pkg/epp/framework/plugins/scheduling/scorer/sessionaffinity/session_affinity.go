package sessionaffinity

import (
	"context"
	"encoding/base64"
	"encoding/json"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-inference-scheduler/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/scheduling"
)

const (
	// SessionAffinityType is the type of the SessionAffinity scorer.
	SessionAffinityType = "session-affinity-scorer"

	sessionTokenHeader = "x-session-token" // name of the session header in request
)

// compile-time type assertion
var _ scheduling.Scorer = &SessionAffinity{}
var _ requestcontrol.ResponseBodyProcessor = &SessionAffinity{}

// Factory defines the factory function for SessionAffinity scorer.
func Factory(name string, _ *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	return NewSessionAffinity().WithName(name), nil
}

// NewSessionAffinity returns a scorer
func NewSessionAffinity() *SessionAffinity {
	return &SessionAffinity{
		typedName: plugin.TypedName{Type: SessionAffinityType},
	}
}

// SessionAffinity is a routing scorer that routes subsequent
// requests in a session to the same pod as the first request in the
// session was sent to, by giving that pod the specified weight and assigning
// zero score to the rest of the targets
type SessionAffinity struct {
	typedName plugin.TypedName
}

// TypedName returns the typed name of the plugin.
func (s *SessionAffinity) TypedName() plugin.TypedName {
	return s.typedName
}

// WithName sets the name of the plugin.
func (s *SessionAffinity) WithName(name string) *SessionAffinity {
	s.typedName.Name = name
	return s
}

// Category returns the preference the scorer applies when scoring candidate endpoints.
func (s *SessionAffinity) Category() scheduling.ScorerCategory {
	return scheduling.Affinity
}

// Score assign a high score to the pod used in previous requests and zero to others
func (s *SessionAffinity) Score(ctx context.Context, _ *scheduling.CycleState, request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) map[scheduling.Endpoint]float64 {
	scoredEndpoints := make(map[scheduling.Endpoint]float64)
	sessionToken := request.Headers[sessionTokenHeader]
	podName := ""

	if sessionToken != "" {
		decodedBytes, err := base64.StdEncoding.DecodeString(sessionToken)
		if err != nil {
			log.FromContext(ctx).Error(err, "Error decoding session header")
		} else {
			podName = string(decodedBytes)
		}
	}
	for _, endpoint := range endpoints {
		scoredEndpoints[endpoint] = 0.0 // initial value
		if endpoint.GetMetadata().NamespacedName.String() == podName {
			scoredEndpoints[endpoint] = 1.0
		}
	}

	return scoredEndpoints
}

// ResponseBody sets the session header on the response sent to the client
// TODO: this should be using a cookie and ensure not overriding any other
// cookie values if present.
// Tracked in https://github.com/llm-d/llm-d-inference-scheduler/issues/28
func (s *SessionAffinity) ResponseBody(ctx context.Context, _ *scheduling.InferenceRequest, response *requestcontrol.Response, targetPod *datalayer.EndpointMetadata) {
	if !response.EndOfStream {
		return
	}
	if response == nil || targetPod == nil {
		reqID := "undefined"
		if response != nil {
			reqID = response.RequestID
		}
		log.FromContext(ctx).V(logutil.DEBUG).Info("Session affinity scorer - skip post response because one of response, targetPod is nil", "req id", reqID)
		return
	}

	if response.Headers == nil { // TODO should always be populated?
		response.Headers = make(map[string]string)
	}

	response.Headers[sessionTokenHeader] = base64.StdEncoding.EncodeToString([]byte(targetPod.NamespacedName.String()))
}
