package sessionaffinity

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/metadata"
)

const (
	// SessionAffinityPluginType is the type of the SessionAffinity plugin.
	SessionAffinityPluginType = "session-affinity"

	defaultSessionTokenHeader = "x-session-token"

	// SessionAffinityInfoKey is the key used to store SessionAffinityInfo in endpoint attributes.
	SessionAffinityInfoKey = "session-affinity-info"
)

// SessionAffinityInfo stores per-request affinity information on the endpoint.
type SessionAffinityInfo struct {
	IsTarget bool
}

// Clone makes a copy of SessionAffinityInfo.
func (s *SessionAffinityInfo) Clone() datalayer.Cloneable {
	return &SessionAffinityInfo{IsTarget: s.IsTarget}
}

// compile-time type assertions
var _ requestcontrol.DataProducer = &Plugin{}
var _ scheduling.Filter = &Plugin{}
var _ requestcontrol.PreRequest = &Plugin{}
var _ requestcontrol.ResponseHeaderProcessor = &Plugin{}

type sessionAffinityParameters struct {
	SessionTokenHeader string `json:"sessionTokenHeader,omitempty"`
}

// Factory defines the factory function for SessionAffinity plugin.
func Factory(name string, rawParameters json.RawMessage, handle plugin.Handle) (plugin.Plugin, error) {
	parameters := sessionAffinityParameters{
		SessionTokenHeader: defaultSessionTokenHeader,
	}
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' plugin - %w", SessionAffinityPluginType, err)
		}
	}
	if parameters.SessionTokenHeader == "" {
		parameters.SessionTokenHeader = defaultSessionTokenHeader
	}
	return NewPlugin(name, parameters.SessionTokenHeader), nil
}

// NewPlugin returns a new plugin instance.
func NewPlugin(name string, sessionTokenHeader string) *Plugin {
	return &Plugin{
		typedName:          plugin.TypedName{Type: SessionAffinityPluginType, Name: name},
		sessionTokenHeader: sessionTokenHeader,
		state:              make(map[string]string),
	}
}

// Plugin implements DataProducer, Filter, and ResponseHeaderProcessor to handle session affinity.
type Plugin struct {
	typedName          plugin.TypedName
	sessionTokenHeader string

	mu    sync.RWMutex
	state map[string]string // sessionID -> endpointName
}

// TypedName returns the typed name of the plugin.
func (p *Plugin) TypedName() plugin.TypedName {
	return p.typedName
}

// Produces returns the data keys this plugin produces.
func (p *Plugin) Produces() map[string]any {
	return map[string]any{SessionAffinityInfoKey: &SessionAffinityInfo{}}
}

// Consumes returns the data keys this plugin requires.
func (p *Plugin) Consumes() map[string]any {
	return nil
}

// PrepareRequestData resolves the affinity early and stores it in endpoint attributes.
func (p *Plugin) PrepareRequestData(ctx context.Context, request *scheduling.InferenceRequest, pods []scheduling.Endpoint) error {
	sessionID := request.Headers[p.sessionTokenHeader]

	// If no session ID, mark all as potential targets
	if sessionID == "" {
		for _, pod := range pods {
			pod.Put(SessionAffinityInfoKey, &SessionAffinityInfo{IsTarget: true})
		}
		return nil
	}

	p.mu.RLock()
	mappedEndpointName, ok := p.state[sessionID]
	p.mu.RUnlock()

	// If no mapping, mark all as potential targets
	if !ok {
		for _, pod := range pods {
			pod.Put(SessionAffinityInfoKey, &SessionAffinityInfo{IsTarget: true})
		}
		return nil
	}

	// Check if the mapped endpoint is available
	found := false
	for _, endpoint := range pods {
		if endpoint.GetMetadata().NamespacedName.String() == mappedEndpointName {
			found = true
			break
		}
	}

	if found {
		for _, pod := range pods {
			isTarget := pod.GetMetadata().NamespacedName.String() == mappedEndpointName
			pod.Put(SessionAffinityInfoKey, &SessionAffinityInfo{IsTarget: isTarget})
		}
		return nil
	}

	// Mapped endpoint is no longer available
	p.mu.Lock()
	if currentMapping, exists := p.state[sessionID]; exists && currentMapping == mappedEndpointName {
		delete(p.state, sessionID)
	}
	p.mu.Unlock()

	log.FromContext(ctx).V(1).Info("Mapped endpoint for session is no longer available, removing mapping", "sessionID", sessionID, "endpoint", mappedEndpointName)

	// Mark all as targets for fallback
	for _, pod := range pods {
		pod.Put(SessionAffinityInfoKey, &SessionAffinityInfo{IsTarget: true})
	}

	return nil
}

// Filter reads the target endpoint from endpoint attributes and enforces it.
func (p *Plugin) Filter(_ context.Context, _ *scheduling.CycleState, _ *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) []scheduling.Endpoint {
	filteredEndpoints := []scheduling.Endpoint{}

	for _, endpoint := range endpoints {
		val, ok := endpoint.Get(SessionAffinityInfoKey)
		if !ok {
			// If info is missing, assume it's a target to be safe (or filter it out? Let's keep it to be safe).
			filteredEndpoints = append(filteredEndpoints, endpoint)
			continue
		}

		info, ok := val.(*SessionAffinityInfo)
		if !ok {
			// Invalid type, keep it to be safe
			filteredEndpoints = append(filteredEndpoints, endpoint)
			continue
		}

		if info.IsTarget {
			filteredEndpoints = append(filteredEndpoints, endpoint)
		}
	}

	return filteredEndpoints
}

// PreRequest updates the mapping as soon as a scheduling decision is made.
func (p *Plugin) PreRequest(ctx context.Context, request *scheduling.InferenceRequest, schedulingResult *scheduling.SchedulingResult) {
	sessionID := request.Headers[p.sessionTokenHeader]
	if sessionID == "" {
		return
	}

	primaryProfileResult, ok := schedulingResult.ProfileResults[schedulingResult.PrimaryProfileName]
	if !ok || len(primaryProfileResult.TargetEndpoints) == 0 {
		return
	}

	selectedEndpoint := primaryProfileResult.TargetEndpoints[0]
	endpointName := selectedEndpoint.GetMetadata().NamespacedName.String()

	p.mu.Lock()
	defer p.mu.Unlock()

	currentMapping, exists := p.state[sessionID]
	if !exists || currentMapping != endpointName {
		p.state[sessionID] = endpointName
		log.FromContext(ctx).V(1).Info("Updated session affinity mapping in PreRequest", "sessionID", sessionID, "endpoint", endpointName)
	}
}

// ResponseHeader learns the mapping by checking the response headers.
func (p *Plugin) ResponseHeader(ctx context.Context, request *scheduling.InferenceRequest, response *requestcontrol.Response, _ *datalayer.EndpointMetadata) {
	sessionID := request.Headers[p.sessionTokenHeader]
	if sessionID == "" {
		return
	}

	servedEndpoint := response.Headers[metadata.DestinationEndpointServedKey]
	if servedEndpoint == "" {
		return
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	currentMapping, exists := p.state[sessionID]
	if !exists || currentMapping != servedEndpoint {
		p.state[sessionID] = servedEndpoint
		log.FromContext(ctx).V(1).Info("Updated session affinity mapping in ResponseHeader", "sessionID", sessionID, "endpoint", servedEndpoint)
	}
}
