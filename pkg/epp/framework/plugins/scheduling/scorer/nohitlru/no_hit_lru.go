package nohitlru

import (
	"context"
	"encoding/json"
	"fmt"

	lru "github.com/hashicorp/golang-lru/v2"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-inference-scheduler/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/scheduling/scorer/prefix"
)

const (
	// NoHitLRUType is the type of the NoHitLRU scorer
	NoHitLRUType = "no-hit-lru-scorer"

	// defaultLRUSize is the maximum number of endpoints we'll consider in the cache
	defaultLRUSize = 1024

	// defaultPrefillProfile is the name of the prefill profile
	//
	// This is currently hardcoded until we have a defined proper config interface.
	// (See also https://github.com/kubernetes-sigs/gateway-api-inference-extension/pull/2104/	)
	defaultPrefillProfile = "prefill"
)

// compile-time type assertions
var _ scheduling.Scorer = &NoHitLRU{}
var _ requestcontrol.PreRequest = &NoHitLRU{}

// Parameters defines the parameters for the NoHitLRU scorer.
type Parameters struct {
	// PrefixPluginType defines the type of the prefix cache plugin to read state from.
	// Defaults to "prefix-cache-scorer".
	PrefixPluginType string `json:"prefixPluginType"`
	// PrefixPluginName defines the name of the prefix cache plugin to read state from.
	// Defaults to "prefix-cache-scorer".
	PrefixPluginName string `json:"prefixPluginName"`

	// LRUSize defines the maximum number of endpoints to track in the LRU cache.
	LRUSize int `json:"lruSize"`
}

// coldRequestState tracks whether a request triggered a KV cache hit
// when the cache is missed, isCold is true.
type coldRequestState struct {
	isCold bool
}

// Clone implements the plugin.StateData interface
func (c *coldRequestState) Clone() plugin.StateData {
	return &coldRequestState{isCold: c.isCold}
}

// Factory defines the factory function for the NoHitLRU
func Factory(name string, rawParameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	parameters := Parameters{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' scorer - %w", NoHitLRUType, err)
		}
	}

	if parameters.PrefixPluginName == "" {
		parameters.PrefixPluginName = prefix.PrefixCacheScorerPluginType
	}

	// Note: We don't enforce that the prefix plugin exists here
	// The scorer will gracefully handle missing prefix cache state as an optimization

	return NewNoHitLRU(handle.Context(), &parameters).WithName(name), nil
}

// NewNoHitLRU creates a new NoHitLRU scorer
func NewNoHitLRU(ctx context.Context, params *Parameters) *NoHitLRU {
	prefixPluginType := prefix.PrefixCacheScorerPluginType
	prefixPluginName := prefix.PrefixCacheScorerPluginType
	lruSize := defaultLRUSize

	if params != nil {
		if params.PrefixPluginType != "" {
			prefixPluginType = params.PrefixPluginType
		}
		if params.PrefixPluginName != "" {
			prefixPluginName = params.PrefixPluginName
		}
		if params.LRUSize > 0 {
			lruSize = params.LRUSize
		}
	}

	lruCache, err := lru.New[string, struct{}](lruSize)
	if err != nil {
		log.FromContext(ctx).Error(err, fmt.Sprintf("failed to initialize NoHitLRU scorer: could not create LRU cache with size %d", lruSize))
		return nil
	}

	return &NoHitLRU{
		typedName:             plugin.TypedName{Type: NoHitLRUType},
		lruCache:              lruCache,
		prefixPluginTypedName: plugin.TypedName{Type: prefixPluginType, Name: prefixPluginName},
		pluginState:           plugin.NewPluginState(ctx),
	}
}

// NoHitLRU scorer that favors endpoints that were least recently used for cold requests.
// This can help evenly distribute cache growth, since cold requests result in more
// new KV blocks.
type NoHitLRU struct {
	typedName             plugin.TypedName
	lruCache              *lru.Cache[string, struct{}] // endpoint name -> dummy value (we only care about order)
	prefixPluginTypedName plugin.TypedName
	pluginState           *plugin.PluginState
}

// TypedName returns the typed name of the plugin.
func (s *NoHitLRU) TypedName() plugin.TypedName {
	return s.typedName
}

// WithName sets the name of the plugin.
func (s *NoHitLRU) WithName(name string) *NoHitLRU {
	s.typedName.Name = name
	return s
}

// Category returns the preference the scorer applies when scoring candidate endpoints.
func (s *NoHitLRU) Category() scheduling.ScorerCategory {
	return scheduling.Distribution
}

// isColdRequest determines if a request is cold by checking endpoint prefix-cache attributes.
// Returns true when no endpoint reports any cache-hit blocks, or when no attribute is present.
func (s *NoHitLRU) isColdRequest(ctx context.Context, endpoints []scheduling.Endpoint) bool {
	logger := log.FromContext(ctx).V(logging.DEBUG)

	for _, ep := range endpoints {
		attr, ok := ep.Get(attrprefix.PrefixCacheMatchInfoKey)
		if !ok {
			continue
		}
		info, ok := attr.(*attrprefix.PrefixCacheMatchInfo)
		if ok && info.MatchBlocks() > 0 {
			logger.Info("Cache hit detected on endpoint", "endpoint", ep.GetMetadata().NamespacedName)
			return false
		}
	}
	return true
}

// scoreNeutral returns neutral scores (0.5) for all endpoints.
// Used when a request has cache hits and LRU optimization should not apply.
func (s *NoHitLRU) scoreNeutral(endpoints []scheduling.Endpoint) map[scheduling.Endpoint]float64 {
	scoredEndpoints := make(map[scheduling.Endpoint]float64, len(endpoints))
	for _, endpoint := range endpoints {
		scoredEndpoints[endpoint] = 0.5
	}
	return scoredEndpoints
}

// getLRUPositions returns a map of endpoint names to their LRU position.
// Position 0 represents the oldest (least recently used) entry.
func (s *NoHitLRU) getLRUPositions() map[string]int {
	// Get all keys from LRU cache in order (oldest first)
	// https://pkg.go.dev/github.com/hashicorp/golang-lru/v2#Cache.Keys
	lruKeys := s.lruCache.Keys()

	lruPosition := make(map[string]int, len(lruKeys))
	for i, key := range lruKeys {
		lruPosition[key] = i
	}
	return lruPosition
}

// partitionPodsByUsage separates endpoints into those that have received cold requests
// (usedPods) and those that have never received cold requests (neverUsedPods).
func (s *NoHitLRU) partitionPodsByUsage(endpoints []scheduling.Endpoint, lruPosition map[string]int) (usedEndpoints, neverUsedEndpoints []scheduling.Endpoint) {
	for _, endpoint := range endpoints {
		endpointName := endpoint.GetMetadata().NamespacedName.String()
		if _, exists := lruPosition[endpointName]; exists {
			usedEndpoints = append(usedEndpoints, endpoint)
		} else {
			neverUsedEndpoints = append(neverUsedEndpoints, endpoint)
		}
	}
	return usedEndpoints, neverUsedEndpoints
}

// scoreNeverUsedEndpoints assigns scores to endpoints that have never received a cold request.
// The first never-used endpoint gets the highest score (1.0), with subsequent endpoints
// receiving progressively lower scores.
func (s *NoHitLRU) scoreNeverUsedPods(scoredPods map[scheduling.Endpoint]float64, neverUsedPods []scheduling.Endpoint, totalEndpoints int) {
	// Avoid possibility of dividing by zero.
	if totalEndpoints <= 1 {
		return
	}
	for i, endpoint := range neverUsedPods {
		score := 1.0 - float64(i)/float64(totalEndpoints-1)
		scoredPods[endpoint] = score
	}
}

// scoreUsedPods assigns scores to endpoints based on their LRU position.
// Pods that were least recently used for cold requests receive higher scores.
func (s *NoHitLRU) scoreUsedPods(scoredEndpoints map[scheduling.Endpoint]float64, usedPods []scheduling.Endpoint, lruPosition map[string]int, neverUsedCount, totalEndpoints int) {
	// Avoid possibility of dividing by zero.
	if totalEndpoints <= 1 {
		return
	}
	for _, endpoint := range usedPods {
		endpointName := endpoint.GetMetadata().NamespacedName.String()
		lruPos := lruPosition[endpointName]
		// LRU keys are oldest to newest so rank 0 = oldest
		// The never used endpoint count is added to the rank so that
		// a never-used endpoint will always have the highest score.
		rank := neverUsedCount + lruPos
		score := 1.0 - float64(rank)/float64(totalEndpoints-1)
		if score < 0 {
			score = 0
		}
		scoredEndpoints[endpoint] = score
	}
}

// scoreColdRequestByLRU scores endpoints based on their LRU position for cold requests.
// Pods that have never received a cold request get the highest scores.
// Among previously used endpoints, least recently used ones get higher scores.
func (s *NoHitLRU) scoreColdRequestByLRU(endpoints []scheduling.Endpoint) map[scheduling.Endpoint]float64 {
	scoredEndpoints := make(map[scheduling.Endpoint]float64, len(endpoints))
	totalEndpoints := len(endpoints)

	// Avoid possibility of dividing by zero.
	if totalEndpoints == 1 {
		scoredEndpoints[endpoints[0]] = 1.0
		return scoredEndpoints
	}

	lruPosition := s.getLRUPositions()
	usedEndpoints, neverUsedEndpoints := s.partitionPodsByUsage(endpoints, lruPosition)

	s.scoreNeverUsedPods(scoredEndpoints, neverUsedEndpoints, totalEndpoints)
	s.scoreUsedPods(scoredEndpoints, usedEndpoints, lruPosition, len(neverUsedEndpoints), totalEndpoints)

	return scoredEndpoints
}

// Score scores the given endpoints based on LRU for cold requests.
// For cache hits, returns neutral scores (0.5) for all endpoints.
// For cache misses, ranks endpoints by their LRU order.
// - LRU ordering is with respect to when a endpoint last received a cold request.
// - Least recently used (or never used) endpoints get highest score (1.0)
// - Most recently used endpoints get lowest score (approaching 0.0)
func (s *NoHitLRU) Score(ctx context.Context, cycleState *scheduling.CycleState, request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) map[scheduling.Endpoint]float64 {
	logger := log.FromContext(ctx).V(logging.DEBUG)

	isCold := s.isColdRequest(ctx, endpoints)

	// Store the cold request state in plugin state for PreRequest to use
	coldState := &coldRequestState{isCold: isCold}
	s.pluginState.Write(request.RequestID, plugin.StateKey(s.typedName.String()), coldState)

	if !isCold {
		logger.Info("Cache hit detected, returning neutral scores")
		return s.scoreNeutral(endpoints)
	}

	logger.Info("Cold request detected, scoring endpoints by LRU")
	return s.scoreColdRequestByLRU(endpoints)
}

// PreRequest is called before a request is sent to the target endpoint.
// For cold requests, it updates the LRU cache to track which endpoints have been used recently.
func (s *NoHitLRU) PreRequest(ctx context.Context, request *scheduling.InferenceRequest, schedulingResult *scheduling.SchedulingResult) {
	logger := log.FromContext(ctx).V(logging.DEBUG)

	if schedulingResult == nil || len(schedulingResult.ProfileResults) == 0 {
		logger.Info("No scheduling result available")
		return
	}

	// Read the cold request state we stored in Score
	coldState, err := plugin.ReadPluginStateKey[*coldRequestState](s.pluginState, request.RequestID, plugin.StateKey(s.typedName.String()))
	// After fetching the cold state, drop it from the plugin state immediately (otherwise it will hang around until it becomes stale).
	s.pluginState.Delete(request.RequestID)

	if err != nil {
		logger.Info("No cold request state found, treating as non-cold request", "error", err)
		return
	}

	if !coldState.isCold {
		logger.Info("Not a cold request, skipping LRU update")
		return
	}

	if targetProfile, ok := schedulingResult.ProfileResults[schedulingResult.PrimaryProfileName]; ok && targetProfile != nil && len(targetProfile.TargetEndpoints) != 0 {
		s.moveTargetPodToFront(ctx, request, targetProfile, schedulingResult.PrimaryProfileName)
	}
	if targetProfile, ok := schedulingResult.ProfileResults[defaultPrefillProfile]; ok && targetProfile != nil && len(targetProfile.TargetEndpoints) != 0 {
		s.moveTargetPodToFront(ctx, request, targetProfile, defaultPrefillProfile)
	}
}

func (s *NoHitLRU) moveTargetPodToFront(ctx context.Context, request *scheduling.InferenceRequest, targetProfile *scheduling.ProfileRunResult, profileName string) {
	logger := log.FromContext(ctx).V(logging.DEBUG)

	targetPod := targetProfile.TargetEndpoints[0]
	endpointName := targetPod.GetMetadata().NamespacedName.String()

	// Move the endpoint to the front of the LRU.
	var present struct{} // dummy value
	s.lruCache.Add(endpointName, present)

	logger.Info("Updated LRU cache for cold request", "profile", profileName, "endpoint", endpointName, "requestId", request.RequestID)
}
