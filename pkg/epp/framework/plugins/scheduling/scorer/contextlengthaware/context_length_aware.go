package contextlengthaware

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-inference-scheduler/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/scheduling"
)

const (
	// ContextLengthAwareType is the type of the ContextLengthAware plugin
	ContextLengthAwareType = "context-length-aware"

	// DefaultContextLengthLabel is the default label name used to identify context length ranges on pods
	DefaultContextLengthLabel = "llm-d.ai/context-length-range"

	// charToTokenMultiplier defines the multiplier to convert characters to tokens
	// This is an approximate value and may vary based on the tokenizer used
	// Used as a fallback when no tokenizer is configured
	charToTokenMultiplier = 0.25
)

type contextLengthAwareParameters struct {
	// Label is the pod label name to check for context length range.
	// Format expected: "min-max" (e.g., "0-2048" or "2048-8192"), where min and max are positive integers.
	Label string `json:"label"`

	// EnableFiltering determines whether the plugin also filters pods that don't match a
	// request's context length.
	// If false, the plugin only scores pods.
	// Default is false.
	EnableFiltering bool `json:"enableFiltering"`
}

// contextRange represents a single context length range.
type contextRange struct {
	min int
	max int
}

var _ scheduling.Filter = &ContextLengthAware{} // validate interface conformance
var _ scheduling.Scorer = &ContextLengthAware{} // validate interface conformance

// Factory defines the factory function for the ContextLengthAware plugin.
func Factory(name string, rawParameters json.RawMessage, _ plugin.Handle) (plugin.Plugin, error) {
	parameters := &contextLengthAwareParameters{
		Label:           DefaultContextLengthLabel,
		EnableFiltering: false,
	}

	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' plugin - %w", ContextLengthAwareType, err)
		}
	}

	if parameters.Label == "" {
		return nil, fmt.Errorf("invalid configuration for '%s' plugin: 'label' must be specified", ContextLengthAwareType)
	}

	return NewContextLengthAware(name, parameters), nil
}

// NewContextLengthAware creates and returns an instance of the ContextLengthAware plugin.
func NewContextLengthAware(name string, params *contextLengthAwareParameters) *ContextLengthAware {
	return &ContextLengthAware{
		typedName:       plugin.TypedName{Type: ContextLengthAwareType, Name: name},
		labelName:       params.Label,
		enableFiltering: params.EnableFiltering,
	}
}

// ContextLengthAware is a plugin that filters or scores endpoints based on their association
// with input context length groups.
// It checks for a specific label on endpoints that defines the context length ranges they support.
// If filtering is enabled, endpoints that don't support the request's context length are filtered out.
// Additionally, it scores endpoints based on how well their context length ranges match the request.
//
// For precise token counting, this plugin reads InferenceRequestBody.TokenizedPrompt as
// populated by the tokenizer DataProducer plugin. When tokens are not available
// (tokenizer plugin not configured), it falls back to character-based estimation.
type ContextLengthAware struct {
	// typedName defines the plugin typed name
	typedName plugin.TypedName
	// labelName defines the name of the label to be checked
	labelName string
	// enableFiltering indicates whether filtering is enabled
	enableFiltering bool
}

// TypedName returns the typed name of the plugin.
func (p *ContextLengthAware) TypedName() plugin.TypedName {
	return p.typedName
}

// WithName sets the name of the plugin.
func (p *ContextLengthAware) WithName(name string) *ContextLengthAware {
	p.typedName.Name = name
	return p
}

// Filter filters out endpoints that don't have a context length range matching the request.
// This is only active when enableFiltering is true.
func (p *ContextLengthAware) Filter(ctx context.Context, cycleState *scheduling.CycleState, request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) []scheduling.Endpoint {
	if !p.enableFiltering {
		return endpoints // pass through if not in filter mode
	}

	logger := log.FromContext(ctx).V(logging.DEBUG).WithName("ContextLengthAware.Filter")
	contextLength, usedTokenizer := p.getContextLength(ctx, cycleState, request)
	logger.V(logging.TRACE).Info("Filtering endpoints by context length", "contextLength", contextLength, "usedTokenizer", usedTokenizer)

	filteredEndpoints := []scheduling.Endpoint{}

	for _, endpoint := range endpoints {
		metadata := endpoint.GetMetadata()
		if metadata == nil {
			// Endpoints without metadata are included (treated as no-label).
			filteredEndpoints = append(filteredEndpoints, endpoint)
			continue
		}

		rangeStr, hasLabel := metadata.Labels[p.labelName]
		if !hasLabel {
			// Endpoints without the label are included (they accept any context length).
			filteredEndpoints = append(filteredEndpoints, endpoint)
			continue
		}

		r, err := parseContextRange(rangeStr)
		if err != nil {
			logger.Error(err, "Failed to parse context range label", "endpoint", metadata.NamespacedName, "rangeStr", rangeStr)
			continue
		}

		if contextLength >= r.min && contextLength <= r.max {
			filteredEndpoints = append(filteredEndpoints, endpoint)
		}
	}

	logger.V(logging.TRACE).Info("Filtered endpoints", "originalCount", len(endpoints),
		"filteredCount", len(filteredEndpoints))
	return filteredEndpoints
}

// Score scores endpoints based on how well their context length ranges match the request.
// Endpoints with tighter/more specific ranges matching the request get higher scores.
func (p *ContextLengthAware) Score(ctx context.Context, cycleState *scheduling.CycleState, request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) map[scheduling.Endpoint]float64 {
	logger := log.FromContext(ctx).V(logging.DEBUG).WithName("ContextLengthAware.Score")
	contextLength, usedTokenizer := p.getContextLength(ctx, cycleState, request)
	logger.V(logging.TRACE).Info("Scoring endpoints by context length", "contextLength", contextLength, "usedTokenizer", usedTokenizer)

	scoredEndpoints := make(map[scheduling.Endpoint]float64)

	for _, endpoint := range endpoints {
		metadata := endpoint.GetMetadata()
		if metadata == nil {
			scoredEndpoints[endpoint] = 0.5
			continue
		}

		rangeStr, hasLabel := metadata.Labels[p.labelName]
		if !hasLabel {
			// Endpoints without the label get a neutral score
			scoredEndpoints[endpoint] = 0.5
			continue
		}

		r, err := parseContextRange(rangeStr)
		if err != nil {
			logger.Error(err, "Failed to parse context range label", "endpoint", metadata.NamespacedName, "rangeStr", rangeStr)
			scoredEndpoints[endpoint] = 0.0
			continue
		}

		score := calculateRangeScore(contextLength, r)
		scoredEndpoints[endpoint] = score
	}

	logger.V(logging.TRACE).Info("Scored endpoints", "scores", scoredEndpoints)
	return scoredEndpoints
}

// Category returns the preference the scorer applies when scoring candidate endpoints.
func (p *ContextLengthAware) Category() scheduling.ScorerCategory {
	return scheduling.Affinity
}

// getContextLength returns the context length (token count) for the request.
// It reads tokenized data from InferenceRequestBody.TokenizedPrompt as populated by the
// tokenizer DataProducer plugin, falling back to character-based estimation
// when tokens are not available.
// Returns the token count and a boolean indicating whether precise tokenization was used.
func (p *ContextLengthAware) getContextLength(ctx context.Context, _ *scheduling.CycleState, request *scheduling.InferenceRequest) (int, bool) {
	if request == nil || request.Body == nil {
		return 0, false
	}

	if tp := request.Body.TokenizedPrompt; tp != nil && len(tp.TokenIDs) > 0 {
		return len(tp.TokenIDs), true
	}

	logger := log.FromContext(ctx).V(logging.DEBUG).WithName("ContextLengthAware")
	logger.Info("TokenizedPrompt not available, falling back to character-based estimation")

	return estimateContextLength(request), false
}

// estimateContextLength estimates the context length from the request using character count.
// This is a fallback when no tokenizer is configured.
func estimateContextLength(request *scheduling.InferenceRequest) int {
	if request == nil || request.Body == nil {
		return 0
	}

	// Exact token count is available for pre-tokenized completion prompts;
	// prefer it over character-based estimation when present.
	if request.Body.Completions != nil {
		if hint := request.Body.Completions.Prompt.TokenCountHint(); hint >= 0 {
			return hint
		}
	}

	totalChars := 0

	// Handle chat completions
	if request.Body.ChatCompletions != nil {
		for _, msg := range request.Body.ChatCompletions.Messages {
			totalChars += len(msg.Content.Raw)
		}
	}

	// Handle regular completions
	if request.Body.Completions != nil {
		totalChars += len(request.Body.Completions.Prompt.PlainText())
	}

	// Convert characters to approximate token count
	estimatedTokens := int(float64(totalChars) * charToTokenMultiplier)
	return estimatedTokens
}

// parseContextRange parses a label value into a single context range.
// Expected format: "min-max", where min and max are positive integers.
// Examples: "0-2048", "2048-8192".
func parseContextRange(rangeStr string) (contextRange, error) {
	if rangeStr == "" {
		return contextRange{}, errors.New("empty range string")
	}

	bounds := strings.Split(rangeStr, "-")
	if len(bounds) != 2 {
		return contextRange{}, fmt.Errorf("invalid range format: %s (expected 'min-max')", rangeStr)
	}

	minVal, err := strconv.Atoi(strings.TrimSpace(bounds[0]))
	if err != nil {
		return contextRange{}, fmt.Errorf("invalid min value: %s", bounds[0])
	}

	maxVal, err := strconv.Atoi(strings.TrimSpace(bounds[1]))
	if err != nil {
		return contextRange{}, fmt.Errorf("invalid max value: %s", bounds[1])
	}

	if minVal > maxVal {
		return contextRange{}, fmt.Errorf("min (%d) cannot be greater than max (%d)", minVal, maxVal)
	}

	return contextRange{min: minVal, max: maxVal}, nil
}

// calculateRangeScore calculates a score for how well a pod's context range matches the request.
//
// Scoring tiers (higher tier always wins over lower):
//   - In-range match (0.3–1.0]: tighter ranges and more headroom score higher.
//     The minimum in-range score is strictly above maxFallbackScore, so an in-range
//     match always beats any out-of-range fallback.
//   - Out-of-range fallback [0.0–0.3): scored by proximity to the nearest boundary,
//     so the pod closest to the request is preferred for out-of-range requests.
//
// This ensures that out-of-range requests still get routed to the most reasonable pod
// rather than being scored equally (0.0) and selected arbitrarily.
func calculateRangeScore(contextLength int, r contextRange) float64 {
	const maxFallbackScore = 0.3

	// In-range match: score in (maxFallbackScore, 1.0].
	if contextLength >= r.min && contextLength <= r.max {
		rangeWidth := r.max - r.min
		if rangeWidth == 0 {
			return 1.0
		}

		// rawScore is in [0.0, 1.0]: tighter ranges and more headroom score higher.
		widthScore := 1.0 / (1.0 + float64(rangeWidth)/10000.0)
		headroom := float64(r.max - contextLength)
		positionScore := headroom / float64(rangeWidth)
		rawScore := 0.7*widthScore + 0.3*positionScore

		// Map rawScore from [0.0, 1.0] into (maxFallbackScore, 1.0].
		return maxFallbackScore + rawScore*(1.0-maxFallbackScore)
	}

	// Out-of-range fallback: score by proximity to the nearest range boundary.
	// Strictly below maxFallbackScore so a fallback never outscores an in-range match.
	var distance int
	if contextLength > r.max {
		distance = contextLength - r.max
	} else {
		distance = r.min - contextLength
	}

	return maxFallbackScore / (1.0 + float64(distance)/1000.0)
}
