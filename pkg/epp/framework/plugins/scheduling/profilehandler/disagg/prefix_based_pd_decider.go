package disagg

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-inference-scheduler/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/metrics"
)

const (
	// PrefixBasedPDDeciderPluginType is the type-name of the prefixBasedPDDecider plugin.
	PrefixBasedPDDeciderPluginType = "prefix-based-pd-decider"

	// AverageCharactersPerToken is an estimated average characters per token,
	// used since the request we cache is not tokenized.
	AverageCharactersPerToken = 4
)

// PrefixBasedPDDeciderConfig holds the configuration for the prefixBasedPDDecider plugin.
type PrefixBasedPDDeciderConfig struct {
	// NonCachedTokens non cached minimum tokens that triggers disaggregated PD
	NonCachedTokens int `json:"nonCachedTokens"`
}

func (p PrefixBasedPDDeciderConfig) validate() error {
	if p.NonCachedTokens < 0 {
		return errors.New("nonCachedTokens parameter of prefix disaggregation decider cannot be negative")
	}

	return nil
}

// compile-time type assertion
var _ deciderPlugin = &PrefixBasedPDDecider{}

// PrefixBasedPDDecider is a PD decider plugin which decision is based prefix aware
type PrefixBasedPDDecider struct {
	typedName plugin.TypedName
	config    PrefixBasedPDDeciderConfig
}

// PrefixBasedPDDeciderPluginFactory defines the factory function for creating
// a new instance of the prefixBasedPDDecider.
func PrefixBasedPDDeciderPluginFactory(name string, rawParameters json.RawMessage,
	handle plugin.Handle) (plugin.Plugin, error) {
	config := PrefixBasedPDDeciderConfig{
		NonCachedTokens: 0,
	}

	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &config); err != nil {
			return nil, fmt.Errorf("failed to parse %s plugin config: %w", PrefixBasedPDDeciderPluginType, err)
		}
	}

	decider, err := NewPrefixBasedPDDecider(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create %s plugin: %w", PrefixBasedPDDeciderPluginType, err)
	}

	return decider.WithName(name), nil
}

// NewPrefixBasedPDDecider initializes a NewPrefixBasedPDDecider prefix based PD decider Plugin and returns its pointer.
// If the configuration is invalid an error is returned.
func NewPrefixBasedPDDecider(config PrefixBasedPDDeciderConfig) (*PrefixBasedPDDecider, error) {
	if err := config.validate(); err != nil {
		return nil, err
	}

	if config.NonCachedTokens == 0 {
		log.Log.Info("Prefix-based PD disabled (NonCachedTokens=0)")
	}

	return &PrefixBasedPDDecider{
		typedName: plugin.TypedName{Type: PrefixBasedPDDeciderPluginType},
		config:    config,
	}, nil
}

// TypedName returns the typed name of the plugin.
func (d *PrefixBasedPDDecider) TypedName() plugin.TypedName {
	return d.typedName
}

// WithName sets the name of the plugin.
func (d *PrefixBasedPDDecider) WithName(name string) *PrefixBasedPDDecider {
	d.typedName.Name = name
	return d
}

func (d *PrefixBasedPDDecider) disaggregate(ctx context.Context, request *scheduling.InferenceRequest, endpoint scheduling.Endpoint) bool {
	logger := log.FromContext(ctx)
	debugLogger := log.FromContext(ctx).V(logging.DEBUG)

	modelName := ""
	if request != nil {
		modelName = request.TargetModel
	}
	deciderName := d.typedName.Type

	// NonCachedTokens defines the minimum number of non-cached tokens required
	// to trigger disaggregated PD. A value of 0 disables disaggregation.
	if d.config.NonCachedTokens == 0 {
		metrics.RecordDeciderEvaluation(modelName, deciderName, metrics.DeciderReasonDisabled)
		return false
	}
	if endpoint == nil {
		logger.Error(nil, "prefix decider: endpoint is nil")
		metrics.RecordDeciderEvaluation(modelName, deciderName, metrics.DeciderReasonError)
		return false
	}
	inputTokens, err := getUserInputLenInTokens(request)
	if err != nil {
		logger.Error(err, "prefix decider: failed to get user input length in tokens")
		metrics.RecordDeciderEvaluation(modelName, deciderName, metrics.DeciderReasonError)
		return false
	}
	if inputTokens < d.config.NonCachedTokens {
		debugLogger.Info("Input is shorter than the nonCachedToken, no disaggregated PD")
		metrics.RecordDeciderEvaluation(modelName, deciderName, metrics.DeciderReasonInputTooShort)
		return false
	}
	// inspect the decode endpoint to disaggregate if prefill should run or not.
	// if the non-cached part is short enough - no disaggregation.
	prefixInfoRaw, ok := endpoint.Get(attrprefix.PrefixCacheMatchInfoKey)
	if !ok || prefixInfoRaw == nil {
		logger.Error(nil, "unable to read prefix cache state")
		metrics.RecordDeciderEvaluation(modelName, deciderName, metrics.DeciderReasonError)
		return false
	}
	prefixCacheMatchInfo, ok := prefixInfoRaw.(*attrprefix.PrefixCacheMatchInfo)
	if !ok {
		logger.Error(nil, "wrong type of prefix cache match info")
		metrics.RecordDeciderEvaluation(modelName, deciderName, metrics.DeciderReasonError)
		return false
	}

	// number of cached tokens
	hitPrefixTokens := prefixCacheMatchInfo.MatchBlocks() * prefixCacheMatchInfo.BlockSizeTokens()
	// length of non-cached suffix in tokens
	nonCachedTokens := inputTokens - hitPrefixTokens

	debugLogger.Info("Computed hit percentage for prefix cache",
		"absolute hit prefix len (tokens)", hitPrefixTokens,
		"prompt length (token)", inputTokens)

	if nonCachedTokens < d.config.NonCachedTokens {
		debugLogger.Info("Non-cached suffix is smaller than threshold, using decode profile only")
		metrics.RecordDeciderEvaluation(modelName, deciderName, metrics.DeciderReasonSuffixCached)
		return false // do not run prefill
	}

	metrics.RecordDeciderEvaluation(modelName, deciderName, metrics.DeciderReasonDisaggregated)
	return true
}

// getUserInputLenInTokens returns an estimated token count for the user input.
func getUserInputLenInTokens(request *scheduling.InferenceRequest) (int, error) {
	if request == nil || request.Body == nil {
		return 0, errors.New("request or request body is nil")
	}
	if request.Body.Completions != nil {
		return len(request.Body.Completions.Prompt.Raw) / AverageCharactersPerToken, nil
	}
	if request.Body.ChatCompletions == nil {
		return 0, errors.New("request has neither completions nor chat completions body")
	}
	prompt, err := json.Marshal(request.Body.ChatCompletions.Messages)
	if err != nil {
		return 0, err
	}
	return len(prompt) / AverageCharactersPerToken, nil
}
