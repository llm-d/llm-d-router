package disagg

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"sigs.k8s.io/controller-runtime/pkg/log"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

const (
	// PrefixBasedPDDeciderPluginType is the type-name of the prefixBasedPDDecider plugin.
	PrefixBasedPDDeciderPluginType = "prefix-based-pd-decider"
)

// PrefixBasedPDDeciderConfig holds the configuration for the prefixBasedPDDecider plugin.
type PrefixBasedPDDeciderConfig struct {
	// NonCachedTokens non cached minimum tokens that triggers disaggregated PD
	NonCachedTokens int `json:"nonCachedTokens"`

	// PromptTokens is the minimum estimated prompt length in tokens (approximated
	// as character-count / AverageCharactersPerToken) required before applying
	// prefix-cache-based disaggregation logic. Zero disables this prompt-length gate.
	PromptTokens int `json:"promptTokens"`
}

func (p PrefixBasedPDDeciderConfig) validate() error {
	if p.NonCachedTokens < 0 {
		return errors.New("nonCachedTokens parameter of prefix disaggregation decider cannot be negative")
	}

	if p.PromptTokens < 0 {
		return errors.New("promptTokens parameter of prefix disaggregation decider cannot be negative")
	}

	return nil
}

// compile-time type assertions
var (
	_ deciderPlugin    = &PrefixBasedPDDecider{}
	_ fwkrc.PreRequest = &PrefixBasedPDDecider{}
)

// errCondDecodeCacheMiss is the 412 returned by the conditional-decode gate
// when the chosen decoder cannot satisfy the request from its KV cache.
var errCondDecodeCacheMiss = errcommon.Error{
	Code: errcommon.PreconditionFailed,
	Msg:  "no decode worker has the requested KV cache",
}

// PrefixBasedPDDecider is a PD decider plugin which decision is based prefix aware
type PrefixBasedPDDecider struct {
	typedName plugin.TypedName
	config    PrefixBasedPDDeciderConfig
}

// PrefixBasedPDDeciderPluginFactory defines the factory function for creating
// a new instance of the prefixBasedPDDecider.
func PrefixBasedPDDeciderPluginFactory(name string, rawParameters *json.Decoder,
	handle plugin.Handle) (plugin.Plugin, error) {
	config := PrefixBasedPDDeciderConfig{
		NonCachedTokens: 0,
		PromptTokens:    0,
	}

	if rawParameters != nil {
		if err := rawParameters.Decode(&config); err != nil {
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

// PreRequest gates requests carrying "Prefer: if-available". It rejects with
// HTTP 412 when the non-cached suffix on the chosen decode endpoint reaches
// NonCachedTokens, mirroring disaggregate()'s promptTokens/NonCachedTokens
// shortcuts so both decisions honor the same knobs. NonCachedTokens == 0
// disables the gate. Fails closed (412) when the endpoint's cache state
// cannot be read.
func (d *PrefixBasedPDDecider) PreRequest(ctx context.Context, request *scheduling.InferenceRequest,
	schedulingResult *scheduling.SchedulingResult) error {
	if request == nil || !routing.IsConditionalDecode(request.Headers) {
		return nil
	}
	if d.config.NonCachedTokens == 0 {
		return nil
	}
	debugLogger := log.FromContext(ctx).V(logging.DEBUG)

	endpoint := primaryDecodeEndpoint(schedulingResult)
	if endpoint == nil {
		debugLogger.Info("conditional-decode: no primary decode endpoint, rejecting")
		return errCondDecodeCacheMiss
	}
	prefixInfoRaw, ok := endpoint.Get(attrprefix.PrefixCacheMatchInfoDataKey)
	if !ok || prefixInfoRaw == nil {
		debugLogger.Info("conditional-decode: endpoint has no prefix-cache match info, rejecting")
		return errCondDecodeCacheMiss
	}
	info, ok := prefixInfoRaw.(*attrprefix.PrefixCacheMatchInfo)
	if !ok {
		debugLogger.Info("conditional-decode: prefix-cache match info has unexpected type, rejecting",
			"type", fmt.Sprintf("%T", prefixInfoRaw))
		return errCondDecodeCacheMiss
	}
	inputTokens, err := getUserInputLenInTokens(request)
	if err != nil {
		debugLogger.Info("conditional-decode: failed to read input token count, rejecting", "error", err)
		return errCondDecodeCacheMiss
	}
	if d.config.PromptTokens > 0 && inputTokens < d.config.PromptTokens {
		debugLogger.Info("conditional-decode: prompt below promptTokens threshold, forwarding",
			"inputTokens", inputTokens, "promptTokens", d.config.PromptTokens)
		return nil
	}
	hitPrefixTokens := info.CachedBlockCount() * info.BlockSizeTokens()
	nonCachedTokens := inputTokens - hitPrefixTokens
	if nonCachedTokens >= d.config.NonCachedTokens {
		debugLogger.Info("conditional-decode: non-cached suffix at or above threshold, rejecting",
			"nonCachedTokens", nonCachedTokens, "threshold", d.config.NonCachedTokens)
		return errCondDecodeCacheMiss
	}
	debugLogger.Info("conditional-decode: non-cached suffix below threshold, forwarding",
		"nonCachedTokens", nonCachedTokens, "threshold", d.config.NonCachedTokens)
	return nil
}

// primaryDecodeEndpoint returns the first endpoint chosen by the primary
// profile, or nil when the scheduling result is missing, malformed, or the
// primary profile produced no endpoint.
func primaryDecodeEndpoint(result *scheduling.SchedulingResult) scheduling.Endpoint {
	if result == nil || result.PrimaryProfileName == "" || result.ProfileResults == nil {
		return nil
	}
	primary := result.ProfileResults[result.PrimaryProfileName]
	if primary == nil || len(primary.TargetEndpoints) == 0 {
		return nil
	}
	return primary.TargetEndpoints[0]
}

func (d *PrefixBasedPDDecider) disaggregate(ctx context.Context, request *scheduling.InferenceRequest, endpoint scheduling.Endpoint) bool {
	logger := log.FromContext(ctx)
	debugLogger := log.FromContext(ctx).V(logging.DEBUG)

	// NonCachedTokens defines the minimum number of non-cached tokens required
	// to trigger disaggregated PD. A value of 0 disables disaggregation.
	if d.config.NonCachedTokens == 0 {
		return false
	}
	if endpoint == nil {
		logger.Error(nil, "prefix decider: endpoint is nil")
		return false
	}
	inputTokens, err := getUserInputLenInTokens(request)
	if err != nil {
		logger.Error(err, "prefix decider: failed to get user input length in tokens")
		return false
	}

	if d.config.PromptTokens > 0 && inputTokens < d.config.PromptTokens {
		debugLogger.Info("Input is shorter than the promptTokens, no disaggregated PD")
		return false
	}

	if inputTokens < d.config.NonCachedTokens {
		debugLogger.Info("Input is shorter than the nonCachedToken, no disaggregated PD")
		return false
	}
	// inspect the decode endpoint to disaggregate if prefill should run or not.
	// if the non-cached part is short enough - no disaggregation.
	prefixInfoRaw, ok := endpoint.Get(attrprefix.PrefixCacheMatchInfoDataKey)
	if !ok || prefixInfoRaw == nil {
		logger.Error(nil, "unable to read prefix cache state")
		return false
	}
	prefixCacheMatchInfo, ok := prefixInfoRaw.(*attrprefix.PrefixCacheMatchInfo)
	if !ok {
		logger.Error(nil, "wrong type of prefix cache match info")
		return false
	}

	// number of cached tokens. Use the unweighted cached-block count, not the
	// tier-weighted match score: a RAM-cached prefix must contribute its full
	// token count here, otherwise the non-cached suffix is overestimated and
	// requests with large local-RAM hits are misrouted to remote prefill.
	hitPrefixTokens := prefixCacheMatchInfo.CachedBlockCount() * prefixCacheMatchInfo.BlockSizeTokens()
	// length of non-cached suffix in tokens
	nonCachedTokens := inputTokens - hitPrefixTokens

	debugLogger.Info("Computed hit percentage for prefix cache",
		"absolute hit prefix len (tokens)", hitPrefixTokens,
		"prompt length (token)", inputTokens)

	if nonCachedTokens < d.config.NonCachedTokens {
		debugLogger.Info("Non-cached suffix is smaller than threshold, using decode profile only")
		return false // do not run prefill
	}

	return true
}

// getUserInputLenInTokens returns an estimated token count for the user input.
func getUserInputLenInTokens(request *scheduling.InferenceRequest) (int, error) {
	if request == nil || request.Body == nil {
		return 0, errors.New("request or request body is nil")
	}

	if tp := request.Body.TokenizedRequest; tp != nil {
		return tp.TokenCount(), nil
	}
	return 0, nil
}
