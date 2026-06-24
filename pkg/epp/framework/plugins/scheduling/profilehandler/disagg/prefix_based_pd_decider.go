package disagg

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
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
}

func (p PrefixBasedPDDeciderConfig) validate() error {
	if p.NonCachedTokens < 0 {
		return errors.New("nonCachedTokens parameter of prefix disaggregation decider cannot be negative")
	}

	return nil
}

// compile-time type assertions
var (
	_ deciderPlugin                  = &PrefixBasedPDDecider{}
	_ fwkrc.ConditionalDecodeDecider = &PrefixBasedPDDecider{}
)

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

func (d *PrefixBasedPDDecider) disaggregate(ctx context.Context, request *scheduling.InferenceRequest, endpoint scheduling.Endpoint) bool {
	debugLogger := log.FromContext(ctx).V(logging.DEBUG)

	// NonCachedTokens defines the minimum number of non-cached tokens required
	// to trigger disaggregated PD. A value of 0 disables disaggregation.
	if d.config.NonCachedTokens == 0 {
		return false
	}
	nonCachedTokens, ok := d.computeNonCachedTokens(ctx, request, endpoint)
	if !ok {
		return false
	}
	if nonCachedTokens < d.config.NonCachedTokens {
		debugLogger.Info("Non-cached suffix is smaller than threshold, using decode profile only")
		return false // do not run prefill
	}
	return true
}

// ShouldRejectConditionalDecode reports whether a conditional-decode request
// (RFC 7240 "Prefer: if-available") should be rejected with HTTP 412 because
// the chosen decode endpoint's KV cache does not cover enough of the prompt.
//
// Returns true (reject) when the non-cached suffix meets or exceeds
// NonCachedTokens, or when the prefix cache state cannot be read from the
// endpoint. Returns false when NonCachedTokens is 0 (gate disabled) or the
// non-cached suffix is below the threshold.
func (d *PrefixBasedPDDecider) ShouldRejectConditionalDecode(ctx context.Context, request *scheduling.InferenceRequest, endpoint scheduling.Endpoint) bool {
	debugLogger := log.FromContext(ctx).V(logging.DEBUG)
	if d.config.NonCachedTokens == 0 {
		return false
	}
	nonCachedTokens, ok := d.computeNonCachedTokens(ctx, request, endpoint)
	if !ok {
		// Cannot read prefix cache state - fail closed (reject) to preserve
		// the current 412 behavior when no prefix-cache producer is wired up.
		return true
	}
	if nonCachedTokens < d.config.NonCachedTokens {
		debugLogger.Info("conditional-decode: non-cached suffix below threshold, forwarding",
			"nonCachedTokens", nonCachedTokens, "threshold", d.config.NonCachedTokens)
		return false
	}
	debugLogger.Info("conditional-decode: non-cached suffix at or above threshold, rejecting",
		"nonCachedTokens", nonCachedTokens, "threshold", d.config.NonCachedTokens)
	return true
}

// computeNonCachedTokens returns the length of the non-cached prompt suffix in
// tokens for the given endpoint. ok is false when the input length or the
// endpoint's PrefixCacheMatchInfo attribute could not be read.
//
// Uses the unweighted cached-block count, not the tier-weighted match score:
// a RAM-cached prefix must contribute its full token count, otherwise the
// non-cached suffix is overestimated and requests with large local-RAM hits
// are misrouted to remote prefill.
func (d *PrefixBasedPDDecider) computeNonCachedTokens(ctx context.Context, request *scheduling.InferenceRequest, endpoint scheduling.Endpoint) (int, bool) {
	logger := log.FromContext(ctx)
	debugLogger := logger.V(logging.DEBUG)

	if endpoint == nil {
		logger.Error(nil, "prefix decider: endpoint is nil")
		return 0, false
	}
	inputTokens, err := getUserInputLenInTokens(request)
	if err != nil {
		logger.Error(err, "prefix decider: failed to get user input length in tokens")
		return 0, false
	}
	prefixInfoRaw, ok := endpoint.Get(attrprefix.PrefixCacheMatchInfoDataKey.String())
	if !ok || prefixInfoRaw == nil {
		logger.Error(nil, "unable to read prefix cache state")
		return 0, false
	}
	prefixCacheMatchInfo, ok := prefixInfoRaw.(*attrprefix.PrefixCacheMatchInfo)
	if !ok {
		logger.Error(nil, "wrong type of prefix cache match info")
		return 0, false
	}

	hitPrefixTokens := prefixCacheMatchInfo.CachedBlockCount() * prefixCacheMatchInfo.BlockSizeTokens()
	nonCachedTokens := inputTokens - hitPrefixTokens
	debugLogger.Info("Computed hit percentage for prefix cache",
		"absolute hit prefix len (tokens)", hitPrefixTokens,
		"prompt length (token)", inputTokens)
	return nonCachedTokens, true
}

// getUserInputLenInTokens returns an estimated token count for the user input.
func getUserInputLenInTokens(request *scheduling.InferenceRequest) (int, error) {
	if request == nil || request.Body == nil {
		return 0, errors.New("request or request body is nil")
	}

	if tp := request.Body.TokenizedPrompt; tp != nil {
		return tp.TokenCount(), nil
	}
	return 0, nil
}
