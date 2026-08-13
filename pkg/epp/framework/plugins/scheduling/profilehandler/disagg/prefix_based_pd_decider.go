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
	tokenproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
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
	_ deciderPlugin         = &PrefixBasedPDDecider{}
	_ fwkrc.PreRequest      = &PrefixBasedPDDecider{}
	_ plugin.ConsumerPlugin = &PrefixBasedPDDecider{}
)

// errCondDecodeCacheMiss is the 412 returned by the conditional-decode gate
// when the chosen decoder cannot satisfy the request from its KV cache.
var errCondDecodeCacheMiss = errcommon.Error{
	Code: errcommon.PreconditionFailed,
	Msg:  "no decode worker has the requested KV cache",
}

// remotePrefillDecisionAttributeKey memoizes the needsRemotePrefill outcome on
// the request so PreRequest can reuse the decision that disaggregate computed
// earlier in scheduling.
var remotePrefillDecisionAttributeKey = plugin.NewDataKey("remote-prefill-decision", PrefixBasedPDDeciderPluginType)

type remotePrefillDecision struct {
	needs bool
	err   error
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

// Consumes declares the request- and endpoint-scoped data the plugin reads
// when evaluating the gate. Required so operators opting into the plugin
// standalone (not just as a disagg-profile-handler decider) get startup
// validation instead of a silent per-request forward when a producer is
// missing.
func (d *PrefixBasedPDDecider) Consumes() plugin.DataDependencies {
	return plugin.DataDependencies{
		Required: map[plugin.DataKey]any{
			attrprefix.PrefixCacheMatchInfoDataKey: attrprefix.PrefixCacheMatchInfo{},
			tokenproducer.TokenizedPromptDataKey:   scheduling.TokenizedRequest{},
		},
	}
}

// PreRequest gates requests carrying "Prefer: if-available". It returns HTTP
// 412 only when the gate legitimately fires — the chosen decode endpoint's
// non-cached suffix meets NonCachedTokens — or when the scheduling result
// has no primary decode endpoint (defensive). Any error evaluating the gate
// (missing tokenization, missing/wrong-typed cache attribute) is logged and
// the request is forwarded, matching disaggregate's soft-fail.
// Requests without the header are a no-op.
func (d *PrefixBasedPDDecider) PreRequest(ctx context.Context, request *scheduling.InferenceRequest,
	schedulingResult *scheduling.SchedulingResult) error {
	if request == nil || !routing.IsConditionalDecode(request.Headers) {
		return nil
	}
	// Claim ownership of the header for the director's default-deny check:
	// this plugin evaluated the preference, even if its own decision is to
	// forward (below).
	request.PutAttribute(fwkrc.ConditionalDecodeHandledAttributeKey, true)
	if d.config.NonCachedTokens == 0 {
		return nil
	}
	logger := log.FromContext(ctx)
	debugLogger := logger.V(logging.DEBUG)
	endpoint := primaryDecodeEndpoint(schedulingResult)
	if endpoint == nil {
		debugLogger.Info("conditional-decode: no primary decode endpoint, rejecting")
		return errCondDecodeCacheMiss
	}
	needs, err := d.needsRemotePrefill(ctx, request, endpoint)
	if err != nil {
		logger.Error(err, "conditional-decode: error evaluating gate, forwarding")
		return nil
	}
	if needs {
		debugLogger.Info("conditional-decode: non-cached suffix at or above threshold, rejecting")
		return errCondDecodeCacheMiss
	}
	debugLogger.Info("conditional-decode: forwarding")
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

// disaggregate reports whether remote prefill should run for this request.
// Fails soft: any read failure is logged and returns false so scheduling
// falls back to the decode-only path.
func (d *PrefixBasedPDDecider) disaggregate(ctx context.Context, request *scheduling.InferenceRequest, endpoint scheduling.Endpoint) bool {
	needs, err := d.needsRemotePrefill(ctx, request, endpoint)
	if err != nil {
		log.FromContext(ctx).Error(err, "prefix decider: error evaluating disaggregation, decode only")
		return false
	}
	return needs
}

// needsRemotePrefill answers whether the request's non-cached suffix on the
// chosen endpoint meets NonCachedTokens. Returns (false, nil) when the plugin
// is disabled, the prompt is shorter than PromptTokens, or the prompt is
// shorter than NonCachedTokens. A non-nil error means the endpoint's cache
// state or the request's input length could not be read; both callers
// soft-fail on error — disaggregate returns false, PreRequest forwards.
//
// The outcome is memoized on the request: disaggregate populates it during
// scheduling, and PreRequest reuses it without recomputing.
func (d *PrefixBasedPDDecider) needsRemotePrefill(ctx context.Context, request *scheduling.InferenceRequest, endpoint scheduling.Endpoint) (bool, error) {
	if request == nil {
		return false, errors.New("request is nil")
	}
	if cached, ok := scheduling.ReadRequestAttribute[remotePrefillDecision](request, remotePrefillDecisionAttributeKey); ok {
		return cached.needs, cached.err
	}
	needs, err := d.computeNeedsRemotePrefill(ctx, request, endpoint)
	request.PutAttribute(remotePrefillDecisionAttributeKey, remotePrefillDecision{needs: needs, err: err})
	return needs, err
}

// computeNeedsRemotePrefill is the uncached implementation of
// needsRemotePrefill. It reads the endpoint's unweighted cached-block count —
// not the tier-weighted match score — so a RAM-cached prefix contributes its
// full token count, otherwise the non-cached suffix is overestimated and large
// local-RAM hits are misrouted to remote prefill.
func (d *PrefixBasedPDDecider) computeNeedsRemotePrefill(ctx context.Context, request *scheduling.InferenceRequest, endpoint scheduling.Endpoint) (bool, error) {
	debugLogger := log.FromContext(ctx).V(logging.DEBUG)

	if d.config.NonCachedTokens == 0 {
		return false, nil
	}
	if endpoint == nil {
		return false, errors.New("endpoint is nil")
	}
	inputTokens, err := getUserInputLenInTokens(request)
	if err != nil {
		return false, fmt.Errorf("failed to get user input length in tokens: %w", err)
	}
	if d.config.PromptTokens > 0 && inputTokens < d.config.PromptTokens {
		debugLogger.Info("Input shorter than promptTokens, disaggregation not required",
			"inputTokens", inputTokens, "promptTokens", d.config.PromptTokens)
		return false, nil
	}
	if inputTokens < d.config.NonCachedTokens {
		debugLogger.Info("Input shorter than nonCachedTokens threshold, disaggregation not required",
			"inputTokens", inputTokens, "threshold", d.config.NonCachedTokens)
		return false, nil
	}
	prefixInfoRaw, ok := endpoint.Get(attrprefix.PrefixCacheMatchInfoDataKey)
	if !ok || prefixInfoRaw == nil {
		return false, errors.New("unable to read prefix cache state")
	}
	info, ok := prefixInfoRaw.(*attrprefix.PrefixCacheMatchInfo)
	if !ok {
		return false, fmt.Errorf("prefix cache match info has unexpected type: %T", prefixInfoRaw)
	}
	hitPrefixTokens := info.CachedBlockCount() * info.BlockSizeTokens()
	nonCachedTokens := inputTokens - hitPrefixTokens
	debugLogger.Info("Computed non-cached suffix",
		"hitPrefixTokens", hitPrefixTokens, "inputTokens", inputTokens,
		"nonCachedTokens", nonCachedTokens, "threshold", d.config.NonCachedTokens)
	return nonCachedTokens >= d.config.NonCachedTokens, nil
}

// getUserInputLenInTokens returns the token count carried on the request.
// A missing TokenizedRequest is reported as an error so the soft-fail branch
// in PreRequest and disaggregate surfaces the tokenizer failure in the log
// rather than silently treating an untokenized request as zero-length input.
func getUserInputLenInTokens(request *scheduling.InferenceRequest) (int, error) {
	if request == nil || request.Body == nil {
		return 0, errors.New("request or request body is nil")
	}
	if request.Body.TokenizedRequest == nil {
		return 0, errors.New("prompt not tokenized")
	}
	return request.Body.TokenizedRequest.TokenCount(), nil
}
