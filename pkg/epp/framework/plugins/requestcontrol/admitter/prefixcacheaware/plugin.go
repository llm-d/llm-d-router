/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package prefixcacheaware provides an Admitter for cache-aware load shedding:
// it denies cache-cold requests while the pool is near saturation. Below the
// shed threshold every request is admitted. At or above it, a request is admitted only if some endpoint has a
// prefix cache hit ratio of at least minHitRatio; otherwise it is denied with
// the same typed error flow control uses for a saturated pool (429 by default,
// 503 optional, reason "rejected-saturated"), so clients see ordinary
// backpressure and nothing about the cache state behind the decision.
//
// The Admitter runs after the data producers and before scheduling, so it sees
// the full candidate pool and the request's prefix match on every endpoint. It
// does not narrow the candidate set; an admitted hot request is routed by the
// profile's own filters and scorers (typically the prefix cache affinity
// filter), which keep it on its warm endpoint. Whether that endpoint has room
// is left to the scheduling profile, as it is for every admitted request.
package prefixcacheaware

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"

	"sigs.k8s.io/controller-runtime/pkg/log"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

const (
	PluginType = "prefix-cache-aware-admitter"
)

var (
	_ fwkrc.Admitter           = &Plugin{}
	_ fwkplugin.ConsumerPlugin = &Plugin{}
)

// Plugin is the prefix-cache-aware admitter.
type Plugin struct {
	typedName                    fwkplugin.TypedName
	config                       Config
	errorCode                    string
	detector                     flowcontrol.SaturationDetector
	prefixMatchDataKey           fwkplugin.DataKey
	inFlightLoadDataKey          fwkplugin.DataKey
	uncachedRequestTokensDataKey fwkplugin.DataKey
}

// Factory builds the admitter. The referenced detector must already be
// instantiated; the config loader guarantees this through the pluginRef tag.
func Factory(name string, rawParameters *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	raw, err := ConfigParser(rawParameters, handle)
	if err != nil {
		return nil, err
	}
	config := raw.(*Config)
	if handle == nil {
		return nil, fmt.Errorf("%s requires a plugin handle to resolve saturationDetectorRef", PluginType)
	}
	p := handle.Plugin(config.SaturationDetectorRef)
	if p == nil {
		return nil, fmt.Errorf("saturationDetectorRef plugin not found: %s", config.SaturationDetectorRef)
	}
	det, ok := p.(flowcontrol.SaturationDetector)
	if !ok {
		return nil, fmt.Errorf("plugin %s must implement flowcontrol.SaturationDetector", config.SaturationDetectorRef)
	}
	if err := registerMetrics(handle.Metrics()); err != nil {
		return nil, err
	}
	return New(name, *config, det)
}

// New builds the admitter around an already resolved detector. It validates
// the config, so a caller cannot end up with an empty error code.
func New(name string, config Config, det flowcontrol.SaturationDetector) (*Plugin, error) {
	if err := config.validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}
	if det == nil {
		return nil, errors.New("saturation detector is required")
	}
	code, err := config.errorCode()
	if err != nil {
		return nil, err
	}
	return &Plugin{
		typedName:                    fwkplugin.TypedName{Type: PluginType, Name: name},
		config:                       config,
		errorCode:                    code,
		detector:                     det,
		prefixMatchDataKey:           attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(config.PrefixMatchInfoProducerName),
		inFlightLoadDataKey:          attrconcurrency.InFlightLoadDataKey.WithNonEmptyProducerName(config.InFlightLoadProducerName),
		uncachedRequestTokensDataKey: attrconcurrency.UncachedRequestTokensDataKey.WithNonEmptyProducerName(config.InFlightLoadProducerName),
	}, nil
}

func (p *Plugin) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// Consumes declares the endpoint attributes the admitter and its detector read.
// UncachedRequestTokens is required only when the small-request exemption is
// on; a missing value never exempts a request (see smallestUncached).
func (p *Plugin) Consumes() fwkplugin.DataDependencies {
	deps := fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{
			p.prefixMatchDataKey:  attrprefix.PrefixCacheMatchInfo{},
			p.inFlightLoadDataKey: attrconcurrency.InFlightLoad{},
		},
		Optional: map[fwkplugin.DataKey]any{},
	}
	if p.config.MinUncachedTokens > 0 {
		deps.Required[p.uncachedRequestTokensDataKey] = attrconcurrency.UncachedRequestTokens{}
	} else {
		deps.Optional[p.uncachedRequestTokensDataKey] = attrconcurrency.UncachedRequestTokens{}
	}
	return deps
}

// shedMessage is the response body of a shed. It is deliberately the generic
// saturation wording: the decision inputs go to the debug log, not to the client.
const shedMessage = "pool saturated, request rejected"

// Admit applies the shed rule. It returns nil to admit, or a typed
// errcommon.Error carrying the configured status code and the
// rejected-saturated reason header to deny. See the package documentation.
//
// The request is intentionally unused: everything the rule needs, the prefix
// match and the uncached token count, has already been projected onto the
// endpoints by the data producers. A nil request is therefore fine.
func (p *Plugin) Admit(ctx context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) error {
	logger := log.FromContext(ctx).V(logutil.DEBUG)
	if len(endpoints) == 0 {
		recordDecision(p.typedName.Name, outcomeNotApplicable)
		return nil
	}

	saturation := p.detector.Saturation(ctx, asPool(endpoints))
	recordSaturation(p.typedName.Name, saturation)
	if saturation < p.config.ShedThreshold {
		recordDecision(p.typedName.Name, outcomeAdmittedBelowThreshold)
		return nil
	}

	hot := 0
	bestHit := 0.0
	for _, ep := range endpoints {
		if ep == nil {
			continue
		}
		ratio := p.hitRatio(ep)
		bestHit = max(bestHit, ratio)
		if ratio >= p.config.MinHitRatio {
			hot++
		}
	}

	if hot > 0 {
		logger.Info("PrefixCacheAwareAdmitter: shedding, hot request admitted",
			"saturation", saturation, "hotEndpoints", hot, "endpoints", len(endpoints))
		recordDecision(p.typedName.Name, outcomeAdmittedHot)
		return nil
	}

	if p.config.MinUncachedTokens > 0 {
		if smallest, known := p.smallestUncached(endpoints); known && smallest < p.config.MinUncachedTokens {
			logger.Info("PrefixCacheAwareAdmitter: shedding, small request exempt",
				"saturation", saturation, "uncachedTokens", smallest, "minUncachedTokens", p.config.MinUncachedTokens)
			recordDecision(p.typedName.Name, outcomeAdmittedSmallRequest)
			return nil
		}
	}

	logger.Info("PrefixCacheAwareAdmitter: shedding cold request",
		"saturation", saturation, "bestHitRatio", bestHit, "minHitRatio", p.config.MinHitRatio, "endpoints", len(endpoints))
	recordDecision(p.typedName.Name, outcomeShedCold)
	return errcommon.Error{
		Code:    p.errorCode,
		Msg:     shedMessage,
		Headers: map[string]string{errcommon.RequestDroppedReasonHeaderKey: string(errcommon.RequestDroppedReasonSaturated)},
	}
}

func (p *Plugin) hitRatio(ep fwksched.Endpoint) float64 {
	raw, ok := ep.Get(p.prefixMatchDataKey)
	if !ok {
		return 0
	}
	info, ok := raw.(*attrprefix.PrefixCacheMatchInfo)
	if !ok || info == nil || info.TotalBlocks() <= 0 {
		return 0
	}
	return float64(info.MatchBlocks()) / float64(info.TotalBlocks())
}

// uncachedTokens reads the request's uncached token projection on one endpoint.
// The second result is false when the endpoint has no such attribute.
func (p *Plugin) uncachedTokens(ep fwksched.Endpoint) (int64, bool) {
	raw, ok := ep.Get(p.uncachedRequestTokensDataKey)
	if !ok {
		return 0, false
	}
	tokens, ok := raw.(*attrconcurrency.UncachedRequestTokens)
	if !ok || tokens == nil {
		return 0, false
	}
	return tokens.Tokens, true
}

// smallestUncached is the uncached token count of the request on the endpoint
// where it is cheapest, i.e. the least prefill the pool would have to do for it.
// Endpoints without the attribute are skipped; if none has it the count is
// unknown and the caller must not exempt the request on the strength of it.
func (p *Plugin) smallestUncached(endpoints []fwksched.Endpoint) (smallest int64, known bool) {
	smallest = math.MaxInt64
	for _, ep := range endpoints {
		if ep == nil {
			continue
		}
		if tokens, ok := p.uncachedTokens(ep); ok {
			smallest = min(smallest, tokens)
			known = true
		}
	}
	return smallest, known
}

// poolEndpoint adapts a scheduling endpoint to the datalayer endpoint the
// detector's Saturation method reads. Updates are not supported and not needed.
type poolEndpoint struct {
	fwksched.Endpoint
}

func (e poolEndpoint) GetAttributes() fwkdl.AttributeMap      { return e.Endpoint }
func (e poolEndpoint) UpdateMetadata(*fwkdl.EndpointMetadata) {}
func (e poolEndpoint) UpdateMetrics(*fwkdl.Metrics)           {}

func asPool(endpoints []fwksched.Endpoint) []fwkdl.Endpoint {
	pool := make([]fwkdl.Endpoint, 0, len(endpoints))
	for _, ep := range endpoints {
		if ep != nil {
			pool = append(pool, poolEndpoint{ep})
		}
	}
	return pool
}
