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

// Package prefixcacheshed provides a scheduling filter that sheds cache-cold
// requests while the pool is near saturation. Below the shed threshold it is a
// pass-through. At or above it, only endpoints that have room and a prefix cache
// hit ratio of at least minHitRatio stay candidates; when none remains the
// request is rejected with a typed error (503 or 429) and a shed reason header.
//
// The filter must be the first filter of its scheduling profile. It computes the
// pool saturation from the endpoint list it receives, and only the first filter
// receives the full pool.
package prefixcacheshed

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sync/atomic"

	"sigs.k8s.io/controller-runtime/pkg/log"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

const (
	PluginType = "prefix-cache-shed-filter"
)

var (
	_ fwksched.Filter          = &Plugin{}
	_ fwkplugin.ConsumerPlugin = &Plugin{}
	_ fwkplugin.ProducerPlugin = &Plugin{}
)

// capacityChecker is the per-endpoint half of the detector contract. The
// concurrency detector implements it.
type capacityChecker interface {
	HasCapacity(endpoint fwksched.Endpoint) bool
}

// detector is what the filter needs from the referenced saturation detector.
type detector interface {
	flowcontrol.SaturationDetector
	capacityChecker
}

// Plugin is the prefix cache shed filter.
type Plugin struct {
	typedName                    fwkplugin.TypedName
	config                       Config
	errorCode                    string
	detector                     detector
	prefixMatchDataKey           fwkplugin.DataKey
	inFlightLoadDataKey          fwkplugin.DataKey
	uncachedRequestTokensDataKey fwkplugin.DataKey
	shedding                     atomic.Bool
}

// Factory builds the filter. The referenced detector must already be
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
	det, ok := p.(detector)
	if !ok {
		return nil, fmt.Errorf("plugin %s must implement flowcontrol.SaturationDetector and HasCapacity", config.SaturationDetectorRef)
	}
	if err := registerMetrics(handle.Metrics()); err != nil {
		return nil, err
	}
	return New(name, *config, det), nil
}

// New builds the filter around an already resolved detector. Config must be valid.
func New(name string, config Config, det detector) *Plugin {
	code, _ := config.errorCode()
	return &Plugin{
		typedName:                    fwkplugin.TypedName{Type: PluginType, Name: name},
		config:                       config,
		errorCode:                    code,
		detector:                     det,
		prefixMatchDataKey:           attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(config.PrefixMatchInfoProducerName),
		inFlightLoadDataKey:          attrconcurrency.InFlightLoadDataKey.WithNonEmptyProducerName(config.InFlightLoadProducerName),
		uncachedRequestTokensDataKey: attrconcurrency.UncachedRequestTokensDataKey.WithNonEmptyProducerName(config.InFlightLoadProducerName),
	}
}

func (p *Plugin) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// Shedding reports whether the filter is currently in the shedding state.
func (p *Plugin) Shedding() bool {
	return p.shedding.Load()
}

// Consumes declares the endpoint attributes the filter and its detector read.
func (p *Plugin) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{
			p.prefixMatchDataKey:  attrprefix.PrefixCacheMatchInfo{},
			p.inFlightLoadDataKey: attrconcurrency.InFlightLoad{},
		},
		Optional: map[fwkplugin.DataKey]any{
			p.uncachedRequestTokensDataKey: attrconcurrency.UncachedRequestTokens{},
		},
	}
}

// Produces declares the request attribute the filter writes on a shed.
func (p *Plugin) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{fwksched.RequestDropDataKey: &fwksched.RequestDrop{}}
}

// Filter applies the shed rule. See the package documentation.
func (p *Plugin) Filter(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	logger := log.FromContext(ctx).V(logutil.DEBUG)
	if len(endpoints) == 0 {
		recordDecision(p.typedName.Name, outcomeNotApplicable)
		return endpoints
	}

	saturation := p.detector.Saturation(ctx, asPool(endpoints))
	wasActive := p.shedding.Load()
	active := p.updateState(saturation)
	recordState(p.typedName.Name, active, saturation)
	if active != wasActive {
		log.FromContext(ctx).V(logutil.DEFAULT).Info("PrefixCacheShedFilter: shedding state changed",
			"shedding", active, "saturation", saturation, "threshold", p.config.ShedThreshold,
			"hysteresis", p.config.Hysteresis)
	}
	if !active {
		recordDecision(p.typedName.Name, outcomePassBelowThreshold)
		return endpoints
	}

	hot := 0
	bestHit := 0.0
	admitted := make([]fwksched.Endpoint, 0, len(endpoints))
	minUncached := int64(math.MaxInt64)
	for _, ep := range endpoints {
		if ep == nil {
			continue
		}
		ratio := p.hitRatio(ep)
		bestHit = max(bestHit, ratio)
		if ratio < p.config.MinHitRatio {
			continue
		}
		hot++
		if !p.detector.HasCapacity(ep) {
			continue
		}
		admitted = append(admitted, ep)
		if tokens := p.uncachedTokens(ep); tokens < minUncached {
			minUncached = tokens
		}
	}

	if len(admitted) > 0 {
		logger.Info("PrefixCacheShedFilter: shedding, narrowed to hot endpoints with room",
			"saturation", saturation, "hot", hot, "admitted", len(admitted), "total", len(endpoints))
		recordDecision(p.typedName.Name, outcomeAdmittedHot)
		return admitted
	}

	if p.config.MinUncachedTokens > 0 && p.smallestUncached(endpoints) < p.config.MinUncachedTokens {
		logger.Info("PrefixCacheShedFilter: shedding, small request exempt",
			"saturation", saturation, "minUncachedTokens", p.config.MinUncachedTokens)
		recordDecision(p.typedName.Name, outcomeBypassSmallRequest)
		return endpoints
	}

	reason := errcommon.RequestDroppedReasonShedCold
	outcome := outcomeShedCold
	msg := "request shed: pool near saturation and the prompt has no prefix cache hit on an endpoint with room"
	if hot > 0 {
		reason = errcommon.RequestDroppedReasonShedDisplacedHot
		outcome = outcomeShedDisplacedHot
		msg = "request shed: pool near saturation and the endpoints holding the prompt's prefix cache are full"
	}
	// The numbers behind the decision travel in the response body so a client (or a load
	// test) can tell a genuinely cold request from an index miss without EPP debug logs.
	msg = fmt.Sprintf("%s (saturation=%.3f bestHitRatio=%.3f hotEndpoints=%d endpoints=%d)",
		msg, saturation, bestHit, hot, len(endpoints))
	if request != nil {
		request.PutAttribute(fwksched.RequestDropDataKey, &fwksched.RequestDrop{
			Code:   p.errorCode,
			Reason: string(reason),
			Msg:    msg,
		})
	}
	logger.Info("PrefixCacheShedFilter: shedding request",
		"saturation", saturation, "reason", reason, "hot", hot, "total", len(endpoints))
	recordDecision(p.typedName.Name, outcome)
	return nil
}

// updateState moves the hysteresis state machine and returns whether shedding is active.
// Enter at S >= T, leave at S < T - hysteresis. The read-modify-write is not atomic; two
// concurrent requests at the boundary can each decide from the previous state, which is
// harmless because both decisions are valid for a saturation inside the band.
func (p *Plugin) updateState(saturation float64) bool {
	active := p.shedding.Load()
	if active {
		if saturation < p.config.ShedThreshold-p.config.Hysteresis {
			active = false
		}
	} else if saturation >= p.config.ShedThreshold {
		active = true
	}
	p.shedding.Store(active)
	return active
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
	ratio := float64(info.MatchBlocks()) / float64(info.TotalBlocks())
	if math.IsNaN(ratio) {
		return 0
	}
	return ratio
}

func (p *Plugin) uncachedTokens(ep fwksched.Endpoint) int64 {
	if raw, ok := ep.Get(p.uncachedRequestTokensDataKey); ok {
		if tokens, ok := raw.(*attrconcurrency.UncachedRequestTokens); ok && tokens != nil {
			return tokens.Tokens
		}
	}
	return 0
}

// smallestUncached is the uncached token count of the request on the endpoint
// with room where it is cheapest. Without an endpoint with room it falls back to
// the cheapest endpoint overall.
func (p *Plugin) smallestUncached(endpoints []fwksched.Endpoint) int64 {
	withRoom := int64(math.MaxInt64)
	overall := int64(math.MaxInt64)
	for _, ep := range endpoints {
		if ep == nil {
			continue
		}
		tokens := p.uncachedTokens(ep)
		overall = min(overall, tokens)
		if p.detector.HasCapacity(ep) {
			withRoom = min(withRoom, tokens)
		}
	}
	if withRoom != math.MaxInt64 {
		return withRoom
	}
	return overall
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
