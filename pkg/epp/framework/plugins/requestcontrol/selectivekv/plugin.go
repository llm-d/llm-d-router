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

// Package selectivekv applies per-request KV load and offload policy.
package selectivekv

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sync"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	commonrequest "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	extractormetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/metrics"
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
	preciseproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/preciseprefixcache"
	parserutil "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/util"
)

const (
	// PluginType is the registered type name of the selective KV policy plugin.
	PluginType = "selective-kv-policy"

	fieldMaxLoadTokens    = "max_load_tokens"
	fieldMaxOffloadTokens = "max_offload_tokens"
	gpuTierKey            = "gpu"

	waitingQueueEWMAHalfLife = 2 * time.Second
	waitingQueueReopenRatio  = 0.5
)

// Policy controls whether the router preserves backend defaults or disables
// one KV transfer direction.
type Policy string

const (
	PolicyPreserve  Policy = "preserve"
	PolicyDisable   Policy = "disable"
	PolicyThreshold Policy = "threshold"
)

// Config configures per-request KV transfer policy.
type Config struct {
	LoadPolicy                  Policy `json:"loadPolicy,omitempty"`
	OffloadPolicy               Policy `json:"offloadPolicy,omitempty"`
	MinExternalReusableTokens   int    `json:"minExternalReusableTokens,omitempty"`
	MaxWaitingRequests          int    `json:"maxWaitingRequests,omitempty"`
	PrefixMatchInfoProducerName string `json:"prefixMatchInfoProducerName,omitempty"`
}

// DefaultConfig preserves both transfer directions. Threshold mode uses the
// precise producer because it requires engine-reported per-tier cache data.
var DefaultConfig = Config{
	LoadPolicy:                  PolicyPreserve,
	OffloadPolicy:               PolicyPreserve,
	PrefixMatchInfoProducerName: preciseproducer.PluginType,
}

type loadAction string

const (
	loadPreserve loadAction = "preserve"
	loadDisable  loadAction = "disable"
	loadEnable   loadAction = "enable"
)

var (
	_ requestcontrol.PreRequest = &Plugin{}
	_ plugin.ConsumerPlugin     = &Plugin{}
	_ fwkdl.EndpointExtractor   = &Plugin{}
	_ fwkdl.Registrant          = &Plugin{}
)

// Plugin applies the configured policy immediately before backend dispatch.
type Plugin struct {
	typedName                 plugin.TypedName
	loadPolicy                Policy
	offloadPolicy             Policy
	minExternalReusableTokens int
	maxWaitingRequests        int
	prefixMatchInfoDataKey    plugin.DataKey
	waitingQueueMu            sync.Mutex
	waitingQueueByEndpoint    map[string]waitingQueueState
}

type waitingQueueState struct {
	ewma       float64
	sampleTime time.Time
	closed     bool
}

// PluginFactory constructs a selective KV policy plugin.
func PluginFactory(name string, rawParameters *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	if rawParameters == nil {
		return nil, fmt.Errorf("%s: parameters are required", PluginType)
	}
	cfg := DefaultConfig
	if err := rawParameters.Decode(&cfg); err != nil {
		return nil, fmt.Errorf("failed to parse %s plugin config: %w", PluginType, err)
	}
	return New(name, cfg)
}

// New constructs a selective KV policy plugin from validated configuration.
func New(name string, cfg Config) (*Plugin, error) {
	if cfg.LoadPolicy != PolicyPreserve && cfg.LoadPolicy != PolicyDisable &&
		cfg.LoadPolicy != PolicyThreshold {
		return nil, fmt.Errorf("%s: loadPolicy must be %q, %q, or %q, got %q",
			PluginType, PolicyPreserve, PolicyDisable, PolicyThreshold,
			cfg.LoadPolicy)
	}
	if cfg.OffloadPolicy != PolicyPreserve && cfg.OffloadPolicy != PolicyDisable {
		return nil, fmt.Errorf("%s: offloadPolicy must be %q or %q, got %q",
			PluginType, PolicyPreserve, PolicyDisable, cfg.OffloadPolicy)
	}
	if cfg.LoadPolicy == PolicyPreserve && cfg.OffloadPolicy == PolicyPreserve {
		return nil, fmt.Errorf(
			"%s: at least one of loadPolicy or offloadPolicy must be active",
			PluginType)
	}
	if cfg.LoadPolicy == PolicyThreshold && cfg.MinExternalReusableTokens <= 0 {
		return nil, fmt.Errorf("%s: minExternalReusableTokens must be greater than zero for threshold policy", PluginType)
	}
	if cfg.MaxWaitingRequests < 0 {
		return nil, fmt.Errorf("%s: maxWaitingRequests must not be negative", PluginType)
	}
	if cfg.MaxWaitingRequests > 0 && cfg.LoadPolicy != PolicyThreshold {
		return nil, fmt.Errorf(
			"%s: maxWaitingRequests requires threshold loadPolicy", PluginType)
	}

	return &Plugin{
		typedName:                 plugin.TypedName{Type: PluginType, Name: name},
		loadPolicy:                cfg.LoadPolicy,
		offloadPolicy:             cfg.OffloadPolicy,
		minExternalReusableTokens: cfg.MinExternalReusableTokens,
		maxWaitingRequests:        cfg.MaxWaitingRequests,
		prefixMatchInfoDataKey: attrprefix.PrefixCacheMatchInfoDataKey.
			WithNonEmptyProducerName(cfg.PrefixMatchInfoProducerName),
		waitingQueueByEndpoint: make(map[string]waitingQueueState),
	}, nil
}

// TypedName returns the plugin's registered type and configured name.
func (p *Plugin) TypedName() plugin.TypedName {
	return p.typedName
}

// Consumes declares the endpoint data used by threshold loading.
func (p *Plugin) Consumes() plugin.DataDependencies {
	if p.loadPolicy != PolicyThreshold {
		return plugin.DataDependencies{}
	}
	required := map[plugin.DataKey]any{
		p.prefixMatchInfoDataKey: attrprefix.PrefixCacheMatchInfo{},
	}
	if p.maxWaitingRequests > 0 {
		required[plugin.NewDataKey(extractormetrics.WaitingQueueSizeKey,
			extractormetrics.MetricsExtractorType)] = int(0)
	}
	return plugin.DataDependencies{Required: required}
}

// RegisterDependencies subscribes queue-aware policies to endpoint lifecycle events.
func (p *Plugin) RegisterDependencies(registrar fwkdl.Registrar) error {
	if p.maxWaitingRequests == 0 {
		return nil
	}
	return registrar.Register(fwkdl.PendingRegistration{
		Owner:      p.TypedName(),
		SourceType: sourcenotifications.EndpointNotificationSourceType,
		Extractor:  p,
		DefaultSource: sourcenotifications.NewEndpointDataSource(
			sourcenotifications.EndpointNotificationSourceType,
			sourcenotifications.EndpointNotificationSourceType,
		),
	})
}

// Extract removes queue state when an endpoint leaves the pool.
func (p *Plugin) Extract(_ context.Context, event fwkdl.EndpointEvent) error {
	if event.Type != fwkdl.EventDelete || event.Endpoint == nil {
		return nil
	}
	metadata := event.Endpoint.GetMetadata()
	if metadata == nil || metadata.ID.Name == "" {
		return nil
	}

	p.waitingQueueMu.Lock()
	delete(p.waitingQueueByEndpoint, metadata.ID.String())
	p.waitingQueueMu.Unlock()
	return nil
}

// PreRequest overwrites the configured KV transfer controls in parsed JSON
// requests. Other kv_transfer_params fields are preserved.
func (p *Plugin) PreRequest(ctx context.Context, request *scheduling.InferenceRequest,
	schedulingResult *scheduling.SchedulingResult) error {
	load := p.decideLoad(ctx, schedulingResult)
	if load == loadPreserve && p.offloadPolicy == PolicyPreserve {
		return nil
	}
	logger := log.FromContext(ctx).WithName(p.typedName.String()).V(logging.DEBUG)
	if request == nil || request.Body == nil || request.Body.Payload == nil {
		logger.Info("skipping selective KV policy: request has no body payload")
		return nil
	}
	if request.Body.Generate != nil {
		logger.Info("skipping selective KV policy: native generate request",
			"requestID", request.RequestID)
		return nil
	}
	payload, ok := request.Body.Payload.(requesthandling.PayloadMap)
	if !ok {
		logger.Info("skipping selective KV policy: request payload is not mutable JSON",
			"requestID", request.RequestID)
		return nil
	}

	if mutateKVTransferParams(payload, load, p.offloadPolicy) {
		request.Body.Mutated = true
	}
	return nil
}

func mutateKVTransferParams(payload requesthandling.PayloadMap, load loadAction,
	offload Policy) bool {
	value := payload[commonrequest.FieldKVTransferParams]
	params, decoded := value.(map[string]any)
	if !decoded {
		raw, ok := value.(json.RawMessage)
		if ok {
			if err := parserutil.Unmarshal(raw, &params); err == nil && params != nil {
				decoded = true
			}
		}
	}
	if !decoded {
		if load != loadDisable && offload != PolicyDisable {
			return false
		}
		params = map[string]any{}
	}

	changed := false
	switch load {
	case loadDisable:
		changed = setZero(params, fieldMaxLoadTokens)
	case loadEnable:
		if _, exists := params[fieldMaxLoadTokens]; exists {
			delete(params, fieldMaxLoadTokens)
			changed = true
		}
	}
	if offload == PolicyDisable {
		changed = setZero(params, fieldMaxOffloadTokens) || changed
	}
	if changed {
		payload[commonrequest.FieldKVTransferParams] = params
	}
	return changed
}

func setZero(params map[string]any, field string) bool {
	switch value := params[field].(type) {
	case json.Number:
		if number, err := value.Float64(); err == nil && number == 0 {
			return false
		}
	case float64:
		if value == 0 {
			return false
		}
	}
	params[field] = json.Number("0")
	return true
}

func (p *Plugin) decideLoad(ctx context.Context,
	result *scheduling.SchedulingResult) loadAction {
	switch p.loadPolicy {
	case PolicyDisable:
		return loadDisable
	case PolicyThreshold:
		logger := log.FromContext(ctx).WithName(p.typedName.String()).V(logging.DEBUG)
		endpoint := selectedEndpoint(result)
		if endpoint == nil {
			logger.Info("no target endpoint; preserving KV load")
			return loadPreserve
		}
		tokens, ok := p.externalReusableTokens(endpoint)
		if !ok {
			logger.Info("no tier evidence; preserving KV load")
			return loadPreserve
		}
		action := loadEnable
		if tokens < p.minExternalReusableTokens {
			action = loadDisable
		} else if p.waitingQueueVeto(endpoint) {
			action = loadDisable
		}
		logger.Info("evaluated selective KV load policy",
			"externalReusableTokens", tokens,
			"minExternalReusableTokens", p.minExternalReusableTokens,
			"loadAction", action)
		return action
	default:
		return loadPreserve
	}
}

func (p *Plugin) waitingQueueVeto(endpoint scheduling.Endpoint) bool {
	if p.maxWaitingRequests == 0 || endpoint == nil {
		return false
	}
	metadata := endpoint.GetMetadata()
	metrics := endpoint.GetMetrics()
	if metadata == nil || metadata.ID.Name == "" || metrics == nil ||
		metrics.UpdateTime.IsZero() {
		return false
	}

	endpointID := metadata.ID.String()
	sampleTime := metrics.UpdateTime
	sample := float64(metrics.WaitingQueueSize)

	p.waitingQueueMu.Lock()
	defer p.waitingQueueMu.Unlock()

	state, found := p.waitingQueueByEndpoint[endpointID]
	if !found || sampleTime.Before(state.sampleTime) {
		state = waitingQueueState{ewma: sample, sampleTime: sampleTime}
	} else if sampleTime.After(state.sampleTime) {
		elapsed := sampleTime.Sub(state.sampleTime)
		weight := 1 - math.Exp(-math.Ln2*
			elapsed.Seconds()/waitingQueueEWMAHalfLife.Seconds())
		state.ewma += weight * (sample - state.ewma)
		state.sampleTime = sampleTime
	}

	closeAt := float64(p.maxWaitingRequests)
	if state.closed {
		if state.ewma <= closeAt*waitingQueueReopenRatio {
			state.closed = false
		}
	} else if state.ewma >= closeAt {
		state.closed = true
	}
	p.waitingQueueByEndpoint[endpointID] = state
	return state.closed
}

func selectedEndpoint(result *scheduling.SchedulingResult) scheduling.Endpoint {
	if result == nil {
		return nil
	}
	profile := result.ProfileResults[result.PrimaryProfileName]
	if profile == nil || len(profile.TargetEndpoints) == 0 {
		return nil
	}
	return profile.TargetEndpoints[0]
}

func (p *Plugin) externalReusableTokens(endpoint scheduling.Endpoint) (int, bool) {
	raw, ok := endpoint.Get(p.prefixMatchInfoDataKey)
	if !ok {
		return 0, false
	}
	info, ok := raw.(*attrprefix.PrefixCacheMatchInfo)
	if !ok || info == nil || info.BlockSizeTokens() <= 0 {
		return 0, false
	}
	byTier := info.CachedBlocksByTier()
	if byTier == nil {
		return 0, false
	}

	gpuBlocks := byTier[gpuTierKey]
	maxExternalBlocks := 0
	for tier, blocks := range byTier {
		if tier == gpuTierKey || tier == attrprefix.SpeculativeTierKey {
			continue
		}
		maxExternalBlocks = max(maxExternalBlocks, blocks)
	}
	return max(0, maxExternalBlocks-gpuBlocks) * info.BlockSizeTokens(), true
}
