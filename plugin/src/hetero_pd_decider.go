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

package disagg

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/metrics"
	tokenproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/disagg/hetero"
)

const (
	// HeteroPDDeciderPluginType is the type-name of the heteroPDDecider plugin.
	HeteroPDDeciderPluginType = "hetero-pd-decider"

	defaultAcceleratorLabel   = "llm-d.ai/accelerator-class"
	defaultCompatibilityLabel = "llm-d.ai/kv-compat-key"
	defaultFingerprintLabel   = "llm-d.ai/model-fingerprint"
	defaultDisableLabel       = "llm-d.ai/hetero-pd-disabled"
	defaultAcceptLabelPrefix  = "accept.kv-compat.llm-d.ai/"
	defaultSpeedLabel         = "llm-d.ai/prefill-tokens-per-sec"

	// inflightMaxAge bounds how long an unfinished request's pair is remembered.
	inflightMaxAge = 15 * time.Minute
)

// HeteroPDDeciderConfig holds the configuration for the heteroPDDecider plugin.
// The decision parameters are documented on hetero.Config.
type HeteroPDDeciderConfig struct {
	hetero.Config

	// AcceleratorLabel is the pod label holding the accelerator class used to look up profiles.
	AcceleratorLabel string `json:"acceleratorLabel"`
	// CompatibilityLabel is the pod label holding the KV compatibility key: one value for
	// everything that decides whether two pods can exchange KV (model path, vLLM build,
	// attention backend, KV layout, block size, KV dtype, parallelism). See
	// hetero.Config.Compatibility for how keys are compared.
	CompatibilityLabel string `json:"compatibilityLabel"`
	// FingerprintLabel is read as the compatibility key when a pod has no CompatibilityLabel.
	// Kept for configurations written before CompatibilityLabel existed.
	FingerprintLabel string `json:"fingerprintLabel"`
	// PrefillProfile names the scheduling profile that picks the prefill pod. The plugin
	// reads its pick after scheduling to learn which pair served a request, so failed
	// handoffs can exclude that pair.
	PrefillProfile string `json:"prefillProfile"`
	// AcceptLabelPrefix marks live approvals: a decode pod labelled <prefix><prefillKey>=true
	// accepts KV from prefill pods with that compatibility key. An approval controller sets and
	// removes these labels, so approvals apply without restarting the router. Empty disables.
	AcceptLabelPrefix string `json:"acceptLabelPrefix"`
	// SpeedLabel is the pod label holding the pod's own measured prefill speed in tokens per
	// second. A controller re-measures it whenever the pod's compatibility key changes and the
	// plugin reads it live; it takes precedence over the class profile. Empty disables.
	SpeedLabel string `json:"speedLabel"`
	// DisableLabel excludes a pod from remote prefill, as source or destination, when set to "true".
	DisableLabel string `json:"disableLabel"`
	// FallbackNonCachedTokens is the threshold applied when the decode pod's class has no profile.
	// Zero keeps such requests local.
	FallbackNonCachedTokens int `json:"fallbackNonCachedTokens"`
	// PrefixMatchInfoProducerName selects which prefix-cache producer's
	// PrefixCacheMatchInfo to read on the decode endpoint. Empty defaults to the
	// approximate-prefix producer.
	PrefixMatchInfoProducerName string `json:"prefixMatchInfoProducerName,omitempty"`
}

func defaultHeteroPDDeciderConfig() HeteroPDDeciderConfig {
	return HeteroPDDeciderConfig{
		Config:                  hetero.DefaultConfig(),
		AcceleratorLabel:        defaultAcceleratorLabel,
		CompatibilityLabel:      defaultCompatibilityLabel,
		FingerprintLabel:        defaultFingerprintLabel,
		PrefillProfile:          defaultPrefillProfile,
		AcceptLabelPrefix:       defaultAcceptLabelPrefix,
		SpeedLabel:              defaultSpeedLabel,
		DisableLabel:            defaultDisableLabel,
		FallbackNonCachedTokens: 1024,
	}
}

func (c HeteroPDDeciderConfig) validate() error {
	if c.AcceleratorLabel == "" {
		return errors.New("acceleratorLabel cannot be empty")
	}
	if c.FallbackNonCachedTokens < 0 {
		return errors.New("fallbackNonCachedTokens cannot be negative")
	}
	return c.Config.Validate()
}

// compile-time type assertions
var (
	_ deciderPlugin                 = &HeteroPDDecider{}
	_ fwkrc.DataProducer            = &HeteroPDDecider{}
	_ fwkrc.PreRequest              = &HeteroPDDecider{}
	_ fwkrc.ResponseHeaderProcessor = &HeteroPDDecider{}
	_ fwkrc.ResponseBodyProcessor   = &HeteroPDDecider{}
	_ plugin.ConsumerPlugin         = &HeteroPDDecider{}
	_ prefixMatchInfoConsumer       = &HeteroPDDecider{}
)

var prefillCapableRoles = map[string]struct{}{
	bylabel.RolePrefill:             {},
	bylabel.RoleEncodePrefill:       {},
	bylabel.RolePrefillDecode:       {},
	bylabel.RoleEncodePrefillDecode: {},
}

// HeteroPDDecider decides per request whether to prefill on another pod,
// comparing estimated time to first token plus the stall local prefill
// imposes on running decode streams. Accelerator classes are read from a pod
// label and mapped to measured prefill throughput.
//
// It is also a DataProducer: Produce runs before scheduling with every
// candidate endpoint and caches the prefill-capable pods, because the decider
// itself only receives the selected decode endpoint.
type HeteroPDDecider struct {
	typedName         plugin.TypedName
	config            HeteroPDDeciderConfig
	prefixMatchInfoDK plugin.DataKey
	prefillPods       atomic.Pointer[[]hetero.Pod]
	warnNoPrefillView sync.Once
	// exclusions holds pairs excluded after failed handoffs; inflight maps a request ID
	// to the pair serving it until the response ends.
	exclusions *hetero.Exclusions
	inflight   sync.Map
	inflightN  atomic.Uint64
	now        func() time.Time
}

type inflightPair struct {
	pair  string
	start time.Time
}

// HeteroPDDeciderPluginFactory defines the factory function for creating
// a new instance of the heteroPDDecider.
func HeteroPDDeciderPluginFactory(name string, rawParameters *json.Decoder,
	_ plugin.Handle) (plugin.Plugin, error) {
	config := defaultHeteroPDDeciderConfig()
	if rawParameters != nil {
		if err := rawParameters.Decode(&config); err != nil {
			return nil, fmt.Errorf("failed to parse %s plugin config: %w", HeteroPDDeciderPluginType, err)
		}
	}
	decider, err := NewHeteroPDDecider(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create %s plugin: %w", HeteroPDDeciderPluginType, err)
	}
	return decider.WithName(name), nil
}

// NewHeteroPDDecider validates the configuration and returns the plugin.
func NewHeteroPDDecider(config HeteroPDDeciderConfig) (*HeteroPDDecider, error) {
	if err := config.validate(); err != nil {
		return nil, err
	}
	return &HeteroPDDecider{
		typedName:  plugin.TypedName{Type: HeteroPDDeciderPluginType},
		config:     config,
		exclusions: hetero.NewExclusions(config.FailureThreshold, time.Duration(config.ExclusionSeconds)*time.Second),
		now:        time.Now,
		prefixMatchInfoDK: attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(
			config.PrefixMatchInfoProducerName,
		),
	}, nil
}

// TypedName returns the typed name of the plugin.
func (d *HeteroPDDecider) TypedName() plugin.TypedName {
	return d.typedName
}

// WithName sets the name of the plugin.
func (d *HeteroPDDecider) WithName(name string) *HeteroPDDecider {
	d.typedName.Name = name
	return d
}

func (d *HeteroPDDecider) prefixMatchInfoDataKey() plugin.DataKey {
	return d.prefixMatchInfoDK
}

// Produces declares no request attributes; the prefill view is plugin state.
func (d *HeteroPDDecider) Produces() map[plugin.DataKey]any {
	return map[plugin.DataKey]any{}
}

// Consumes declares the tokenized prompt, the decode endpoint's prefix match
// info, and the endpoint metrics the estimate reads.
func (d *HeteroPDDecider) Consumes() plugin.DataDependencies {
	return plugin.DataDependencies{
		Required: map[plugin.DataKey]any{
			d.prefixMatchInfoDK:                  attrprefix.PrefixCacheMatchInfo{},
			tokenproducer.TokenizedPromptDataKey: scheduling.TokenizedRequest{},
			plugin.NewDataKey(metrics.KVCacheUsagePercentKey, metrics.MetricsExtractorType): float64(0),
			plugin.NewDataKey(metrics.WaitingQueueSizeKey, metrics.MetricsExtractorType):    int(0),
			plugin.NewDataKey(metrics.RunningRequestsSizeKey, metrics.MetricsExtractorType): int(0),
		},
	}
}

// Produce caches the prefill-capable endpoints and their load for the decision.
func (d *HeteroPDDecider) Produce(_ context.Context, _ *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) error {
	pods := make([]hetero.Pod, 0, len(endpoints))
	for _, ep := range endpoints {
		meta := ep.GetMetadata()
		if meta == nil {
			continue
		}
		if _, ok := prefillCapableRoles[meta.Labels[bylabel.RoleLabel]]; !ok {
			continue
		}
		pods = append(pods, d.toPod(ep))
	}
	d.prefillPods.Store(&pods)
	return nil
}

func (d *HeteroPDDecider) toPod(ep scheduling.Endpoint) hetero.Pod {
	pod := hetero.Pod{}
	if meta := ep.GetMetadata(); meta != nil {
		pod.Name = meta.ID.String()
		pod.Class = meta.Labels[d.config.AcceleratorLabel]
		if d.config.CompatibilityLabel != "" {
			pod.CompatKey = meta.Labels[d.config.CompatibilityLabel]
		}
		if pod.CompatKey == "" && d.config.FingerprintLabel != "" {
			pod.CompatKey = meta.Labels[d.config.FingerprintLabel]
		}
		if p := d.config.AcceptLabelPrefix; p != "" {
			for k, v := range meta.Labels {
				if v == "true" && len(k) > len(p) && k[:len(p)] == p {
					pod.Accepts = append(pod.Accepts, k[len(p):])
				}
			}
		}
		if d.config.SpeedLabel != "" {
			if v, err := strconv.ParseFloat(meta.Labels[d.config.SpeedLabel], 64); err == nil && v > 0 && !math.IsInf(v, 0) {
				pod.PrefillTokensPerSec = v
			}
		}
		if d.config.DisableLabel != "" {
			pod.Disabled, _ = strconv.ParseBool(meta.Labels[d.config.DisableLabel])
		}
	}
	if m := ep.GetMetrics(); m != nil {
		pod.Running = m.RunningRequestsSize
		pod.Waiting = m.WaitingQueueSize
		pod.KVUsage = m.KVCacheUsagePercent
	}
	return pod
}

func (d *HeteroPDDecider) uncachedTokens(request *scheduling.InferenceRequest, endpoint scheduling.Endpoint) (int, error) {
	inputTokens, err := getUserInputLenInTokens(request)
	if err != nil {
		return 0, fmt.Errorf("failed to get user input length in tokens: %w", err)
	}
	raw, ok := endpoint.Get(d.prefixMatchInfoDK)
	if !ok || raw == nil {
		return inputTokens, nil
	}
	info, ok := raw.(*attrprefix.PrefixCacheMatchInfo)
	if !ok {
		return 0, fmt.Errorf("prefix cache match info from %s has unexpected type: %T", d.prefixMatchInfoDK, raw)
	}
	return max(inputTokens-info.CachedBlockCount()*info.BlockSizeTokens(), 0), nil
}

// disaggregate reports whether remote prefill should run for this request.
// Any read failure is logged and returns false so scheduling falls back to
// the decode-only path.
func (d *HeteroPDDecider) disaggregate(ctx context.Context, request *scheduling.InferenceRequest, endpoint scheduling.Endpoint) bool {
	logger := log.FromContext(ctx)
	if endpoint == nil {
		logger.Error(errors.New("endpoint is nil"), "hetero-pd-decider: skipping remote prefill")
		return false
	}
	uncached, err := d.uncachedTokens(request, endpoint)
	if err != nil {
		logger.Error(err, "hetero-pd-decider: skipping remote prefill")
		return false
	}

	decodePod := d.toPod(endpoint)
	now := d.now()
	var prefill []hetero.Pod
	if snapshot := d.prefillPods.Load(); snapshot != nil {
		// Copy: the snapshot is shared, and exclusion is relative to this decode pod.
		prefill = make([]hetero.Pod, len(*snapshot))
		for i, p := range *snapshot {
			p.Excluded = d.exclusions.Excluded(hetero.PairKey(p, decodePod), now)
			prefill[i] = p
		}
	} else {
		d.warnNoPrefillView.Do(func() {
			logger.Info("hetero-pd-decider has no prefill pod view; Produce has not run. " +
				"Confirm the plugin is listed under plugins so the request-control layer calls it.")
		})
	}

	decision := hetero.Decide(d.config.Config, uncached, decodePod, prefill)
	if decision.Fallback {
		// Decide names a compatible prefill pod on fallback; without one, never split.
		decision.Remote = decision.Prefill != "" && d.config.FallbackNonCachedTokens > 0 &&
			uncached >= d.config.FallbackNonCachedTokens
	}
	recordHeteroPDDecision(d.typedName.Name, decision.Reason, decision.Remote)

	if v := logger.V(logging.DEBUG); v.Enabled() {
		v.Info("hetero-pd-decider decision",
			"remote", decision.Remote, "reason", decision.Reason, "strategy", decision.Strategy,
			"fallback", decision.Fallback,
			"uncachedTokens", uncached, "prefillPod", decision.Prefill,
			"remoteSeconds", decision.RemoteSeconds, "localSeconds", decision.LocalSeconds,
			"stallSeconds", decision.StallSeconds, "transferSeconds", decision.TransferSeconds,
			"handoffSeconds", decision.HandoffSeconds)
	}
	return decision.Remote
}

// CompatiblePrefill returns the candidates that may prefill for decode: the same rule the
// decider applies when pricing (compatibility keys, approved pairs, exclusions, disable
// label). The kv-compat filter calls it so the prefill profile can only pick a pod the
// decider would accept. It fails closed: an empty result means no pod may prefill.
func (d *HeteroPDDecider) CompatiblePrefill(decode scheduling.Endpoint, candidates []scheduling.Endpoint) []scheduling.Endpoint {
	decodePod := d.toPod(decode)
	now := d.now()
	pods := make([]hetero.Pod, len(candidates))
	for i, ep := range candidates {
		pods[i] = d.toPod(ep)
		pods[i].Excluded = d.exclusions.Excluded(hetero.PairKey(pods[i], decodePod), now)
	}
	mixed := hetero.Mixed(decodePod, pods)
	out := make([]scheduling.Endpoint, 0, len(candidates))
	for i, ep := range candidates {
		if d.config.Eligible(pods[i], decodePod, mixed) {
			out = append(out, ep)
		}
	}
	return out
}

// PreRequest remembers which prefill/decode pair serves a disaggregated request, so the
// response can count toward excluding that pair. It never fails the request.
func (d *HeteroPDDecider) PreRequest(_ context.Context, request *scheduling.InferenceRequest,
	result *scheduling.SchedulingResult) error {
	if request == nil || result == nil || d.config.FailureThreshold <= 0 {
		return nil
	}
	decode := result.PrimaryEndpoint()
	prefill := result.ProfileResults[d.config.PrefillProfile].FirstEndpoint()
	if decode == nil || prefill == nil {
		return nil
	}
	now := d.now()
	d.inflight.Store(request.RequestID, inflightPair{pair: hetero.PairKey(d.toPod(prefill), d.toPod(decode)), start: now})
	if d.inflightN.Add(1)%1024 == 0 {
		d.inflight.Range(func(k, v any) bool {
			if now.Sub(v.(inflightPair).start) > inflightMaxAge {
				d.inflight.Delete(k)
			}
			return true
		})
	}
	return nil
}

// ResponseHeader counts a non-200 status on a disaggregated request as a failed handoff.
// The status alone cannot tell a rejected KV handshake from other failures on that path;
// the consecutive-failure threshold keeps one-off errors from excluding a pair.
func (d *HeteroPDDecider) ResponseHeader(ctx context.Context, request *scheduling.InferenceRequest,
	response *fwkrc.Response, _ *datalayer.EndpointMetadata) {
	if response == nil {
		return
	}
	status := response.Headers[":status"]
	if status == "" {
		status = response.Headers["status"]
	}
	if status != "" && status != "200" {
		d.finish(ctx, requestID(request, response), false)
	}
}

// ResponseBody records the outcome when the response ends: a natural end is a success,
// an error end a failure. Client disconnects and evictions say nothing about the pair.
func (d *HeteroPDDecider) ResponseBody(ctx context.Context, request *scheduling.InferenceRequest,
	response *fwkrc.Response, _ *datalayer.EndpointMetadata) {
	if response == nil || !response.EndOfStream {
		return
	}
	id := requestID(request, response)
	switch response.TerminationCause {
	case fwkrc.TerminationCauseNatural, "":
		d.finish(ctx, id, true)
	case fwkrc.TerminationCauseError:
		d.finish(ctx, id, false)
	default:
		d.inflight.Delete(id)
	}
}

func requestID(request *scheduling.InferenceRequest, response *fwkrc.Response) string {
	if request != nil && request.RequestID != "" {
		return request.RequestID
	}
	return response.RequestID
}

func (d *HeteroPDDecider) finish(ctx context.Context, id string, ok bool) {
	v, found := d.inflight.LoadAndDelete(id)
	if !found {
		return
	}
	pair := v.(inflightPair).pair
	if d.exclusions.Record(pair, ok, d.now()) {
		recordHeteroPDExclusion(d.typedName.Name)
		log.FromContext(ctx).Info("hetero-pd-decider: excluding prefill/decode pair after failed handoffs",
			"pair", pair, "failures", d.config.FailureThreshold, "seconds", d.config.ExclusionSeconds)
	}
}
