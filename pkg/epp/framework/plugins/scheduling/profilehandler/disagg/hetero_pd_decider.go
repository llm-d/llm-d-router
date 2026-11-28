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
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/go-logr/logr"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/disagg/hetero"
)

const (
	// HeteroPDDeciderPluginType is the type of the decider that compares the estimated cost of
	// prefilling on the decode endpoint with prefilling on another endpoint and moving the KV.
	HeteroPDDeciderPluginType = "hetero-pd-decider"

	defaultAcceleratorLabel        = "llm-d.ai/accelerator-class"
	defaultSpeedLabel              = "llm-d.ai/prefill-tokens-per-sec"
	defaultHeteroDisableLabel      = "llm-d.ai/hetero-pd-disabled"
	defaultHeteroMaxCandidates     = 8
	defaultHeteroRefreshInterval   = 100 * time.Millisecond
	defaultHeteroFallbackNonCached = 1024
)

var decodeCapableRoles = map[string]struct{}{
	bylabel.RoleDecode:              {},
	bylabel.RolePrefillDecode:       {},
	bylabel.RoleEncodePrefillDecode: {},
}

// HeteroPDDeciderConfig holds the configuration of the hetero-pd-decider. The decision
// parameters are documented on hetero.Config.
type HeteroPDDeciderConfig struct {
	hetero.Config

	// AcceleratorLabel is the endpoint label holding the accelerator class used to look up profiles.
	AcceleratorLabel string `json:"acceleratorLabel"`
	// SpeedLabel is the endpoint label holding the endpoint's own measured prefill speed in tokens
	// per second. It takes precedence over the class profile. Empty disables it.
	SpeedLabel string `json:"speedLabel"`
	// DisableLabel excludes an endpoint from remote prefill, as source or destination, when set to "true".
	DisableLabel string `json:"disableLabel"`
	// FallbackNonCachedTokens is the threshold applied when the decode endpoint's class has no
	// profile. Zero keeps such requests local.
	FallbackNonCachedTokens int `json:"fallbackNonCachedTokens"`
	// MaxCandidates is how many prefill endpoints per decode endpoint the decider prices, the
	// cheapest ones by the latest refresh.
	MaxCandidates int `json:"maxCandidates"`
	// RefreshInterval is how often the candidates are re-ranked from the endpoints' metrics.
	RefreshInterval metav1.Duration `json:"refreshInterval"`
	// PrefixMatchInfoProducerName selects which prefix-cache producer's PrefixCacheMatchInfo to
	// read on the decode endpoint. Empty defaults to the approximate-prefix producer.
	PrefixMatchInfoProducerName string `json:"prefixMatchInfoProducerName,omitempty"`
}

func defaultHeteroPDDeciderConfig() HeteroPDDeciderConfig {
	return HeteroPDDeciderConfig{
		Config:                  hetero.DefaultConfig(),
		AcceleratorLabel:        defaultAcceleratorLabel,
		SpeedLabel:              defaultSpeedLabel,
		DisableLabel:            defaultHeteroDisableLabel,
		FallbackNonCachedTokens: defaultHeteroFallbackNonCached,
		MaxCandidates:           defaultHeteroMaxCandidates,
		RefreshInterval:         metav1.Duration{Duration: defaultHeteroRefreshInterval},
	}
}

func (c HeteroPDDeciderConfig) validate() error {
	switch {
	case c.AcceleratorLabel == "":
		return errors.New("acceleratorLabel cannot be empty")
	case c.FallbackNonCachedTokens < 0:
		return errors.New("fallbackNonCachedTokens cannot be negative")
	case c.MaxCandidates <= 0:
		return errors.New("maxCandidates must be positive")
	case c.RefreshInterval.Duration <= 0:
		return errors.New("refreshInterval must be positive")
	}
	return c.Validate()
}

// kvCompatAware is implemented by deciders that limit their candidates to the endpoints a
// kv-compat-pd-decider allows. The kv-compat-pd-decider wrapping such a decider hands itself over.
type kvCompatAware interface {
	useKVCompat(*KVCompatPDDecider)
}

// HeteroPDDecider decides per request whether to prefill on another endpoint, comparing
// estimated time to first token plus the stall a local prefill imposes on the streams already
// running on the decode endpoint.
//
// The candidates are ranked in the background: every RefreshInterval the decider prices each
// prefill endpoint from its latest metrics and keeps, for each decode endpoint, the
// MaxCandidates cheapest ones it may hand KV to. A decision prices only those candidates, so its
// cost does not grow with the size of the pool.
type HeteroPDDecider struct {
	typedName         plugin.TypedName
	config            HeteroPDDeciderConfig
	prefixMatchInfoDK plugin.DataKey
	compat            atomic.Pointer[KVCompatPDDecider]

	mu        sync.Mutex
	endpoints map[fwkdl.ID]fwkdl.Endpoint
	snapshot  atomic.Pointer[heteroSnapshot]
}

// heteroSnapshot holds the ranked candidates of the latest refresh.
type heteroSnapshot struct {
	byDecode map[fwkdl.ID]heteroDecodeEntry
}

// heteroDecodeEntry holds the candidates of one decode endpoint, cheapest first.
type heteroDecodeEntry struct {
	name       string
	candidates []hetero.Pod
}

var (
	_ deciderPlugin           = &HeteroPDDecider{}
	_ prefixMatchInfoConsumer = &HeteroPDDecider{}
	_ kvCompatAware           = &HeteroPDDecider{}
	_ fwkdl.EndpointExtractor = &HeteroPDDecider{}
	_ fwkdl.Registrant        = &HeteroPDDecider{}
)

// HeteroPDDeciderPluginFactory creates a hetero-pd-decider and starts its candidate refresh.
func HeteroPDDeciderPluginFactory(name string, rawParameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	if handle == nil {
		return nil, errors.New("plugin handle is required")
	}
	config := defaultHeteroPDDeciderConfig()
	if rawParameters != nil {
		if err := rawParameters.Decode(&config); err != nil {
			return nil, fmt.Errorf("failed to parse %s plugin config: %w", HeteroPDDeciderPluginType, err)
		}
	}
	if err := registerMetrics(handle.Metrics()); err != nil {
		return nil, err
	}
	d, err := NewHeteroPDDecider(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create %s plugin: %w", HeteroPDDeciderPluginType, err)
	}
	d.typedName.Name = name
	go d.refreshLoop(handle.Context())
	return d, nil
}

// NewHeteroPDDecider validates the configuration and returns the plugin. It does not start the
// candidate refresh; call refresh, or use the factory.
func NewHeteroPDDecider(config HeteroPDDeciderConfig) (*HeteroPDDecider, error) {
	if err := config.validate(); err != nil {
		return nil, err
	}
	d := &HeteroPDDecider{
		typedName:         plugin.TypedName{Type: HeteroPDDeciderPluginType, Name: HeteroPDDeciderPluginType},
		config:            config,
		prefixMatchInfoDK: attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(config.PrefixMatchInfoProducerName),
		endpoints:         map[fwkdl.ID]fwkdl.Endpoint{},
	}
	d.snapshot.Store(&heteroSnapshot{})
	return d, nil
}

// TypedName returns the typed name of the plugin.
func (d *HeteroPDDecider) TypedName() plugin.TypedName {
	return d.typedName
}

func (d *HeteroPDDecider) prefixMatchInfoDataKey() plugin.DataKey {
	return d.prefixMatchInfoDK
}

func (d *HeteroPDDecider) useKVCompat(k *KVCompatPDDecider) {
	d.compat.Store(k)
}

// Extract keeps the set of endpoints the refresh ranks.
func (d *HeteroPDDecider) Extract(_ context.Context, event fwkdl.EndpointEvent) error {
	if event.Endpoint == nil || event.Endpoint.GetMetadata() == nil {
		return nil
	}
	id := event.Endpoint.GetMetadata().ID
	d.mu.Lock()
	defer d.mu.Unlock()
	if event.Type == fwkdl.EventDelete {
		delete(d.endpoints, id)
	} else {
		d.endpoints[id] = event.Endpoint
	}
	return nil
}

// RegisterDependencies subscribes the decider to endpoint lifecycle events.
func (d *HeteroPDDecider) RegisterDependencies(r fwkdl.Registrar) error {
	return r.Register(fwkdl.PendingRegistration{
		Owner:      d.TypedName(),
		SourceType: sourcenotifications.EndpointNotificationSourceType,
		Extractor:  d,
		DefaultSource: sourcenotifications.NewEndpointDataSource(sourcenotifications.EndpointNotificationSourceType,
			sourcenotifications.EndpointNotificationSourceType),
	})
}

func (d *HeteroPDDecider) refreshLoop(ctx context.Context) {
	ticker := time.NewTicker(d.config.RefreshInterval.Duration)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			d.refresh(ctx)
		}
	}
}

// refresh ranks the prefill endpoints by their estimated cost for a prompt of
// StockNonCachedTokens uncached tokens and stores, per decode endpoint, the cheapest
// MaxCandidates it may hand KV to.
func (d *HeteroPDDecider) refresh(ctx context.Context) {
	d.mu.Lock()
	prefill := make([]hetero.Pod, 0, len(d.endpoints))
	prefillMeta := make([]*fwkdl.EndpointMetadata, 0, len(d.endpoints))
	decode := make([]*fwkdl.EndpointMetadata, 0, len(d.endpoints))
	for _, ep := range d.endpoints {
		md := ep.GetMetadata()
		if md == nil {
			continue
		}
		role := md.Labels[bylabel.RoleLabel]
		if _, ok := prefillCapableRoles[role]; ok {
			pod := d.toPod(md, ep.GetMetrics())
			pod.Name = md.ID.String()
			prefill = append(prefill, pod)
			prefillMeta = append(prefillMeta, md)
		}
		if _, ok := decodeCapableRoles[role]; ok {
			decode = append(decode, md)
		}
	}
	d.mu.Unlock()

	order := make([]int, len(prefill))
	cost := make([]float64, len(prefill))
	u := float64(d.config.StockNonCachedTokens)
	for i, p := range prefill {
		order[i] = i
		cost[i] = math.Inf(1)
		if rate, ok := d.config.PodRate(p); ok && !p.Disabled {
			cost[i] = u/rate + d.config.QueueSeconds(p, rate)
		}
	}
	slices.SortStableFunc(order, func(a, b int) int { return cmp.Compare(cost[a], cost[b]) })

	compat := d.compat.Load()
	snap := &heteroSnapshot{byDecode: make(map[fwkdl.ID]heteroDecodeEntry, len(decode))}
	for _, dm := range decode {
		excl, nowMs := noKVCompatExclusions, int64(0)
		if compat != nil {
			excl, nowMs = compat.exclusionsFor(ctx, dm.ID), compat.now().UnixMilli()
		}
		name := dm.ID.String()
		picked := make([]hetero.Pod, 0, d.config.MaxCandidates)
		for _, i := range order {
			if len(picked) == d.config.MaxCandidates {
				break
			}
			if prefill[i].Name == name {
				continue
			}
			if compat != nil && (!compat.compatible(prefillMeta[i].Labels, dm.Labels) ||
				excl.excluded(kvCompatPairOf(compat.rule, prefillMeta[i]), nowMs)) {
				continue
			}
			picked = append(picked, prefill[i])
		}
		snap.byDecode[dm.ID] = heteroDecodeEntry{name: name, candidates: picked}
	}
	d.snapshot.Store(snap)
}

// toPod returns the decision view of an endpoint, without its name.
func (d *HeteroPDDecider) toPod(md *fwkdl.EndpointMetadata, m *fwkdl.Metrics) hetero.Pod {
	pod := hetero.Pod{Class: md.Labels[d.config.AcceleratorLabel]}
	if raw := md.Labels[d.config.SpeedLabel]; d.config.SpeedLabel != "" && raw != "" {
		if v, err := strconv.ParseFloat(raw, 64); err == nil && v > 0 && !math.IsInf(v, 0) {
			pod.PrefillTokensPerSec = v
		}
	}
	if raw := md.Labels[d.config.DisableLabel]; d.config.DisableLabel != "" && raw != "" {
		pod.Disabled, _ = strconv.ParseBool(raw)
	}
	if m != nil {
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

// disaggregate reports whether remote prefill should run for this request. A read failure is
// logged and returns false, so the request is served decode-only.
func (d *HeteroPDDecider) disaggregate(ctx context.Context, request *scheduling.InferenceRequest, endpoint scheduling.Endpoint) bool {
	logger := logr.FromContextOrDiscard(ctx)
	if endpoint == nil || endpoint.GetMetadata() == nil {
		logger.Error(errors.New("endpoint is nil"), "hetero-pd-decider: skipping remote prefill")
		return false
	}
	uncached, err := d.uncachedTokens(request, endpoint)
	if err != nil {
		logger.Error(err, "hetero-pd-decider: skipping remote prefill")
		return false
	}
	md := endpoint.GetMetadata()
	entry := d.snapshot.Load().byDecode[md.ID]
	decodePod := d.toPod(md, endpoint.GetMetrics())
	decodePod.Name = entry.name
	candidates := entry.candidates

	decision := hetero.Decide(d.config.Config, uncached, decodePod, candidates)
	if decision.Fallback {
		decision.Remote = decision.Prefill != "" && d.config.FallbackNonCachedTokens > 0 &&
			uncached >= d.config.FallbackNonCachedTokens
	}
	recordHeteroPDDecision(d.typedName.Name, decision.Reason, decision.Remote)
	if v := logger.V(logging.DEBUG); v.Enabled() {
		v.Info("hetero-pd-decider decision",
			"remote", decision.Remote, "reason", decision.Reason, "strategy", decision.Strategy,
			"fallback", decision.Fallback, "uncachedTokens", uncached, "candidates", len(candidates),
			"prefillPod", decision.Prefill, "remoteSeconds", decision.RemoteSeconds, "localSeconds", decision.LocalSeconds,
			"stallSeconds", decision.StallSeconds, "transferSeconds", decision.TransferSeconds)
	}
	return decision.Remote
}
