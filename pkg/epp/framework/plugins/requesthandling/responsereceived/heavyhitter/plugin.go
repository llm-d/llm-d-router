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

// Package heavyhitter provides a response plugin that tracks the fairness IDs carrying the
// highest token-weighted load over a sliding window.
package heavyhitter

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"time"
	"unicode/utf8"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/common/duration"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/responsereceived/heavyhitter/spacesaving"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

const (
	// PluginType is the registered plugin type.
	PluginType = "heavy-hitter-tracker"

	// maxRequestWeight caps one request's charge, so a request with an extreme token count
	// cannot dominate a slice's accounting on its own.
	maxRequestWeight = math.MaxUint32

	// defaultSlotsPerSlice sizes the table when no threshold-based sizing applies.
	defaultSlotsPerSlice = 256

	// slotsHeadroom multiplies the derived table size, covering window load above the
	// operator's estimate.
	slotsHeadroom = 2

	// maxSlices and maxTrackedSlots bound the window's memory: slices * slotsPerSlice slots.
	// The cap also bounds the lock hold of a snapshot: the default sizing scans in hundreds of
	// microseconds, and the cap keeps the worst case near a millisecond.
	maxSlices        = 3600
	maxSlotsPerSlice = 1 << 16
	maxTrackedSlots  = 1 << 17

	// maxMetricTopK bounds the per-ID series the collector emits.
	maxMetricTopK = 100
)

// Parameters is the plugin configuration. The cost of a completed request, the quantity whose
// heaviest IDs the tracker reports, is
//
//	cost = costPerRequest + costPerInputToken*promptTokens + costPerOutputToken*completionTokens
type Parameters struct {
	// WindowSeconds is the retained span of the sliding window.
	WindowSeconds float64 `json:"windowSeconds,omitempty"`
	// SliceSeconds is the slice length; the window boundary moves in slice steps.
	SliceSeconds float64 `json:"sliceSeconds,omitempty"`
	// HeavyThreshold is the windowed load at or above which an ID is heavy. 0 reports the top
	// metricTopK IDs instead of the heavy ones.
	HeavyThreshold float64 `json:"heavyThreshold,omitempty"`
	// ExpectedWindowLoad is the operator's estimate of the peak total load of all IDs in one
	// window, in cost units. With HeavyThreshold it sizes the table: every ID whose windowed
	// load exceeds the window total divided by the slot count is guaranteed tracked, so a
	// table of slotsHeadroom * ExpectedWindowLoad / HeavyThreshold slots keeps every heavy ID
	// tracked with headroom.
	ExpectedWindowLoad float64 `json:"expectedWindowLoad,omitempty"`
	// SlotsPerSlice overrides the derived table size for expert tuning. 0 derives it from
	// HeavyThreshold and ExpectedWindowLoad.
	SlotsPerSlice int `json:"slotsPerSlice,omitempty"`
	// CostPerRequest is the fixed cost of one completed request.
	CostPerRequest float64 `json:"costPerRequest,omitempty"`
	// CostPerInputToken is the cost per prompt token.
	CostPerInputToken float64 `json:"costPerInputToken,omitempty"`
	// CostPerOutputToken is the cost per completion token.
	CostPerOutputToken float64 `json:"costPerOutputToken,omitempty"`
	// MetricTopK bounds the per-ID series the collector emits.
	MetricTopK int `json:"metricTopK,omitempty"`
}

func defaultParameters() Parameters {
	return Parameters{
		WindowSeconds:      300,
		SliceSeconds:       15,
		SlotsPerSlice:      0,
		CostPerInputToken:  1,
		CostPerOutputToken: 2,
		MetricTopK:         10,
	}
}

func (p Parameters) validate() error {
	if err := duration.ValidateSeconds("windowSeconds", p.WindowSeconds); err != nil {
		return err
	}
	if err := duration.ValidateSeconds("sliceSeconds", p.SliceSeconds); err != nil {
		return err
	}
	// Bound the ratio in float space: an int conversion of an out-of-range float is
	// implementation-defined, and the ratio of two valid durations can be astronomical.
	ratio := p.WindowSeconds / p.SliceSeconds
	switch {
	case p.SliceSeconds > p.WindowSeconds:
		return fmt.Errorf("sliceSeconds must be <= windowSeconds, got %v > %v", p.SliceSeconds, p.WindowSeconds)
	case ratio < 2:
		return fmt.Errorf("windowSeconds/sliceSeconds must yield at least 2 slices, got %v", ratio)
	case ratio > maxSlices:
		return fmt.Errorf("windowSeconds/sliceSeconds must yield at most %d slices, got %v", maxSlices, ratio)
	case p.HeavyThreshold < 0:
		return fmt.Errorf("heavyThreshold must be >= 0, got %v", p.HeavyThreshold)
	case p.ExpectedWindowLoad < 0:
		return fmt.Errorf("expectedWindowLoad must be >= 0, got %v", p.ExpectedWindowLoad)
	case p.HeavyThreshold > 0 && p.ExpectedWindowLoad == 0 && p.SlotsPerSlice == 0:
		return errors.New("heavyThreshold needs expectedWindowLoad or slotsPerSlice to size the table")
	case p.ExpectedWindowLoad > 0 && p.HeavyThreshold == 0:
		return errors.New("expectedWindowLoad needs heavyThreshold")
	case p.SlotsPerSlice < 0:
		return fmt.Errorf("slotsPerSlice must be >= 0, got %d", p.SlotsPerSlice)
	case p.SlotsPerSlice > maxSlotsPerSlice:
		return fmt.Errorf("slotsPerSlice must be <= %d, got %d", maxSlotsPerSlice, p.SlotsPerSlice)
	case p.CostPerRequest < 0:
		return fmt.Errorf("costPerRequest must be >= 0, got %v", p.CostPerRequest)
	case p.CostPerInputToken < 0:
		return fmt.Errorf("costPerInputToken must be >= 0, got %v", p.CostPerInputToken)
	case p.CostPerOutputToken < 0:
		return fmt.Errorf("costPerOutputToken must be >= 0, got %v", p.CostPerOutputToken)
	case p.CostPerRequest == 0 && p.CostPerInputToken == 0 && p.CostPerOutputToken == 0:
		return errors.New("cost coefficients cannot all be zero")
	case p.MetricTopK < 0:
		return fmt.Errorf("metricTopK must be >= 0, got %d", p.MetricTopK)
	case p.MetricTopK > maxMetricTopK:
		return fmt.Errorf("metricTopK must be <= %d, got %d", maxMetricTopK, p.MetricTopK)
	}
	return nil
}

// slotsPerSlice returns the Space-Saving table size per slice.
func (p Parameters) slotsPerSlice() (int, error) {
	if p.SlotsPerSlice > 0 {
		return p.SlotsPerSlice, nil
	}
	if p.HeavyThreshold == 0 {
		return defaultSlotsPerSlice, nil
	}
	// Compare in float space: an int conversion of an out-of-range float is
	// implementation-defined.
	sized := slotsHeadroom * p.ExpectedWindowLoad / p.HeavyThreshold
	if sized > maxSlotsPerSlice {
		return 0, fmt.Errorf("derived slotsPerSlice %.0f exceeds %d: raise heavyThreshold, lower expectedWindowLoad, or set slotsPerSlice", sized, maxSlotsPerSlice)
	}
	return int(math.Ceil(sized)), nil
}

// Plugin charges each completed request's cost to its user over a sliding window. See the
// package README.
type Plugin struct {
	typedName fwkplugin.TypedName
	params    Parameters
	window    *spacesaving.Window
}

var _ fwkrh.ResponseBodyProcessor = &Plugin{}

// PluginFactory builds the plugin from its parameters.
func PluginFactory(name string, rawParameters *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params := defaultParameters()
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' plugin - %w", PluginType, err)
		}
	}
	if err := params.validate(); err != nil {
		return nil, fmt.Errorf("%s plugin %q: %w", PluginType, name, err)
	}
	slots, err := params.slotsPerSlice()
	if err != nil {
		return nil, fmt.Errorf("%s plugin %q: %w", PluginType, name, err)
	}
	slices := int(math.Ceil(params.WindowSeconds / params.SliceSeconds))
	if slices*slots > maxTrackedSlots {
		return nil, fmt.Errorf("%s plugin %q: slices*slotsPerSlice must be <= %d, got %d*%d",
			PluginType, name, maxTrackedSlots, slices, slots)
	}
	if name == "" {
		name = PluginType
	}
	window, err := spacesaving.New(spacesaving.Config{
		Window: time.Duration(params.WindowSeconds * float64(time.Second)),
		Slice:  time.Duration(params.SliceSeconds * float64(time.Second)),
		Slots:  slots,
	}, time.Now())
	if err != nil {
		return nil, fmt.Errorf("%s plugin %q: %w", PluginType, name, err)
	}
	p := &Plugin{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: name},
		params:    params,
		window:    window,
	}
	if handle != nil {
		if reg := handle.Metrics(); reg != nil {
			if err := reg.Register(newCollector(name, p)); err != nil {
				return nil, fmt.Errorf("%s plugin %q: registering metrics: %w", PluginType, name, err)
			}
		}
	}
	return p, nil
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *Plugin) TypedName() fwkplugin.TypedName { return p.typedName }

// ResponseBody charges the request's cost to its user at end of stream. Every termination
// cause is counted: a disconnected or errored stream consumed capacity too.
func (p *Plugin) ResponseBody(
	_ context.Context,
	request *fwksched.InferenceRequest,
	response *fwkrh.Response,
	_ *datalayer.EndpointMetadata,
) {
	if request == nil || response == nil || !response.EndOfStream {
		return
	}
	cost := p.cost(request, response)
	if cost <= 0 {
		return
	}
	weight := math.Ceil(min(cost, maxRequestWeight))
	// The deprecated fairness-header alias is accepted, as the director accepts it.
	tenant, _ := metadata.GetLowerCaseHeaderValue(request.Headers, metadata.FlowFairnessIDKey)
	user := request.Headers[metadata.FlowUserIDKey]
	p.window.Add(time.Now(), normalizeID(trackingKey(tenant, user)), int64(weight))
}

// Top returns the heaviest IDs in the current window, heaviest first, at most k of them (all
// when k is negative).
func (p *Plugin) Top(k int) []spacesaving.Entry {
	return p.window.Snapshot(time.Now(), k).Entries
}

// Lookup returns the windowed entry for a tracking key, applying the same truncation as the
// update path, so a caller passing a key longer than the label bound still finds the entry.
func (p *Plugin) Lookup(key string) (spacesaving.Entry, bool) {
	return p.window.Lookup(time.Now(), normalizeID(key))
}

// IsHeavy reports whether the tracking key's windowed load is at or above the configured
// threshold, using the lower bound Weight - Error so an ID is never called heavy on load it
// may not have sent. It reports false while the threshold is unset.
func (p *Plugin) IsHeavy(key string) bool {
	if p.params.HeavyThreshold <= 0 {
		return false
	}
	entry, ok := p.Lookup(key)
	return ok && float64(entry.Weight-entry.Error) >= p.params.HeavyThreshold
}

func (p *Plugin) cost(request *fwksched.InferenceRequest, response *fwkrh.Response) float64 {
	promptTokens := response.Usage.PromptTokens
	completionTokens := response.Usage.CompletionTokens
	if completionTokens <= 0 && response.StreamedEvents > 0 {
		completionTokens = response.StreamedEvents
	}
	// Unlike weighted-fair-share, the prompt fallback also covers a response that ends with
	// no usage and no stream events: a request whose prefill ran but produced no output still
	// consumed the prompt's capacity.
	if promptTokens <= 0 && request.Body != nil {
		promptTokens = request.Body.TokenizedRequest.TokenCount()
	}
	return p.params.CostPerRequest +
		p.params.CostPerInputToken*float64(promptTokens) +
		p.params.CostPerOutputToken*float64(completionTokens)
}

// trackingKey builds the key a request is charged to: the user ID from its header, inside the
// tenant the fairness-ID header names, joined so the first "/"-separated segment of the key
// is exactly the tenant (a tenant-only key is escaped and so contains no separator; a
// user-only key composes onto the default tenant, as the director does for a request with no
// tenant). A request with neither identity falls back to the default flow.
func trackingKey(tenant, user string) string {
	switch {
	case user == "" && tenant == "":
		return reqcommon.DefaultFairnessID
	case user == "":
		return metadata.EscapeTenantID(tenant)
	case tenant == "":
		return reqcommon.DefaultTenantID + metadata.FairnessIDTenantSeparator + user
	default:
		return metadata.EscapeTenantID(tenant) + metadata.FairnessIDTenantSeparator + user
	}
}

// normalizeID truncates a tracking key at a rune boundary to stay admissible as a fairness_id
// label. Distinct keys sharing the truncated prefix are tracked as one; a key whose whole
// prefix is continuation bytes has no admissible cut and folds to the overflow value.
func normalizeID(id string) string {
	if id == "" {
		return reqcommon.DefaultFairnessID
	}
	if len(id) <= metricsutil.MaxLabelValueLength {
		return id
	}
	n := metricsutil.MaxLabelValueLength
	for n > 0 && !utf8.RuneStart(id[n]) {
		n--
	}
	if n == 0 {
		return metricsutil.OverflowValue
	}
	return id[:n]
}
