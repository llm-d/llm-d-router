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
	"sync"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/clock"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

const (
	// NixlHealthPDDeciderPluginType is the type-name of the nixlHealthPDDecider plugin.
	NixlHealthPDDeciderPluginType = "nixl-health-pd-decider"

	defaultNixlFailedTransfersThreshold = 5
	defaultNixlFailureWindow            = time.Minute
)

// NixlHealthPDDeciderConfig holds the configuration for the nixlHealthPDDecider plugin.
type NixlHealthPDDeciderConfig struct {
	// Decider names the PD decider that decides for a decode endpoint whose
	// NIXL transfers are not failing.
	Decider string `json:"decider" pluginRef:""`

	// FailedTransfersThreshold is the number of failed NIXL transfers within
	// Window at which the decode endpoint serves requests without remote prefill.
	FailedTransfersThreshold *int `json:"failedTransfersThreshold,omitempty"`

	// Window is the period over which failed transfers are counted.
	Window *metav1.Duration `json:"window,omitempty"`
}

func (c NixlHealthPDDeciderConfig) validate() error {
	if c.Decider == "" {
		return errors.New("decider parameter of the NIXL health decider is required")
	}
	if c.FailedTransfersThreshold != nil && *c.FailedTransfersThreshold <= 0 {
		return fmt.Errorf("failedTransfersThreshold parameter of the NIXL health decider must be positive, got %d",
			*c.FailedTransfersThreshold)
	}
	if c.Window != nil && c.Window.Duration <= 0 {
		return fmt.Errorf("window parameter of the NIXL health decider must be positive, got %s", c.Window.Duration)
	}
	return nil
}

// compile-time type assertions
var (
	_ deciderPlugin           = &NixlHealthPDDecider{}
	_ handlerScopedConsumer   = &NixlHealthPDDecider{}
	_ plugin.ProducerPlugin   = &NixlHealthPDDecider{}
	_ prefixMatchInfoConsumer = &NixlHealthPDDecider{}
)

// NixlHealthPDDecider declines remote prefill for a decode endpoint whose
// engine reports failed NIXL KV transfers at or above a threshold within a
// window, and otherwise returns the decision of the decider it wraps.
//
// The engine's failed-transfer counter is cumulative. The decider keeps the
// increases it observed per endpoint, so a count reached before the decider
// first evaluated an endpoint does not count toward the threshold.
type NixlHealthPDDecider struct {
	typedName plugin.TypedName
	decider   deciderPlugin
	threshold float64
	window    time.Duration
	clock     clock.PassiveClock

	mu        sync.Mutex
	endpoints map[fwkdl.ID]*transferFailures
	lastSweep time.Time
}

// transferFailures is the failed-transfer history of one decode endpoint.
type transferFailures struct {
	lastValue float64
	lastSeen  time.Time
	// increases holds the counter increases observed within the window, oldest first.
	increases []counterIncrease
	failing   bool
}

type counterIncrease struct {
	at    time.Time
	count float64
}

// NixlHealthPDDeciderConfigParser parses and validates the plugin parameters.
// The returned config declares the wrapped decider as a plugin dependency.
func NixlHealthPDDeciderConfigParser(rawParameters *json.Decoder, _ plugin.Handle) (any, error) {
	config := NixlHealthPDDeciderConfig{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&config); err != nil {
			return nil, fmt.Errorf("failed to parse %s plugin config: %w", NixlHealthPDDeciderPluginType, err)
		}
	}
	if err := config.validate(); err != nil {
		return nil, err
	}
	return config, nil
}

// NixlHealthPDDeciderPluginFactory defines the factory function for creating
// a new instance of the NixlHealthPDDecider.
func NixlHealthPDDeciderPluginFactory(name string, rawParameters *json.Decoder,
	handle plugin.Handle) (plugin.Plugin, error) {
	parsed, err := NixlHealthPDDeciderConfigParser(rawParameters, handle)
	if err != nil {
		return nil, err
	}
	config := parsed.(NixlHealthPDDeciderConfig)

	p := handle.Plugin(config.Decider)
	if p == nil {
		return nil, fmt.Errorf("decider plugin not found: %s", config.Decider)
	}
	decider, ok := p.(deciderPlugin)
	if !ok {
		return nil, fmt.Errorf("plugin %s is not a PD decider", config.Decider)
	}

	threshold := defaultNixlFailedTransfersThreshold
	if config.FailedTransfersThreshold != nil {
		threshold = *config.FailedTransfersThreshold
	}
	window := defaultNixlFailureWindow
	if config.Window != nil {
		window = config.Window.Duration
	}

	d, err := NewNixlHealthPDDecider(decider, threshold, window)
	if err != nil {
		return nil, fmt.Errorf("failed to create %s plugin: %w", NixlHealthPDDeciderPluginType, err)
	}
	return d.WithName(name), nil
}

// NewNixlHealthPDDecider returns a NixlHealthPDDecider that wraps decider.
func NewNixlHealthPDDecider(decider deciderPlugin, threshold int, window time.Duration) (*NixlHealthPDDecider, error) {
	if decider == nil {
		return nil, errors.New("decider is required")
	}
	if threshold <= 0 {
		return nil, fmt.Errorf("threshold must be positive, got %d", threshold)
	}
	if window <= 0 {
		return nil, fmt.Errorf("window must be positive, got %s", window)
	}
	return &NixlHealthPDDecider{
		typedName: plugin.TypedName{Type: NixlHealthPDDeciderPluginType},
		decider:   decider,
		threshold: float64(threshold),
		window:    window,
		clock:     clock.RealClock{},
		endpoints: map[fwkdl.ID]*transferFailures{},
	}, nil
}

// TypedName returns the typed name of the plugin.
func (d *NixlHealthPDDecider) TypedName() plugin.TypedName {
	return d.typedName
}

// WithName sets the name of the plugin.
func (d *NixlHealthPDDecider) WithName(name string) *NixlHealthPDDecider {
	d.typedName.Name = name
	return d
}

// Produces declares what the wrapped decider writes, since it runs through
// this plugin.
func (d *NixlHealthPDDecider) Produces() map[plugin.DataKey]any {
	produced := map[plugin.DataKey]any{}
	if producer, ok := d.decider.(plugin.ProducerPlugin); ok {
		for key, witness := range producer.Produces() {
			produced[key] = witness
		}
	}
	return produced
}

// deciderConsumes declares the failed-transfer counter, optional because engines
// without the NixlConnector do not report it, plus what the wrapped decider reads.
func (d *NixlHealthPDDecider) deciderConsumes() plugin.DataDependencies {
	consumed := plugin.DataDependencies{
		Required: map[plugin.DataKey]any{},
		Optional: map[plugin.DataKey]any{
			attrmetrics.NixlFailedTransfersDataKey: attrmetrics.ScalarMetricValue(0),
		},
	}
	if deps, ok := deciderConsumes(d.decider); ok {
		for key, witness := range deps.Required {
			consumed.Required[key] = witness
		}
		for key, witness := range deps.Optional {
			consumed.Optional[key] = witness
		}
	}
	return consumed
}

func (d *NixlHealthPDDecider) prefixMatchInfoDataKey() plugin.DataKey {
	if consumer, ok := d.decider.(prefixMatchInfoConsumer); ok {
		return consumer.prefixMatchInfoDataKey()
	}
	return attrprefix.PrefixCacheMatchInfoDataKey
}

func (d *NixlHealthPDDecider) disaggregate(ctx context.Context, request *scheduling.InferenceRequest, endpoint scheduling.Endpoint) bool {
	if endpoint != nil && d.transfersFailing(ctx, endpoint) {
		log.FromContext(ctx).V(logging.DEBUG).Info("NIXL transfers failing on decode endpoint, decode only",
			"endpoint", endpoint.GetMetadata().ID)
		return false
	}
	return d.decider.disaggregate(ctx, request, endpoint)
}

// transfersFailing records the endpoint's failed-transfer counter and reports
// whether the failures within the window reach the threshold.
func (d *NixlHealthPDDecider) transfersFailing(ctx context.Context, endpoint scheduling.Endpoint) bool {
	value, ok := attrmetrics.ReadScalarMetricValue(endpoint, attrmetrics.NixlFailedTransfersDataKey)
	if !ok {
		return false
	}
	id := endpoint.GetMetadata().ID
	now := d.clock.Now()

	d.mu.Lock()
	d.forgetIdleEndpoints(now)
	history, seen := d.endpoints[id]
	if !seen {
		d.endpoints[id] = &transferFailures{lastValue: float64(value), lastSeen: now}
		d.mu.Unlock()
		return false
	}
	recent := history.observe(float64(value), now, d.window)
	failing := recent >= d.threshold
	changed := failing != history.failing
	history.failing = failing
	d.mu.Unlock()

	if changed {
		logger := log.FromContext(ctx)
		if failing {
			logger.Info("NIXL failed transfers reached the threshold, serving requests on the decode endpoint without remote prefill",
				"endpoint", id, "failedTransfers", recent, "threshold", d.threshold, "window", d.window)
		} else {
			logger.Info("NIXL failed transfers fell below the threshold, resuming remote prefill for the decode endpoint",
				"endpoint", id, "failedTransfers", recent, "threshold", d.threshold, "window", d.window)
		}
	}
	return failing
}

// forgetIdleEndpoints drops the history of endpoints not evaluated within the
// window, such as replaced pods. It scans at most once per window. The caller
// holds d.mu.
func (d *NixlHealthPDDecider) forgetIdleEndpoints(now time.Time) {
	if now.Sub(d.lastSweep) < d.window {
		return
	}
	d.lastSweep = now
	cutoff := now.Add(-d.window)
	for id, history := range d.endpoints {
		if !history.lastSeen.After(cutoff) {
			delete(d.endpoints, id)
		}
	}
}

// observe records the counter value seen at now and returns the failures
// counted within the window ending at now.
func (f *transferFailures) observe(value float64, now time.Time, window time.Duration) float64 {
	switch {
	case value < f.lastValue:
		// The engine restarted and its counter started again from zero.
		f.increases = f.increases[:0]
		if value > 0 {
			f.increases = append(f.increases, counterIncrease{at: now, count: value})
		}
	case value > f.lastValue:
		f.increases = append(f.increases, counterIncrease{at: now, count: value - f.lastValue})
	}
	f.lastValue = value
	f.lastSeen = now

	cutoff := now.Add(-window)
	expired := 0
	for expired < len(f.increases) && !f.increases[expired].at.After(cutoff) {
		expired++
	}
	f.increases = f.increases[expired:]

	var recent float64
	for _, increase := range f.increases {
		recent += increase.count
	}
	return recent
}
