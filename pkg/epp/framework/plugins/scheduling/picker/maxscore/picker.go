/*
Copyright 2025 The Kubernetes Authors.
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

// Package maxscore implements a scheduling picker that selects the endpoint(s) with the highest
// score calculated during the scoring phase.
//
// For detailed behavioral intent and configuration, see the package README.
package maxscore

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"sync"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker"
)

const (
	// MaxScorePickerType is the registered name of the max score picker plugin.
	MaxScorePickerType = "max-score-picker"
)

// compile-time type validation
var _ fwksched.Picker = &MaxScorePicker{}

// MaxScorePickerFactory defines the factory function for MaxScorePicker.
func MaxScorePickerFactory(name string, rawParameters *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	parameters := picker.PickerParameters{MaxNumOfEndpoints: picker.DefaultMaxNumOfEndpoints}
	if rawParameters != nil {
		if err := rawParameters.Decode(&parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' picker - %w", MaxScorePickerType, err)
		}
	}

	mp := NewMaxScorePicker(parameters.MaxNumOfEndpoints).WithName(name)
	// Issue #1598: pickers reserve the selected endpoint at pick time through any
	// registered EndpointReserver (the inflight-load-producer implements it), so
	// concurrent scheduling cycles observe each pick immediately instead of all
	// scoring against the same pre-pick snapshot (thundering herd). The handle is
	// retained and the reserver resolved lazily at first pick: at factory time,
	// plugins instantiated later in the config (e.g. the producer listed after
	// this picker) are not registered yet, and resolving eagerly would silently
	// leave the reservation path inactive.
	mp.handle = handle
	return mp, nil
}

// findReserver returns the first registered plugin implementing
// fwksched.EndpointReserver, or nil when none is configured.
func findReserver(handle fwkplugin.Handle) fwksched.EndpointReserver {
	for _, pl := range handle.GetAllPlugins() {
		if r, ok := pl.(fwksched.EndpointReserver); ok {
			return r
		}
	}
	return nil
}

// NewMaxScorePicker initializes a new MaxScorePicker and returns its pointer.
func NewMaxScorePicker(maxNumOfEndpoints int) *MaxScorePicker {
	if maxNumOfEndpoints <= 0 {
		maxNumOfEndpoints = picker.DefaultMaxNumOfEndpoints // on invalid configuration value, fallback to default value
	}

	return &MaxScorePicker{
		typedName:         fwkplugin.TypedName{Type: MaxScorePickerType, Name: MaxScorePickerType},
		maxNumOfEndpoints: maxNumOfEndpoints,
	}
}

// MaxScorePicker picks endpoint(s) with the highest score calculated during the scoring phase.
type MaxScorePicker struct {
	typedName         fwkplugin.TypedName
	maxNumOfEndpoints int // maximum number of endpoints to pick
	// handle is retained from the factory to resolve the reserver lazily at
	// first pick (issue #1598): plugins listed after this picker in the config
	// are not yet registered at factory time. Nil when constructed directly.
	handle fwkplugin.Handle
	// mu guards reserver for concurrent Pick calls.
	mu sync.RWMutex
	// reserver, when non-nil, is notified of each pick so concurrent scheduling
	// cycles see the decision immediately (issue #1598). May be nil when no
	// EndpointReserver plugin is configured.
	reserver fwksched.EndpointReserver
}

// WithName sets the picker's name
func (p *MaxScorePicker) WithName(name string) *MaxScorePicker {
	p.typedName.Name = name
	return p
}

// WithReserver wires an EndpointReserver (e.g. the inflight-load-producer) so
// the picker reserves its selection at pick time (issue #1598). Exported for
// tests and for programmatic wiring outside the factory path.
func (p *MaxScorePicker) WithReserver(r fwksched.EndpointReserver) *MaxScorePicker {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.reserver = r
	return p
}

// getReserver returns the configured EndpointReserver, resolving it lazily
// through the plugin handle on first use (and retrying while nil, since plugins
// may register after the first pick).
func (p *MaxScorePicker) getReserver() fwksched.EndpointReserver {
	p.mu.RLock()
	r := p.reserver
	p.mu.RUnlock()
	if r != nil || p.handle == nil {
		return r
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.reserver == nil {
		p.reserver = findReserver(p.handle)
	}
	return p.reserver
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *MaxScorePicker) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// Pick selects the endpoint(s) with the highest score calculated during the scoring phase.
func (p *MaxScorePicker) Pick(ctx context.Context, scoredEndpoints []*fwksched.ScoredEndpoint) *fwksched.ProfileRunResult {
	logger := log.FromContext(ctx)
	if logger.V(logutil.DEBUG).Enabled() {
		logger.V(logutil.DEBUG).Info("Selecting endpoints from candidates sorted by max score", "max-num-of-endpoints", p.maxNumOfEndpoints,
			"num-of-candidates", len(scoredEndpoints), "scored-endpoints", fwksched.EndpointScores(scoredEndpoints))
	}

	slices.SortFunc(scoredEndpoints, func(i, j *fwksched.ScoredEndpoint) int { // highest score first, ties by name
		if i.Score > j.Score {
			return -1
		}
		if i.Score < j.Score {
			return 1
		}
		a, b := i.GetMetadata().GetNamespacedName(), j.GetMetadata().GetNamespacedName()
		return cmp.Or(strings.Compare(a.Namespace, b.Namespace), strings.Compare(a.Name, b.Name))
	})

	// RotateScoredEndpoints provides deterministic round-robin tie-breaking
	// for equal-score candidates. Rotating within each equal-score tier ensures
	// traffic is distributed evenly without distortion from lower-scoring endpoints.
	counter := picker.PickerRand.NextCounter(picker.PoolKey(scoredEndpoints))
	for start := 0; start < len(scoredEndpoints) && start < p.maxNumOfEndpoints; {
		end := start + 1
		for end < len(scoredEndpoints) && scoredEndpoints[end].Score == scoredEndpoints[start].Score {
			end++
		}
		if end-start > 1 {
			picker.RotateScoredEndpoints(scoredEndpoints[start:end], counter)
		}
		start = end
	}

	// if we have enough endpoints to return keep only the "maxNumOfEndpoints" highest scored endpoints
	if p.maxNumOfEndpoints < len(scoredEndpoints) {
		scoredEndpoints = scoredEndpoints[:p.maxNumOfEndpoints]
	}

	targetEndpoints := make([]fwksched.Endpoint, len(scoredEndpoints))
	for i, scoredEndpoint := range scoredEndpoints {
		targetEndpoints[i] = scoredEndpoint
	}

	// Issue #1598: reserve the primary target at pick time so the next
	// concurrently-scheduled request sees this decision in its scoring input.
	// The reserver reads the request's uncached-token estimate from the
	// endpoint's Produce-phase attribute, so both the request count and the
	// token load are reserved.
	if r := p.getReserver(); r != nil && len(targetEndpoints) > 0 {
		r.ReserveFromEndpoint(targetEndpoints[0])
	}

	return &fwksched.ProfileRunResult{TargetEndpoints: targetEndpoints}
}
