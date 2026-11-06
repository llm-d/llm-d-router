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

// Package coldsession provides an opt-in scheduling filter that keeps new
// sessions from piling onto a prefill endpoint that is already busy with a
// cold, full-context prefill.
package coldsession

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/sessionstate"
)

const (
	// ColdSessionFilterType is the type of the cold-session filter.
	ColdSessionFilterType = "cold-session-filter"

	defaultProfileName               = "prefill"
	defaultMaxConcurrentColdPrefills = 1
	defaultRequestTTL                = 10 * time.Minute
)

// parameters configures the filter.
type parameters struct {
	// ProfileName is the scheduling profile whose selected endpoint is tracked.
	// Defaults to "prefill".
	ProfileName string `json:"profileName,omitempty"`
	// MaxConcurrentColdPrefills is the number of cold prefills allowed on one
	// endpoint before new cold sessions are routed elsewhere. Defaults to 1.
	MaxConcurrentColdPrefills int `json:"maxConcurrentColdPrefills,omitempty"`
	// RequestTTLSeconds bounds how long an entry is kept if the response never
	// reaches end-of-stream. Defaults to 600.
	RequestTTLSeconds int `json:"requestTtlSeconds,omitempty"`
}

type config struct {
	profileName string
	max         int
	ttl         time.Duration
}

func (p *parameters) applyDefaults() {
	if p.ProfileName == "" {
		p.ProfileName = defaultProfileName
	}
	if p.MaxConcurrentColdPrefills == 0 {
		p.MaxConcurrentColdPrefills = defaultMaxConcurrentColdPrefills
	}
	if p.RequestTTLSeconds == 0 {
		p.RequestTTLSeconds = int(defaultRequestTTL / time.Second)
	}
}

func (p *parameters) toConfig() (config, error) {
	if p.MaxConcurrentColdPrefills < 1 {
		return config{}, fmt.Errorf("maxConcurrentColdPrefills must be at least 1, got %d", p.MaxConcurrentColdPrefills)
	}
	if p.RequestTTLSeconds < 1 {
		return config{}, fmt.Errorf("requestTtlSeconds must be at least 1, got %d", p.RequestTTLSeconds)
	}
	return config{
		profileName: p.ProfileName,
		max:         p.MaxConcurrentColdPrefills,
		ttl:         time.Duration(p.RequestTTLSeconds) * time.Second,
	}, nil
}

type tracked struct {
	endpoint fwkdl.ID
	expires  time.Time
}

// ColdSessionFilter is a scheduling filter that limits how many new (cold)
// sessions are placed on one prefill endpoint at a time.
type ColdSessionFilter struct {
	typedName   fwkplugin.TypedName
	profileName string
	max         int
	ttl         time.Duration

	mu     sync.Mutex
	active map[string]tracked
	counts map[fwkdl.ID]int
	now    func() time.Time
}

var _ fwksched.Filter = &ColdSessionFilter{}
var _ fwkrc.PreRequest = &ColdSessionFilter{}
var _ fwkrc.ResponseBodyProcessor = &ColdSessionFilter{}

// Factory creates a ColdSessionFilter.
func Factory(name string, rawParameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params := parameters{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' filter - %w", ColdSessionFilterType, err)
		}
	}
	params.applyDefaults()
	cfg, err := params.toConfig()
	if err != nil {
		return nil, fmt.Errorf("invalid parameters of the '%s' filter - %w", ColdSessionFilterType, err)
	}
	if name == "" {
		name = ColdSessionFilterType
	}
	return &ColdSessionFilter{
		typedName:   fwkplugin.TypedName{Type: ColdSessionFilterType, Name: name},
		profileName: cfg.profileName,
		max:         cfg.max,
		ttl:         cfg.ttl,
		active:      make(map[string]tracked),
		counts:      make(map[fwkdl.ID]int),
		now:         time.Now,
	}, nil
}

// TypedName returns the plugin's type and name.
func (f *ColdSessionFilter) TypedName() fwkplugin.TypedName {
	return f.typedName
}

// Filter keeps cold sessions away from endpoints that already have the
// configured number of cold prefills in flight. It fails open when every
// candidate is busy so a new session is never starved by this filter alone.
func (f *ColdSessionFilter) Filter(_ context.Context, request *fwksched.InferenceRequest,
	endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	if !isCold(request) {
		return endpoints
	}

	f.mu.Lock()
	defer f.mu.Unlock()
	f.sweepLocked(f.now())

	allowed := make([]fwksched.Endpoint, 0, len(endpoints))
	for _, endpoint := range endpoints {
		md := endpoint.GetMetadata()
		if md == nil || f.counts[md.ID] < f.max {
			allowed = append(allowed, endpoint)
		}
	}
	if len(allowed) == 0 {
		return endpoints
	}
	return allowed
}

// PreRequest records the prefill endpoint selected for a cold request.
func (f *ColdSessionFilter) PreRequest(_ context.Context, request *fwksched.InferenceRequest,
	result *fwksched.SchedulingResult) error {
	if !isCold(request) || request == nil || result == nil {
		return nil
	}

	profile := result.ProfileResults[f.profileName]
	if profile == nil && result.PrimaryProfileName != "" {
		profile = result.ProfileResults[result.PrimaryProfileName]
	}
	if profile == nil || len(profile.TargetEndpoints) == 0 {
		return nil
	}
	endpoint := profile.TargetEndpoints[0]
	md := endpoint.GetMetadata()
	if md == nil {
		return nil
	}

	f.mu.Lock()
	defer f.mu.Unlock()
	now := f.now()
	f.sweepLocked(now)
	if _, exists := f.active[request.RequestID]; exists {
		return nil
	}
	f.counts[md.ID]++
	f.active[request.RequestID] = tracked{endpoint: md.ID, expires: now.Add(f.ttl)}
	return nil
}

// ResponseBody releases a cold request when its response reaches end-of-stream.
func (f *ColdSessionFilter) ResponseBody(_ context.Context, request *fwksched.InferenceRequest,
	response *fwkrc.Response, _ *fwkdl.EndpointMetadata) {
	if request == nil || response == nil || !response.EndOfStream {
		return
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.releaseLocked(request.RequestID)
}

func (f *ColdSessionFilter) releaseLocked(requestID string) {
	entry, ok := f.active[requestID]
	if !ok {
		return
	}
	delete(f.active, requestID)
	if f.counts[entry.endpoint] <= 1 {
		delete(f.counts, entry.endpoint)
		return
	}
	f.counts[entry.endpoint]--
}

func (f *ColdSessionFilter) sweepLocked(now time.Time) {
	for requestID, entry := range f.active {
		if !entry.expires.After(now) {
			f.releaseLocked(requestID)
		}
	}
}

func isCold(request *fwksched.InferenceRequest) bool {
	if request == nil {
		return false
	}
	state, ok := sessionstate.ReadSessionState(request)
	if !ok {
		return false
	}
	return state.TurnsTaken == 0 && state.InFlightRequests == 0
}
