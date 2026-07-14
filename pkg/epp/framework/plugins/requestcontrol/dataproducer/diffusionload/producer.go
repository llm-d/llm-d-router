/*
Copyright 2026 The Kubernetes Authors.

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

// Package diffusionload tracks the outstanding declared cost of in-flight
// image generation requests per endpoint. Unlike LLM requests, whose output
// length is unknown at admission, a diffusion request declares its compute
// cost in the body (inference steps x output resolution x image count), so
// the tracked load reflects actual GPU work rather than request count.
package diffusionload

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrdiffusion "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/diffusion"
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
	diffusionloadconstants "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/diffusionload/constants"
)

const (
	DiffusionLoadProducerType = diffusionloadconstants.DiffusionLoadProducerType

	// pixelsPerCostUnit normalizes declared pixel counts into megapixel-scale
	// cost units so a 1024x1024 single-image request costs one unit per step.
	pixelsPerCostUnit = 1024 * 1024

	defaultNumInferenceSteps = 50
	defaultSize              = "1024x1024"
)

// Config controls the defaults used when a request does not declare a cost field.
type Config struct {
	// DefaultNumInferenceSteps is the step count assumed when a request omits
	// num_inference_steps. Match it to the served model's default (e.g. a low
	// value for turbo/distilled models). Must be positive. Defaults to 50.
	DefaultNumInferenceSteps int64 `json:"defaultNumInferenceSteps,omitempty"`
	// DefaultSize is the output resolution ("WIDTHxHEIGHT") assumed when a
	// request omits size. Defaults to "1024x1024".
	DefaultSize string `json:"defaultSize,omitempty"`
}

func DiffusionLoadProducerFactory(name string, decoder *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	if handle == nil {
		return nil, errors.New("handle is nil")
	}
	ctx := handle.Context()
	if ctx == nil {
		ctx = context.Background()
	}

	cfg := Config{}
	if decoder != nil {
		if err := decoder.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("failed to decode diffusion-load-producer parameters: %w", err)
		}
	}

	steps := int64(defaultNumInferenceSteps)
	if cfg.DefaultNumInferenceSteps != 0 {
		if cfg.DefaultNumInferenceSteps < 0 {
			return nil, fmt.Errorf("defaultNumInferenceSteps must be positive, got %d", cfg.DefaultNumInferenceSteps)
		}
		steps = cfg.DefaultNumInferenceSteps
	}

	size := defaultSize
	if cfg.DefaultSize != "" {
		size = cfg.DefaultSize
	}
	pixels, err := parseSizePixels(size)
	if err != nil {
		return nil, fmt.Errorf("invalid defaultSize %q: %w", size, err)
	}

	return &DiffusionLoadProducer{
		typedName:     fwkplugin.TypedName{Type: DiffusionLoadProducerType, Name: name},
		costTracker:   newCostTracker(),
		defaultSteps:  steps,
		defaultPixels: pixels,
		dk:            attrdiffusion.DiffusionLoadDataKey.WithNonEmptyProducerName(name),
		PluginState:   fwkplugin.NewPluginState(ctx),
	}, nil
}

var (
	_ requestcontrol.PreRequest            = &DiffusionLoadProducer{}
	_ requestcontrol.ResponseBodyProcessor = &DiffusionLoadProducer{}
	_ requestcontrol.DataProducer          = &DiffusionLoadProducer{}
	_ datalayer.EndpointExtractor          = (*DiffusionLoadProducer)(nil)
	_ datalayer.Registrant                 = &DiffusionLoadProducer{}
)

// DiffusionLoadProducer tracks per-endpoint outstanding declared diffusion
// cost and publishes it as the DiffusionLoad endpoint attribute.
type DiffusionLoadProducer struct {
	typedName           fwkplugin.TypedName
	costTracker         *costTracker
	defaultSteps        int64
	defaultPixels       int64
	dk                  fwkplugin.DataKey
	PluginState         *fwkplugin.PluginState
	registeredEndpoints sync.Map // key: string (NamespacedName), value: datalayer.Endpoint
}

// addedCostEntry tracks a request's contribution to an endpoint's cost
// counter. OnEvicted rolls back the contribution exactly once, whether
// triggered by explicit release at end-of-stream or by the janitor's TTL
// reaper: whichever swaps first does the decrement, the other sees 0 and is
// a no-op. The counter pointer targets the exact instance incremented in
// PreRequest so a release always lands on it, even after an endpoint flap.
type addedCostEntry struct {
	cost        atomic.Int64
	costCounter *atomic.Int64
}

var _ fwkplugin.EvictableStateData = (*addedCostEntry)(nil)

// Clone returns a distinct copy of the entry with the current atomic value.
// The counter-instance pointer stays shared so the clone releases against the
// same counter the original incremented.
func (e *addedCostEntry) Clone() fwkplugin.StateData {
	if e == nil {
		return nil
	}
	clone := &addedCostEntry{costCounter: e.costCounter}
	clone.cost.Store(e.cost.Load())
	return clone
}

func (e *addedCostEntry) OnEvicted(_ string, _ fwkplugin.StateKey) {
	if c := e.cost.Swap(0); c != 0 {
		decrementClamped(e.costCounter, c)
	}
}

func (p *DiffusionLoadProducer) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// RegisterDependencies declares that this plugin needs an endpoint-notification-source
// to track endpoint lifecycle events. The source is auto-created if not already configured.
func (p *DiffusionLoadProducer) RegisterDependencies(r datalayer.Registrar) error {
	return r.Register(datalayer.PendingRegistration{
		Owner:         p.TypedName(),
		SourceType:    sourcenotifications.EndpointNotificationSourceType,
		Extractor:     p,
		DefaultSource: sourcenotifications.NewEndpointDataSource(sourcenotifications.EndpointNotificationSourceType, sourcenotifications.EndpointNotificationSourceType),
	})
}

// Extract releases per-endpoint cost state when an endpoint is removed.
func (p *DiffusionLoadProducer) Extract(ctx context.Context, event datalayer.EndpointEvent) error {
	if event.Endpoint == nil || event.Endpoint.GetMetadata() == nil {
		return nil
	}

	id := event.Endpoint.GetMetadata().NamespacedName.String()

	switch event.Type {
	case datalayer.EventDelete:
		// Assumes the datalayer delivers the same Endpoint pointer for delete as
		// for the preceding add. When the endpoint was replaced under the same
		// NamespacedName, the pointers differ and the stale delete is ignored so
		// it does not drop the live endpoint's counter.
		if registered, ok := p.registeredEndpoints.Load(id); ok && registered != event.Endpoint {
			log.FromContext(ctx).V(logutil.DEFAULT).Info("Ignoring stale delete for replaced endpoint", "endpoint", id)
			break
		}
		p.registeredEndpoints.Delete(id)
		p.costTracker.delete(id)
		log.FromContext(ctx).V(logutil.DEFAULT).Info("Cleaned up diffusion load for deleted endpoint", "endpoint", id)
	case datalayer.EventAddOrUpdate:
		// Track the endpoint pointer so a later EventDelete can be recognized as
		// stale (and ignored) when the endpoint was replaced under the same name.
		p.registeredEndpoints.Store(id, event.Endpoint)
	}
	return nil
}

// Produce publishes each candidate endpoint's current outstanding declared
// diffusion cost as the DiffusionLoad attribute, read by the diffusion-cost
// scorer. The value is a snapshot of the live counter at scheduling time.
func (p *DiffusionLoadProducer) Produce(_ context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) error {
	for _, endpoint := range endpoints {
		if endpoint == nil || endpoint.GetMetadata() == nil {
			continue
		}
		id := endpoint.GetMetadata().NamespacedName.String()
		endpoint.Put(p.dk.String(), &attrdiffusion.DiffusionLoad{CostUnits: p.costTracker.get(id)})
	}
	return nil
}

func (p *DiffusionLoadProducer) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{
		p.dk: attrdiffusion.DiffusionLoad{},
	}
}

func (p *DiffusionLoadProducer) PreRequest(ctx context.Context, request *fwksched.InferenceRequest, result *fwksched.SchedulingResult) {
	if result == nil || len(result.ProfileResults) == 0 {
		return
	}
	if request == nil || request.RequestID == "" || p.PluginState == nil {
		return
	}

	cost := p.requestCost(request)
	if cost == 0 {
		// Not an image generation request; nothing to track.
		return
	}

	for profileName, profileResult := range result.ProfileResults {
		if profileResult == nil || len(profileResult.TargetEndpoints) == 0 {
			continue
		}
		// Only track the primary target endpoint of each profile.
		endpoint := profileResult.TargetEndpoints[0]
		if endpoint == nil || endpoint.GetMetadata() == nil {
			continue
		}
		eid := endpoint.GetMetadata().NamespacedName.String()
		counter := p.costTracker.add(eid, cost)

		entry := &addedCostEntry{costCounter: counter}
		entry.cost.Store(cost)
		p.PluginState.Write(
			request.RequestID,
			fwkplugin.StateKey(addedCostKey(eid, profileName)),
			entry,
		)
	}

	log.FromContext(ctx).V(logutil.DEBUG).Info("Tracked diffusion request cost", "requestID", request.RequestID, "costUnits", cost)
}

// ResponseBody releases the request's cost contribution when the response
// stream ends. Intermediate chunks extend the janitor lifetime.
func (p *DiffusionLoadProducer) ResponseBody(
	_ context.Context,
	request *fwksched.InferenceRequest,
	resp *requestcontrol.Response,
	_ *datalayer.EndpointMetadata,
) {
	if request == nil || resp == nil || request.RequestID == "" || p.PluginState == nil {
		return
	}

	if resp.EndOfStream {
		p.PluginState.Delete(request.RequestID)
	} else {
		p.PluginState.Touch(request.RequestID)
	}
}

// requestCost returns the declared cost of the request in step-megapixel
// units, or 0 when the request is not an image generation request. Fields the
// client omitted fall back to the configured defaults.
func (p *DiffusionLoadProducer) requestCost(request *fwksched.InferenceRequest) int64 {
	if request.Body == nil || request.Body.Images == nil {
		return 0
	}
	images := request.Body.Images

	steps := p.defaultSteps
	if images.NumInferenceSteps != nil && *images.NumInferenceSteps > 0 {
		steps = *images.NumInferenceSteps
	}

	pixels := p.defaultPixels
	if images.Size != "" {
		if parsed, err := parseSizePixels(images.Size); err == nil {
			pixels = parsed
		}
	}

	n := int64(1)
	if images.N != nil && *images.N > 0 {
		n = *images.N
	}

	cost := steps * n * pixels / pixelsPerCostUnit
	if cost < 1 {
		cost = 1
	}
	return cost
}

// parseSizePixels parses a "WIDTHxHEIGHT" size string into a pixel count.
func parseSizePixels(size string) (int64, error) {
	parts := strings.SplitN(strings.ToLower(strings.TrimSpace(size)), "x", 2)
	if len(parts) != 2 {
		return 0, fmt.Errorf("size must be WIDTHxHEIGHT, got %q", size)
	}
	width, err := strconv.ParseInt(strings.TrimSpace(parts[0]), 10, 64)
	if err != nil || width <= 0 {
		return 0, fmt.Errorf("invalid width in size %q", size)
	}
	height, err := strconv.ParseInt(strings.TrimSpace(parts[1]), 10, 64)
	if err != nil || height <= 0 {
		return 0, fmt.Errorf("invalid height in size %q", size)
	}
	return width * height, nil
}

func addedCostKey(endpointID, profileName string) string {
	return endpointID + "|" + profileName + "|addedCost"
}

func (p *DiffusionLoadProducer) GetCostUnits(eid string) int64 {
	return p.costTracker.get(eid)
}

// costTracker manages thread-safe per-endpoint cost counters.
type costTracker struct {
	mu     sync.RWMutex
	counts map[string]*atomic.Int64
}

func newCostTracker() *costTracker {
	return &costTracker{counts: make(map[string]*atomic.Int64)}
}

func (t *costTracker) get(endpointID string) int64 {
	t.mu.RLock()
	counter, exists := t.counts[endpointID]
	t.mu.RUnlock()

	if !exists {
		return 0
	}
	return counter.Load()
}

// add applies delta to the endpoint's counter, creating it if absent, and
// returns the exact counter instance mutated so the matching decrement always
// lands on it, even after an endpoint flap.
func (t *costTracker) add(endpointID string, delta int64) *atomic.Int64 {
	t.mu.RLock()
	counter, exists := t.counts[endpointID]
	t.mu.RUnlock()

	if exists {
		counter.Add(delta)
		return counter
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	if counter, exists = t.counts[endpointID]; exists {
		counter.Add(delta)
		return counter
	}

	counter = &atomic.Int64{}
	counter.Store(delta)
	t.counts[endpointID] = counter
	return counter
}

func (t *costTracker) delete(endpointID string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	delete(t.counts, endpointID)
}

// decrementClamped subtracts delta from counter with a hard floor at zero. The
// CAS loop keeps the floor race-safe against a concurrent increment on the same
// instance, which a plain Add followed by Store(0) could clobber.
func decrementClamped(counter *atomic.Int64, delta int64) {
	for {
		current := counter.Load()
		if current <= 0 {
			return
		}
		next := current - delta
		if next < 0 {
			next = 0
		}
		if counter.CompareAndSwap(current, next) {
			return
		}
	}
}
