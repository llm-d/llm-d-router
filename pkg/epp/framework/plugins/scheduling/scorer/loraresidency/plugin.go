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

// Package loraresidency is an EPP scorer that prefers pods where serving a
// LoRA request is estimated to be cheapest, from a router-side model of each
// pod's GPU and CPU-cache adapters (package residency) and per-adapter demand.
//
// One plugin instance is the scorer, the request-control hooks that feed the
// model (PreRequest, ResponseHeader, ResponseBody) and the endpoint-lifecycle
// extractor that forgets deleted pods. The runner wires every declared plugin
// into each hook it implements, so configuring it once as a scorer is enough.
package loraresidency

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/metrics"
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/loraresidency/lorascore"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/loraresidency/lorascore/residency"
)

// Type is the plugin type name used in EndpointPickerConfig.
const Type = "lora-residency-scorer"

// tombstoneTTL bounds how long a deleted ID refuses new routes. It only has to
// outlast a scheduling cycle that raced the delete (late callbacks find no
// record, since the delete purged them), and expiring it keeps the set from
// growing with every pod name that never returns.
const tombstoneTTL = 5 * time.Minute

// generationKey names the endpoint attribute carrying the plugin's generation
// number for an endpoint object. The director clones endpoint attributes into
// each request's candidate snapshot, so the picked snapshot endpoint says which
// generation the decision was made on.
var generationKey = fwkplugin.NewDataKey("LoRAResidencyGeneration", Type)

// generation is the attribute value stored under generationKey.
type generation uint64

// Clone implements fwkdl.Cloneable.
func (g generation) Clone() fwkdl.Cloneable { return g }

var (
	_ fwksched.Scorer               = &Plugin{}
	_ fwkplugin.ConsumerPlugin      = &Plugin{}
	_ fwkplugin.ProducerPlugin      = &Plugin{}
	_ fwkrc.PreRequest              = &Plugin{}
	_ fwkrc.ResponseHeaderProcessor = &Plugin{}
	_ fwkrc.ResponseBodyProcessor   = &Plugin{}
	_ fwkdl.EndpointExtractor       = &Plugin{}
	_ fwkdl.Registrant              = &Plugin{}
)

// Parameters is the plugin's configuration.
type Parameters struct {
	// BaseModels names the pool's base models. A request whose TargetModel is
	// one of them is a base-model request and is scored neutrally. Required:
	// TargetModel is set for base-model requests too, and absence from the
	// LoRA metrics cannot tell a base model from a cold adapter.
	BaseModels []string `json:"baseModels"`
	// MaxCPULoras is the pods' --max-cpu-loras; 0 means vLLM's default of
	// max_cpu_loras = max_loras.
	MaxCPULoras int `json:"maxCpuLoras,omitempty"`
	// DemandHalfLife is how fast the per-adapter demand estimate forgets, as
	// a Go duration. Default "60s".
	DemandHalfLife string `json:"demandHalfLife,omitempty"`
	// Weights override lorascore.DefaultWeights field by field.
	CopyCost       *float64 `json:"copyCost,omitempty"`
	LoadCost       *float64 `json:"loadCost,omitempty"`
	BlockedCost    *float64 `json:"blockedCost,omitempty"`
	PendingCost    *float64 `json:"pendingCost,omitempty"`
	EvictionWeight *float64 `json:"evictionWeight,omitempty"`
}

// Factory builds the plugin from its configuration parameters.
func Factory(name string, params *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	var p Parameters
	if params != nil {
		if err := params.Decode(&p); err != nil {
			return nil, fmt.Errorf("%s: parameters: %w", Type, err)
		}
	}
	return New(name, p)
}

// request is what the hooks remember about one routed LoRA request.
type request struct {
	pod, adapter string
	errorStatus  bool
	started      bool
}

// Plugin is the scorer and its feeding hooks.
type Plugin struct {
	typedName  fwkplugin.TypedName
	baseModels map[string]bool
	weights    lorascore.Weights
	model      *residency.Model
	demand     *lorascore.Demand
	now        func() time.Time

	mu       sync.Mutex
	requests map[string]*request // by RequestID
	// endpoints is the endpoint object last added under each ID, so a stale
	// delete for a replaced object can be told apart.
	endpoints map[string]fwkdl.Endpoint
	// deleted holds when each ID's delete event arrived, cleared by an add,
	// so a scheduling cycle that raced the delete cannot recreate the pod.
	deleted map[string]time.Time
	// seq numbers endpoint objects; genSeq holds the number of each live
	// ID's current object, which is also stamped on that object under
	// generationKey. The director snapshots candidates before scoring and
	// calls PreRequest after, so PreRequest compares the picked snapshot's
	// stamp with genSeq and rejects an old generation.
	seq    uint64
	genSeq map[string]uint64
}

// New validates p and returns a plugin named name.
func New(name string, p Parameters) (*Plugin, error) {
	if len(p.BaseModels) == 0 {
		return nil, errors.New(Type + ": baseModels is required")
	}
	if p.MaxCPULoras < 0 {
		return nil, fmt.Errorf("%s: maxCpuLoras = %d, want >= 0", Type, p.MaxCPULoras)
	}
	halfLife := time.Minute
	if p.DemandHalfLife != "" {
		d, err := time.ParseDuration(p.DemandHalfLife)
		if err != nil {
			return nil, fmt.Errorf("%s: demandHalfLife: %w", Type, err)
		}
		halfLife = d
	}
	demand, err := lorascore.NewDemand(halfLife)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", Type, err)
	}
	w := lorascore.DefaultWeights()
	for _, o := range []struct {
		src *float64
		dst *float64
	}{
		{p.CopyCost, &w.CopyCost}, {p.LoadCost, &w.LoadCost}, {p.BlockedCost, &w.BlockedCost},
		{p.PendingCost, &w.PendingCost}, {p.EvictionWeight, &w.EvictionWeight},
	} {
		if o.src != nil {
			*o.dst = *o.src
		}
	}
	if err := w.Validate(); err != nil {
		return nil, fmt.Errorf("%s: %w", Type, err)
	}
	bases := make(map[string]bool, len(p.BaseModels))
	for _, b := range p.BaseModels {
		bases[b] = true
	}
	return &Plugin{
		typedName:  fwkplugin.TypedName{Type: Type, Name: name},
		baseModels: bases,
		weights:    w,
		model:      residency.NewModel(p.MaxCPULoras),
		demand:     demand,
		now:        time.Now,
		requests:   map[string]*request{},
		endpoints:  map[string]fwkdl.Endpoint{},
		deleted:    map[string]time.Time{},
		genSeq:     map[string]uint64{},
	}, nil
}

// TypedName returns the type and name tuple of this plugin instance.
func (s *Plugin) TypedName() fwkplugin.TypedName { return s.typedName }

// Category returns the preference the scorer applies.
func (s *Plugin) Category() fwksched.ScorerCategory { return fwksched.Affinity }

// Consumes declares the per-pod LoRA metrics the scorer reads.
func (s *Plugin) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{
			fwkplugin.NewDataKey(metrics.ActiveModelsKey, metrics.MetricsExtractorType): map[string]int{},
		},
	}
}

// Produces declares the generation stamp Extract writes. The scorer's data
// scope hides undeclared keys and lets a producer read back its own, so this
// declaration also keeps the stamp visible to Score.
func (s *Plugin) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{generationKey: generation(0)}
}

// Score refreshes each candidate's capacity and activity snapshot from its
// scraped metrics, then scores it by the estimated cost of serving the request.
func (s *Plugin) Score(_ context.Context, req *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	ids := make([]string, 0, len(endpoints))
	byID := make(map[string]fwksched.Endpoint, len(endpoints))
	for _, ep := range endpoints {
		id := endpointID(ep)
		if id == "" {
			continue
		}
		ids = append(ids, id)
		byID[id] = ep
		// A snapshot taken before a delete or a replacement can be scored
		// after it; observing it would recreate an estimate for a deleted pod
		// that nothing removes, or write the old object's metrics into the
		// replacement's estimate.
		if s.outdated(ep, id) {
			continue
		}
		if m := ep.GetMetrics(); m != nil {
			s.model.ObserveCapacity(id, m.MaxActiveModels)
			// ActiveModels is running OR queued, so it protects, never touches.
			s.model.ObserveActive(id, keys(m.ActiveModels))
		}
	}
	adapter, isBase := "", true
	if req != nil {
		adapter, isBase = req.TargetModel, s.baseModels[req.TargetModel]
	}
	scores := lorascore.Score(s.model, s.demand, adapter, isBase, ids, s.weights, s.now())
	out := make(map[fwksched.Endpoint]float64, len(scores))
	for id, v := range scores {
		out[byID[id]] = v
	}
	return out
}

// PreRequest records the routed request against the pod finally picked.
func (s *Plugin) PreRequest(_ context.Context, req *fwksched.InferenceRequest, result *fwksched.SchedulingResult) error {
	if req == nil || req.RequestID == "" || s.baseModels[req.TargetModel] {
		return nil
	}
	picked := pickedEndpoint(result)
	pod := endpointID(picked)
	if pod == "" {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, dup := s.requests[req.RequestID]; dup || s.tombstoned(pod) || s.staleSnapshot(picked, pod) {
		return nil
	}
	s.requests[req.RequestID] = &request{pod: pod, adapter: req.TargetModel}
	s.model.Routed(pod, req.TargetModel)
	s.demand.Observe(req.TargetModel, s.now())
	return nil
}

// ResponseHeader notes an error status, so that its body is not mistaken for
// evidence that the adapter executed. The EPP copies Envoy's header keys
// verbatim, and Envoy sends the status as the ":status" pseudo-header; a plain
// "status" key is also accepted, as the EPP's own check does.
func (s *Plugin) ResponseHeader(_ context.Context, req *fwksched.InferenceRequest, resp *fwkrc.Response, _ *fwkdl.EndpointMetadata) {
	if req == nil || resp == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if r, ok := s.requests[req.RequestID]; ok {
		status := resp.Headers[":status"]
		if status == "" {
			status = resp.Headers["status"]
		}
		r.errorStatus = status != "" && !strings.HasPrefix(status, "2")
	}
}

// ResponseBody turns the first successful chunk into Started and the end of
// the stream, however it ended, into Finished.
//
// A call with both StartOfStream and EndOfStream is either a complete one-shot
// response or the server's cleanup for a request that exited before any body,
// which carries a non-natural TerminationCause. Only the first ends naturally,
// so only it counts as execution.
func (s *Plugin) ResponseBody(_ context.Context, req *fwksched.InferenceRequest, resp *fwkrc.Response, _ *fwkdl.EndpointMetadata) {
	if req == nil || resp == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	r, ok := s.requests[req.RequestID]
	if !ok {
		return
	}
	now := s.now()
	firstChunk := resp.StartOfStream &&
		(!resp.EndOfStream || resp.TerminationCause == fwkrc.TerminationCauseNatural)
	if firstChunk && !r.started && !r.errorStatus {
		r.started = true
		s.model.Started(r.pod, r.adapter, now)
	}
	if resp.EndOfStream {
		s.model.Finished(r.pod, r.adapter, now, r.started)
		delete(s.requests, req.RequestID)
	}
}

// RegisterDependencies subscribes the plugin to endpoint lifecycle events.
func (s *Plugin) RegisterDependencies(r fwkdl.Registrar) error {
	return r.Register(fwkdl.PendingRegistration{
		Owner:      s.TypedName(),
		SourceType: sourcenotifications.EndpointNotificationSourceType,
		Extractor:  s,
		DefaultSource: sourcenotifications.NewEndpointDataSource(
			sourcenotifications.EndpointNotificationSourceType, sourcenotifications.EndpointNotificationSourceType),
	})
}

// Extract tracks endpoint lifecycle by endpoint object, which is the
// generation: Runtime.NewEndpoint builds a fresh object for each add and
// Runtime.UpdateEndpoint re-dispatches the existing one. A delete forgets the pod's estimate and every request record on
// it, and tombstones the ID until it is added again, so late callbacks cannot
// recreate state that a pod recreated under the same name would inherit. An
// add carrying a different object under a known ID starts a new generation the
// same way, since the old object's delete may arrive later and is then ignored
// as stale.
func (s *Plugin) Extract(_ context.Context, event fwkdl.EndpointEvent) error {
	if event.Endpoint == nil || event.Endpoint.GetMetadata() == nil {
		return nil
	}
	id := event.Endpoint.GetMetadata().ID.String()
	s.mu.Lock()
	defer s.mu.Unlock()
	switch event.Type {
	case fwkdl.EventAddOrUpdate:
		if cur, known := s.endpoints[id]; !known || cur != event.Endpoint {
			// A new object is a new pod: nothing under this ID before it,
			// from an older generation or from before its first add, applies.
			s.forget(id)
			s.seq++
			s.genSeq[id] = s.seq
			event.Endpoint.GetAttributes().Put(generationKey, generation(s.seq))
		}
		s.endpoints[id] = event.Endpoint
		delete(s.deleted, id)
	case fwkdl.EventDelete:
		if cur, ok := s.endpoints[id]; ok && cur != event.Endpoint {
			return nil
		}
		delete(s.endpoints, id)
		now := s.now()
		for d, at := range s.deleted {
			if now.Sub(at) >= tombstoneTTL {
				delete(s.deleted, d)
			}
		}
		s.deleted[id] = now
		delete(s.genSeq, id)
		s.forget(id)
	}
	return nil
}

// staleSnapshot reports whether the picked snapshot endpoint was cloned from an
// endpoint object other than pod's current one: its stamp differs from the
// current generation, or the pod has none (deleted). An endpoint without a
// stamp (added before this plugin saw it) cannot be checked and is accepted.
// Callers hold s.mu.
func (s *Plugin) staleSnapshot(picked fwksched.Endpoint, pod string) bool {
	v, ok := picked.Get(generationKey)
	if !ok {
		return false
	}
	g, ok := v.(generation)
	cur, live := s.genSeq[pod]
	return !ok || !live || uint64(g) != cur
}

// outdated reports, for callers not holding s.mu, whether a snapshot endpoint
// belongs to a deleted pod or to an endpoint object since replaced.
func (s *Plugin) outdated(ep fwksched.Endpoint, id string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.tombstoned(id) || s.staleSnapshot(ep, id)
}

// tombstoned reports whether id was deleted within tombstoneTTL and not added
// since. Callers hold s.mu.
func (s *Plugin) tombstoned(id string) bool {
	at, ok := s.deleted[id]
	return ok && s.now().Sub(at) < tombstoneTTL
}

// forget drops pod id's estimate and every request record on it. Callers hold
// s.mu.
func (s *Plugin) forget(id string) {
	for rid, r := range s.requests {
		if r.pod == id {
			delete(s.requests, rid)
		}
	}
	s.model.Delete(id)
}

func pickedEndpoint(result *fwksched.SchedulingResult) fwksched.Endpoint {
	if result == nil {
		return nil
	}
	pr := result.ProfileResults[result.PrimaryProfileName]
	if pr == nil || len(pr.TargetEndpoints) == 0 {
		return nil
	}
	return pr.TargetEndpoints[0]
}

func endpointID(ep fwksched.Endpoint) string {
	if ep == nil || ep.GetMetadata() == nil {
		return ""
	}
	return ep.GetMetadata().ID.String()
}

func keys(m map[string]int) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}
