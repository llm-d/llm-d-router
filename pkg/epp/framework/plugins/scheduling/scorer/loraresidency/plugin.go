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
	"maps"
	"slices"
	"strings"
	"sync"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
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

var (
	_ fwksched.Scorer               = &Plugin{}
	_ fwkplugin.ConsumerPlugin      = &Plugin{}
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
	lorascore.Weights
}

// DefaultParameters returns parameters with the default weights and no base
// models.
func DefaultParameters() Parameters {
	return Parameters{Weights: lorascore.DefaultWeights()}
}

// Factory builds the plugin from its configuration parameters. Only one
// instance is supported: the datalayer binds one endpoint extractor per type,
// so a second instance would never see endpoint events.
func Factory(name string, params *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	p := DefaultParameters()
	if params != nil {
		if err := params.Decode(&p); err != nil {
			return nil, fmt.Errorf("%s: parameters: %w", Type, err)
		}
	}
	plugin, err := newPlugin(name, p)
	if err != nil {
		return nil, err
	}
	if handle != nil {
		for _, existing := range handle.GetAllPlugins() {
			if existing.TypedName().Type == Type {
				return nil, fmt.Errorf("multiple %q instances configured (%s, %s); only one instance is supported",
					Type, existing.TypedName(), plugin.TypedName())
			}
		}
	}
	return plugin, nil
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
	cpuSlots   int
	demand     *lorascore.Demand
	now        func() time.Time

	mu       sync.Mutex
	requests map[string]*request // by RequestID
	// endpoints is the endpoint object last added under each live ID. The
	// runtime builds a fresh object for each add and dispatches a delete with
	// the object it added, so a delete for a replaced object is told apart.
	endpoints map[string]liveEndpoint
	// pods is the estimate for each live ID.
	pods map[string]*residency.Pod
}

// liveEndpoint is the endpoint object last added under an ID, and when.
type liveEndpoint struct {
	obj   fwkdl.Endpoint
	added time.Time
}

// newPlugin validates p and returns a plugin named name. p's weights are used
// as given, so callers start from DefaultParameters, as Factory does.
func newPlugin(name string, p Parameters) (*Plugin, error) {
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
	if err := p.Validate(); err != nil {
		return nil, fmt.Errorf("%s: %w", Type, err)
	}
	bases := make(map[string]bool, len(p.BaseModels))
	for _, b := range p.BaseModels {
		bases[b] = true
	}
	return &Plugin{
		typedName:  fwkplugin.TypedName{Type: Type, Name: name},
		baseModels: bases,
		weights:    p.Weights,
		cpuSlots:   p.MaxCPULoras,
		demand:     demand,
		now:        time.Now,
		requests:   map[string]*request{},
		endpoints:  map[string]liveEndpoint{},
		pods:       map[string]*residency.Pod{},
	}, nil
}

// TypedName returns the type and name tuple of this plugin instance.
func (s *Plugin) TypedName() fwkplugin.TypedName { return s.typedName }

// Category returns the preference the scorer applies.
func (s *Plugin) Category() fwksched.ScorerCategory { return fwksched.Affinity }

// Consumes declares the per-pod LoRA metrics the scorer reads. MaxActiveModels
// has no data key; the metrics extractor writes it with ActiveModels, from the
// same metric.
func (s *Plugin) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{
			fwkplugin.NewDataKey(metrics.ActiveModelsKey, metrics.MetricsExtractorType): map[string]int{},
		},
	}
}

// isBase reports whether model is scored and tracked as a base-model request.
// An empty model names no adapter.
func (s *Plugin) isBase(model string) bool { return model == "" || s.baseModels[model] }

// Score refreshes each candidate's capacity and activity snapshot from its
// scraped metrics, then scores it by the estimated cost of serving the request.
// The estimate is read under s.mu and priced after it is released, so pricing
// does not hold up the response hooks.
func (s *Plugin) Score(ctx context.Context, req *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	adapter := ""
	if req != nil {
		adapter = req.TargetModel
	}
	isBase := s.isBase(adapter)
	byID := make(map[string]fwksched.Endpoint, len(endpoints))
	admissions := make(map[string]residency.Admission, len(endpoints))
	s.mu.Lock()
	for _, ep := range endpoints {
		id := endpointID(ep)
		if id == "" {
			continue
		}
		byID[id] = ep
		if s.current(ep, id) {
			if m := ep.GetMetrics(); m != nil {
				p := s.pod(id)
				if m.MaxActiveModels > 0 { // 0 means unreported
					p.SetCapacity(m.MaxActiveModels, s.cpuSlots)
				}
				// ActiveModels is running OR queued, so it protects, never touches.
				p.ObserveActive(slices.Collect(maps.Keys(m.ActiveModels)))
			}
		} else {
			log.FromContext(ctx).V(logging.DEBUG).Info("Not observing a stale candidate snapshot", "endpoint", id)
		}
		// A candidate with no estimate, including one whose endpoint was
		// deleted, is still scored, as a pod that holds nothing.
		a := residency.Admission{Tier: residency.Absent}
		if p := s.pods[id]; p != nil && !isBase {
			a = p.Admit(adapter)
		}
		admissions[id] = a
	}
	s.mu.Unlock()
	scores := lorascore.Score(admissions, s.demand, isBase, s.weights, s.now())
	out := make(map[fwksched.Endpoint]float64, len(scores))
	for id, v := range scores {
		out[byID[id]] = v
	}
	return out
}

// PreRequest records the routed request against the pod finally picked.
func (s *Plugin) PreRequest(ctx context.Context, req *fwksched.InferenceRequest, result *fwksched.SchedulingResult) error {
	if req == nil || req.RequestID == "" || s.isBase(req.TargetModel) {
		return nil
	}
	pod := endpointID(result.PrimaryEndpoint())
	if pod == "" {
		return nil
	}
	logger := log.FromContext(ctx)
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, dup := s.requests[req.RequestID]; dup {
		logger.V(logging.DEBUG).Info("Ignoring a repeated PreRequest", "requestId", req.RequestID)
		return nil
	}
	if !s.current(result.PrimaryEndpoint(), pod) {
		logger.V(logging.DEBUG).Info("Not recording a route to a deleted or replaced endpoint", "requestId", req.RequestID, "endpoint", pod)
		return nil
	}
	s.requests[req.RequestID] = &request{pod: pod, adapter: req.TargetModel}
	s.pod(pod).Routed(req.TargetModel)
	s.demand.Observe(req.TargetModel, s.now())
	logger.V(logging.TRACE).Info("Routed", "requestId", req.RequestID, "endpoint", pod, "adapter", req.TargetModel)
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
func (s *Plugin) ResponseBody(ctx context.Context, req *fwksched.InferenceRequest, resp *fwkrc.Response, _ *fwkdl.EndpointMetadata) {
	if req == nil || resp == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	r, ok := s.requests[req.RequestID]
	if !ok {
		return
	}
	logger := log.FromContext(ctx).V(logging.TRACE)
	now := s.now()
	firstChunk := resp.StartOfStream &&
		(!resp.EndOfStream || resp.TerminationCause == fwkrc.TerminationCauseNatural)
	if firstChunk && !r.started && !r.errorStatus {
		r.started = true
		s.pod(r.pod).Started(r.adapter, now)
		logger.Info("Started", "requestId", req.RequestID, "endpoint", r.pod, "adapter", r.adapter)
	}
	if resp.EndOfStream {
		s.pod(r.pod).Finished(r.adapter, now, r.started)
		delete(s.requests, req.RequestID)
		logger.Info("Finished", "requestId", req.RequestID, "endpoint", r.pod, "adapter", r.adapter, "started", r.started)
	}
}

// RegisterDependencies subscribes the plugin to endpoint lifecycle events.
// The default source registers the endpoint-notification datalayer source
// when the configuration does not declare one.
func (s *Plugin) RegisterDependencies(r fwkdl.Registrar) error {
	return r.Register(fwkdl.PendingRegistration{
		Owner:      s.TypedName(),
		SourceType: sourcenotifications.EndpointNotificationSourceType,
		Extractor:  s,
		DefaultSource: sourcenotifications.NewEndpointDataSource(
			sourcenotifications.EndpointNotificationSourceType, sourcenotifications.EndpointNotificationSourceType),
	})
}

// Extract tracks endpoint lifecycle by endpoint object. Runtime.NewEndpoint
// builds a fresh object for each add, and Runtime.UpdateEndpoint re-dispatches
// the existing one. An add carrying a different object under an ID, or a
// delete, forgets the pod's estimate and every request record on it, so a pod
// recreated under the same name starts empty. A delete for an object already
// replaced is ignored.
func (s *Plugin) Extract(ctx context.Context, event fwkdl.EndpointEvent) error {
	if event.Endpoint == nil || event.Endpoint.GetMetadata() == nil {
		return nil
	}
	id := event.Endpoint.GetMetadata().ID.String()
	logger := log.FromContext(ctx)
	s.mu.Lock()
	defer s.mu.Unlock()
	cur, known := s.endpoints[id]
	switch event.Type {
	case fwkdl.EventAddOrUpdate:
		if !known || cur.obj != event.Endpoint {
			s.forget(id)
			s.endpoints[id] = liveEndpoint{obj: event.Endpoint, added: s.now()}
			logger.V(logging.TRACE).Info("Tracking a new endpoint", "endpoint", id, "replaced", known)
		}
	case fwkdl.EventDelete:
		if known && cur.obj != event.Endpoint {
			logger.V(logging.DEBUG).Info("Ignoring a stale delete for a replaced endpoint", "endpoint", id)
			return nil
		}
		delete(s.endpoints, id)
		s.forget(id)
		logger.V(logging.TRACE).Info("Forgot a deleted endpoint", "endpoint", id)
	}
	return nil
}

// current reports whether ep, a candidate snapshot for id, describes id's live
// endpoint object. The snapshot is a copy, so it cannot be compared with the
// object; its metrics' scrape time can. The runtime dispatches a new object's
// add before its collector's first tick, so a snapshot whose metrics were
// scraped before the object was added was taken from the object it replaced.
// An unscraped snapshot is accepted: it is normally the new object's.
//
// The check is not exact. A scrape of the old object still in flight when
// its collector stops can stamp a time after the add, and an old object that
// was never scraped passes too. Either way one score observes the old
// object's activity until the next scrape, and one route is recorded on the
// replacement until that request ends. Callers hold s.mu.
func (s *Plugin) current(ep fwksched.Endpoint, id string) bool {
	cur, live := s.endpoints[id]
	if !live {
		return false
	}
	m := ep.GetMetrics()
	return !m.Updated() || !m.UpdateTime.Before(cur.added)
}

// pod returns id's estimate, creating it with one GPU slot until the pod
// reports max_lora. Callers hold s.mu.
func (s *Plugin) pod(id string) *residency.Pod {
	p, ok := s.pods[id]
	if !ok {
		p = residency.NewPod(1, s.cpuSlots)
		s.pods[id] = p
	}
	return p
}

// forget drops pod id's estimate and every request record on it. Callers hold
// s.mu.
func (s *Plugin) forget(id string) {
	for rid, r := range s.requests {
		if r.pod == id {
			delete(s.requests, rid)
		}
	}
	delete(s.pods, id)
}

func endpointID(ep fwksched.Endpoint) string {
	if ep == nil || ep.GetMetadata() == nil {
		return ""
	}
	return ep.GetMetadata().ID.String()
}
