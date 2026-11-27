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

// Package rendezvous provides a filter that keeps each session on one cluster-endpoint. See README.md.
package rendezvous

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/jellydator/ttlcache/v3"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/endpointattribute"
	rendezvousutil "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/util/rendezvous"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/util/sessionaffinity"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

const (
	// MultiClusterFilterType keys each request to one cluster-endpoint.
	MultiClusterFilterType = "multicluster-rendezvous-filter"
	// pinLabel is the cluster label listing pinned tenants (fairness IDs), comma separated.
	pinLabel = "pins"
	// defaultStickTTLSeconds is the default Config.StickTTLSeconds.
	defaultStickTTLSeconds = 1800
	// defaultMaxQueue is the default Config.MaxQueue.
	defaultMaxQueue = 4
	// defaultMaxWaitSeconds is the default Config.MaxWaitSeconds.
	defaultMaxWaitSeconds = 10

	defaultGuardName = "multicluster-rendezvous-default-guard"

	// maxStickTTL bounds stickTtlSeconds well below time.Duration overflow.
	maxStickTTL = 365 * 24 * time.Hour
	// maxMetricsAge bounds maxMetricsAgeSeconds: older metrics say nothing about a cluster's room.
	maxMetricsAge = time.Hour

	// spillStateKey keys a spill's Filter-to-PreRequest handoff in PluginState.
	spillStateKey plugin.StateKey = MultiClusterFilterType + "-spill"

	// defaultWeight weighs a cluster whose pool does not report its ready pods.
	defaultWeight = 1.0
	// maxMovedKeys bounds the moved table, whose keys are client-supplied; the least recently used is dropped.
	maxMovedKeys = 100_000
	// defaultMaxMetricsAgeSeconds is the default Config.MaxMetricsAgeSeconds.
	defaultMaxMetricsAgeSeconds = 30
)

// sessionSources identify a conversation: the agent-identity plugin's attribute (its session headers,
// e.g. Claude Code, OpenCode, Codex), then x-session-id. Without one, its text up to the first user message keys it.
var sessionSources = []sessionaffinity.SessionIDSource{
	{Attribute: agentIdentityAttribute}, {Header: sessionaffinity.DefaultSessionIDHeader},
}

// agentIdentityAttribute is the request attribute the agent-identity plugin publishes.
const agentIdentityAttribute = "agent-identity"

var (
	_ scheduling.Filter     = &Filter{}
	_ plugin.ConsumerPlugin = &Filter{}
	_ fwkrh.PreRequest      = &Filter{}
)

// Config is the filter's JSON parameters.
type Config struct {
	// MaxQueue is the built-in room check's limit on a cluster's average vLLM queue per pod.
	MaxQueue *float64 `json:"maxQueue,omitempty"`
	// MaxWaitSeconds is the built-in room check's limit on flow-control waits.
	MaxWaitSeconds *float64 `json:"maxWaitSeconds,omitempty"`
	// GuardPluginName replaces the built-in room check with a fail-closed filter.
	GuardPluginName string `json:"guardPluginName,omitempty" pluginRef:""`
	// StickTTLSeconds keeps a spilled conversation on its new cluster until unused this long; 0 disables.
	StickTTLSeconds *float64 `json:"stickTtlSeconds,omitempty"`
	// MaxMetricsAgeSeconds marks a cluster down when its pool metrics are older than this.
	MaxMetricsAgeSeconds *float64 `json:"maxMetricsAgeSeconds,omitempty"`
}

// ConfigParser decodes the parameters so the loader can resolve GuardPluginName first.
func ConfigParser(parameters *json.Decoder, _ plugin.Handle) (any, error) {
	config := &Config{}
	if parameters == nil {
		return config, nil
	}
	if err := parameters.Decode(config); err != nil {
		return nil, fmt.Errorf("failed to parse the parameters of the '%s' filter - %w", MultiClusterFilterType, err)
	}
	return config, nil
}

// MultiClusterFactory builds the filter from its JSON parameters.
func MultiClusterFactory(name string, raw *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	parsed, err := ConfigParser(raw, handle)
	if err != nil {
		return nil, err
	}
	config := *parsed.(*Config)
	var guard scheduling.Filter
	if config.GuardPluginName != "" {
		if config.MaxQueue != nil || config.MaxWaitSeconds != nil {
			return nil, fmt.Errorf("invalid parameters of the '%s' filter - maxQueue and maxWaitSeconds set the built-in "+
				"room check; with guardPluginName set the limits belong to that filter", MultiClusterFilterType)
		}
		guard, err = rendezvousutil.ResolveGuard(handle, config.GuardPluginName, "guardPluginName")
	} else {
		guard, err = newRoomCheck(limit(config.MaxQueue, defaultMaxQueue), limit(config.MaxWaitSeconds, defaultMaxWaitSeconds))
	}
	if err != nil {
		return nil, err
	}
	ctx := context.Background()
	if handle != nil {
		ctx = handle.Context()
	}
	f, err := New(ctx, name, config, guard)
	if err != nil {
		return nil, err
	}
	if handle != nil {
		if err := registerMetrics(handle.Metrics()); err != nil {
			return nil, err
		}
	}
	if handle != nil && f.moved != nil {
		go f.moved.Start() // expires unused entries
		go func() { <-handle.Context().Done(); f.moved.Stop() }()
	}
	return f, nil
}

// roomCheck is the built-in room check. A cluster with flow control is full for a priority-P
// request when a band of priority P or higher (served first) waited over maxWait, or held
// requests with no dispatch for maxWait. Without flow control, it is full when its average
// vLLM queue per pod reaches maxQueue or is not reported.
type roomCheck struct {
	maxWait float64
	queue   scheduling.Filter
	now     func() time.Time
}

func newRoomCheck(maxQueue, maxWait float64) (*roomCheck, error) {
	if !finiteNonNegative(maxQueue) || maxQueue == 0 || !finiteNonNegative(maxWait) || maxWait == 0 {
		return nil, fmt.Errorf("invalid parameters of the '%s' filter - maxQueue and maxWaitSeconds must be finite numbers > 0",
			MultiClusterFilterType)
	}
	params := fmt.Sprintf(`{"attribute":%q,"producer":"","onMissing":"Fail",
		"algorithm":{"type":"threshold","threshold":{"operator":"LessThan","value":%v}}}`,
		attrmetrics.MultiClusterQueueSizeKey, maxQueue)
	queue, err := endpointattribute.EndpointAttributeFilterFactory(defaultGuardName+"-queue",
		json.NewDecoder(strings.NewReader(params)), nil)
	if err != nil {
		return nil, fmt.Errorf("built-in room check of the '%s' filter: %w", MultiClusterFilterType, err)
	}
	return &roomCheck{maxWait: maxWait, queue: queue.(scheduling.Filter), now: time.Now}, nil
}

func (r *roomCheck) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: MultiClusterFilterType + "-room-check", Name: defaultGuardName}
}

func (r *roomCheck) Consumes() plugin.DataDependencies {
	deps := plugin.DataDependencies{Optional: map[plugin.DataKey]any{
		attrmetrics.MultiClusterFlowControlDataKey: attrmetrics.MultiClusterFlowControl{},
	}}
	return rendezvousutil.MergeGuardConsumes(deps, r.queue)
}

func (r *roomCheck) Filter(ctx context.Context, request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) []scheduling.Endpoint {
	priority := 0
	if request != nil {
		priority = request.Objectives.Priority
	}
	now := float64(r.now().UnixNano()) / 1e9
	kept := make([]scheduling.Endpoint, 0, len(endpoints))
	var noFlowControl []scheduling.Endpoint
	for _, endpoint := range endpoints {
		bands, ok := fwkdl.ReadAttribute[attrmetrics.MultiClusterFlowControl](endpoint, attrmetrics.MultiClusterFlowControlDataKey)
		if !ok || len(bands) == 0 {
			noFlowControl = append(noFlowControl, endpoint)
			continue
		}
		if !r.backedUp(bands, priority, now) {
			kept = append(kept, endpoint)
		}
	}
	if len(noFlowControl) > 0 {
		kept = append(kept, r.queue.Filter(ctx, request, noFlowControl)...)
	}
	return kept
}

// backedUp reports whether requests of priority or higher wait too long.
func (r *roomCheck) backedUp(bands attrmetrics.MultiClusterFlowControl, priority int, now float64) bool {
	for p, band := range bands {
		if p < priority {
			continue
		}
		if band.RecentWait > r.maxWait || (band.Held > 0 && now-band.LastProgress > r.maxWait) {
			return true
		}
	}
	return false
}

func limit(value *float64, fallback float64) float64 {
	if value == nil {
		return fallback
	}
	return *value
}

// New validates config and returns the filter. A nil guard turns off spill and failover.
// ctx bounds the per-request state's cleanup.
func New(ctx context.Context, name string, config Config, guard scheduling.Filter) (*Filter, error) {
	ttl := float64(defaultStickTTLSeconds)
	if config.StickTTLSeconds != nil {
		ttl = *config.StickTTLSeconds
		if !finiteNonNegative(ttl) || ttl > maxStickTTL.Seconds() {
			return nil, fmt.Errorf("invalid parameters of the '%s' filter - stickTtlSeconds must be between 0 and %v, got %v",
				MultiClusterFilterType, maxStickTTL.Seconds(), ttl)
		}
	}
	maxAge := float64(defaultMaxMetricsAgeSeconds)
	if config.MaxMetricsAgeSeconds != nil {
		maxAge = *config.MaxMetricsAgeSeconds
		if !finiteNonNegative(maxAge) || maxAge == 0 || maxAge > maxMetricsAge.Seconds() {
			return nil, fmt.Errorf("invalid parameters of the '%s' filter - maxMetricsAgeSeconds must be > 0 and <= %v, got %v",
				MultiClusterFilterType, maxMetricsAge.Seconds(), maxAge)
		}
	}
	f := &Filter{
		typedName: plugin.TypedName{Type: MultiClusterFilterType, Name: cmp.Or(name, MultiClusterFilterType)},
		guard:     guard,
		now:       time.Now,
	}

	// Room, staleness and spill memory all need the guard's pool metrics.
	if guard != nil {
		f.maxAge = maxAge
		if ttl > 0 {
			// A Get refreshes an entry's TTL, so a spilled conversation stays as long as it is used.
			f.moved = ttlcache.New(ttlcache.WithTTL[uint64, string](time.Duration(ttl*float64(time.Second))),
				ttlcache.WithCapacity[uint64, string](maxMovedKeys))
			f.pending = plugin.NewPluginState(ctx)
		}
	}
	return f, nil
}

// Filter keeps the one endpoint a request is moved or hashed to.
type Filter struct {
	typedName plugin.TypedName
	guard     scheduling.Filter
	maxAge    float64 // seconds; 0 = no staleness check
	now       func() time.Time
	// moved maps a spilled key to its cluster; nil when not remembered.
	moved *ttlcache.Cache[uint64, string]
	// pending hands a spilled request's key and home to PreRequest, which records the pick.
	pending *plugin.PluginState
}

// spill is a request handed to the scorers because its cluster was full.
type spill struct {
	key  uint64
	home string
}

func (s *spill) Clone() plugin.StateData { return &spill{key: s.key, home: s.home} }

// TypedName returns the type and name of this plugin instance.
func (f *Filter) TypedName() plugin.TypedName { return f.typedName }

// Consumes declares the metrics timestamp, ready pods, session attributes and what the guard reads.
func (f *Filter) Consumes() plugin.DataDependencies {
	deps := plugin.DataDependencies{Optional: map[plugin.DataKey]any{
		attrmetrics.MultiClusterReadyEndpointsDataKey: attrmetrics.ScalarMetricValue(0),
	}}
	for k, v := range sessionaffinity.SourcesConsumes(sessionSources).Optional {
		deps.Optional[k] = v
	}
	// Without the extractor every cluster would look down, so require it at startup.
	if f.maxAge > 0 {
		deps.Required = map[plugin.DataKey]any{attrmetrics.MultiClusterMetricsUpdatedAtDataKey: attrmetrics.ScalarMetricValue(0)}
	}
	// The guard runs on the endpoints scoped to this declaration.
	if f.guard != nil {
		deps = rendezvousutil.MergeGuardConsumes(deps, f.guard)
	}
	return deps
}

// Filter returns the request's cluster, or the clusters with room when it spills. See README.md for the order.
func (f *Filter) Filter(ctx context.Context, request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) []scheduling.Endpoint {
	if len(endpoints) <= 1 {
		return endpoints
	}
	shares := make(map[string]float64, len(endpoints))
	byName := make(map[string]scheduling.Endpoint, len(endpoints))
	for _, endpoint := range endpoints {
		name := endpoint.GetMetadata().ID.String()
		shares[name] = weight(endpoint)
		byName[name] = endpoint
	}
	alive := f.aliveOf(endpoints)
	room := f.roomOf(ctx, request, endpoints, alive)
	// No ready pods means no room: such a pool stops updating its queue metric, which would read as room.
	hasRoom := func(name string) bool { return shares[name] > 0 && room(name) }
	logger := log.FromContext(ctx).V(logging.DEBUG)

	key, source := rendezvousutil.Key(ctx, request, sessionSources)
	if source == rendezvousutil.SourceNone {
		// Nothing to hash or remember, but a pin still applies.
		if pin := pinned(request, endpoints); pin != "" && hasRoom(pin) {
			logger.Info("Rendezvous - picked endpoint", "plugin", f.typedName.Name, "chosen", pin, "endpoint", pin,
				"by", "pin", "spilled", false)
			f.record(pin, decisionPin)
			return []scheduling.Endpoint{byName[pin]}
		}
		return f.withRoom(endpoints, hasRoom, alive)
	}

	// usable: the cluster answers and has ready pods, room or not.
	usable := func(name string) bool { return byName[name] != nil && alive(name) && shares[name] > 0 }
	spilledTo := ""
	if f.moved != nil {
		if item := f.moved.Get(key); item != nil {
			if spilledTo = item.Value(); byName[spilledTo] != nil && hasRoom(spilledTo) {
				logger.Info("Rendezvous - kept spilled key", "plugin", f.typedName.Name, "endpoint", spilledTo)
				f.record(spilledTo, decisionKept)
				return []scheduling.Endpoint{byName[spilledTo]}
			}
		}
	}

	chosen, why := pinned(request, endpoints), "pin"
	if chosen == "" {
		chosen, why = rendezvousutil.Pick(shares, key), string(source)
	}
	if chosen == "" {
		return f.withRoom(endpoints, hasRoom, alive)
	}
	// A full home cluster spills: the scorers pick among clusters with room, and PreRequest remembers it.
	if !hasRoom(chosen) {
		if candidates := keep(endpoints, hasRoom); len(candidates) > 0 {
			if f.pending != nil && request != nil && request.RequestID != "" {
				f.pending.Write(request.RequestID, spillStateKey, &spill{key: key, home: chosen})
			}
			logger.Info("Rendezvous - spilling", "plugin", f.typedName.Name, "chosen", chosen,
				"by", why, "candidates", len(candidates))
			return candidates
		}
	}
	// No cluster has room: a spilled conversation stays where it is while usable, so its turns are not split.
	if !hasRoom(chosen) && spilledTo != "" && usable(spilledTo) {
		logger.Info("Rendezvous - kept spilled key", "plugin", f.typedName.Name, "endpoint", spilledTo, "noRoom", true)
		f.record(spilledTo, decisionKept)
		return []scheduling.Endpoint{byName[spilledTo]}
	}
	// Otherwise keep home, unless it cannot serve (down or no ready pods).
	final := chosen
	if !usable(final) {
		candidates := make(map[string]float64, len(shares))
		for name, share := range shares {
			if usable(name) {
				candidates[name] = share
			}
		}
		if next := rendezvousutil.Pick(candidates, key); next != "" {
			final = next
		}
	}
	if f.moved != nil {
		if final != chosen {
			f.moved.Set(key, final, ttlcache.DefaultTTL)
		} else {
			f.moved.Delete(key)
		}
	}
	logger.Info("Rendezvous - picked endpoint", "plugin", f.typedName.Name, "chosen", chosen, "endpoint", final,
		"by", why, "spilled", final != chosen)
	switch {
	case final != chosen:
		f.record(final, decisionFailover)
	case why == "pin":
		f.record(final, decisionPin)
	default:
		f.record(final, decisionHome)
	}
	return []scheduling.Endpoint{byName[final]}
}

// aliveOf reports whether an endpoint's pool metrics are fresh; without a guard every endpoint is.
func (f *Filter) aliveOf(endpoints []scheduling.Endpoint) func(string) bool {
	if f.maxAge <= 0 {
		return func(string) bool { return true }
	}
	now := float64(f.now().UnixNano()) / 1e9
	live := make(map[string]bool, len(endpoints))
	for _, endpoint := range endpoints {
		at, ok := attrmetrics.ReadScalarMetricValue(endpoint, attrmetrics.MultiClusterMetricsUpdatedAtDataKey)
		if ok && now-float64(at) <= f.maxAge {
			live[endpoint.GetMetadata().ID.String()] = true
		}
	}
	return func(name string) bool { return live[name] }
}

// roomOf runs the guard once over the live endpoints; without a guard every live endpoint has room.
func (f *Filter) roomOf(ctx context.Context, request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint,
	alive func(string) bool) func(string) bool {
	live := make([]scheduling.Endpoint, 0, len(endpoints))
	for _, endpoint := range endpoints {
		if alive(endpoint.GetMetadata().ID.String()) {
			live = append(live, endpoint)
		}
	}
	room := make(map[string]bool, len(live))
	kept := live
	if f.guard != nil && len(live) > 0 {
		kept = f.guard.Filter(ctx, request, live)
	}
	for _, endpoint := range kept {
		room[endpoint.GetMetadata().ID.String()] = true
	}
	return func(name string) bool { return room[name] }
}

// PreRequest records the cluster the scorers picked for a spill, so the conversation stays there.
func (f *Filter) PreRequest(ctx context.Context, request *scheduling.InferenceRequest, result *scheduling.SchedulingResult) error {
	if f.pending == nil || request == nil || request.RequestID == "" {
		return nil
	}
	// No pick (a later filter emptied the candidates): leave the moved table as it is.
	var target scheduling.Endpoint
	if result != nil {
		target = result.PrimaryEndpoint()
	}
	if target == nil {
		return nil
	}
	s, err := plugin.ReadPluginStateKey[*spill](f.pending, request.RequestID, spillStateKey)
	if err != nil {
		return nil
	}
	f.pending.Delete(request.RequestID)
	final := target.GetMetadata().ID.String()
	if final != s.home {
		f.moved.Set(s.key, final, ttlcache.DefaultTTL)
	} else {
		f.moved.Delete(s.key)
	}
	log.FromContext(ctx).V(logging.DEBUG).Info("Rendezvous - picked endpoint", "plugin", f.typedName.Name,
		"chosen", s.home, "endpoint", final, "by", "score", "spilled", final != s.home)
	if final != s.home {
		f.record(final, decisionSpill)
	} else {
		f.record(final, decisionHome)
	}
	return nil
}

// withRoom keeps the endpoints with room; else the live ones; else all.
func (f *Filter) withRoom(endpoints []scheduling.Endpoint, hasRoom, alive func(string) bool) []scheduling.Endpoint {
	for _, ok := range []func(string) bool{hasRoom, alive} {
		if kept := keep(endpoints, ok); len(kept) > 0 {
			return kept
		}
	}
	return endpoints
}

func keep(endpoints []scheduling.Endpoint, ok func(string) bool) []scheduling.Endpoint {
	kept := make([]scheduling.Endpoint, 0, len(endpoints))
	for _, endpoint := range endpoints {
		if ok(endpoint.GetMetadata().ID.String()) {
			kept = append(kept, endpoint)
		}
	}
	return kept
}

// pinned returns the first endpoint, by name, whose pin label lists the request's tenant.
func pinned(request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) string {
	if request == nil {
		return ""
	}
	tenant, _ := metadata.GetLowerCaseHeaderValue(request.Headers, metadata.FlowFairnessIDKey)
	if tenant = strings.TrimSpace(tenant); tenant == "" {
		return ""
	}
	best := ""
	for _, endpoint := range endpoints {
		meta := endpoint.GetMetadata()
		for _, value := range strings.Split(meta.Labels[pinLabel], ",") {
			if name := meta.ID.String(); strings.TrimSpace(value) == tenant && (best == "" || name < best) {
				best = name
			}
		}
	}
	return best
}

// weight is the endpoint's ready pods, or defaultWeight when not reported.
func weight(endpoint scheduling.Endpoint) float64 {
	ready, ok := attrmetrics.ReadScalarMetricValue(endpoint, attrmetrics.MultiClusterReadyEndpointsDataKey)
	if !ok || !finiteNonNegative(float64(ready)) {
		return defaultWeight
	}
	return float64(ready)
}

func finiteNonNegative(v float64) bool {
	return v >= 0 && metadata.IsFiniteFloat(v)
}
