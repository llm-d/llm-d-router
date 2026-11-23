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

// Package modelhealth observes inference output and checks inactive endpoints.
package modelhealth

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrmodels "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/models"
	extmodels "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/models"
	srcmodels "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/models"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// Config configures checks for each aggregated endpoint's discovered base model.
type Config struct {
	InactivityInterval  metav1.Duration `json:"inactivityInterval,omitempty"`
	RequestTimeout      metav1.Duration `json:"requestTimeout,omitempty"`
	MaxObservationAge   metav1.Duration `json:"maxObservationAge,omitempty"`
	MaxConcurrentProbes int             `json:"maxConcurrentProbes,omitempty"`
	Scheme              string          `json:"scheme,omitempty"`
}

type endpointState struct {
	endpoint             fwkdl.Endpoint
	metadata             *fwkdl.EndpointMetadata
	cancel               context.CancelFunc
	ctx                  context.Context
	busy                 bool
	discoveredAt         time.Time
	nextProbe            time.Time
	revision             uint64
	retryInterval        time.Duration
	failedProbeStartedAt time.Time
	health               attrmodels.InferenceHealth
}

// Monitor owns inference-health judgments for the discovered endpoints.
type Monitor struct {
	name           fwkplugin.TypedName
	key            fwkplugin.DataKey
	observationKey fwkplugin.DataKey
	cfg            Config
	ctx            context.Context
	client         *http.Client
	active         int
	mu             sync.Mutex
	endpoints      map[string]*endpointState
}

// Factory creates an opt-in health monitor. Endpoint notifications start checks.
func Factory(name string, decoder *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	if handle == nil || handle.Context() == nil {
		return nil, errors.New("plugin handle and context are required")
	}
	cfg := Config{InactivityInterval: metav1.Duration{Duration: 10 * time.Second}, RequestTimeout: metav1.Duration{Duration: 3 * time.Second}, MaxObservationAge: metav1.Duration{Duration: time.Minute}, MaxConcurrentProbes: 4, Scheme: "http"}
	if decoder != nil {
		if err := decoder.Decode(&cfg); err != nil {
			return nil, err
		}
	}
	if cfg.Scheme != "http" && cfg.Scheme != "https" {
		return nil, errors.New("scheme must be http or https")
	}
	if cfg.MaxConcurrentProbes < 1 || cfg.MaxConcurrentProbes > 64 {
		return nil, errors.New("maxConcurrentProbes must be between 1 and 64")
	}
	for _, duration := range []metav1.Duration{cfg.InactivityInterval, cfg.RequestTimeout, cfg.MaxObservationAge} {
		if duration.Duration <= 0 || duration.Duration > 24*time.Hour {
			return nil, errors.New("durations must be positive and at most 24h")
		}
	}
	if cfg.MaxObservationAge.Duration <= cfg.InactivityInterval.Duration*11/10+cfg.RequestTimeout.Duration {
		return nil, errors.New("maxObservationAge must exceed inactivityInterval with jitter plus requestTimeout")
	}

	if err := registerMetrics(handle.Metrics()); err != nil {
		return nil, err
	}
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.Proxy = nil
	p := &Monitor{
		name:           fwkplugin.TypedName{Type: attrmodels.InferenceHealthProducerType, Name: name},
		key:            attrmodels.InferenceHealthDataKey.WithNonEmptyProducerName(name),
		observationKey: fwkplugin.NewDataKey("inference-health-observation", attrmodels.InferenceHealthProducerType).WithNonEmptyProducerName(name),
		cfg:            cfg, ctx: handle.Context(),
		client:    &http.Client{Transport: transport, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }},
		endpoints: make(map[string]*endpointState),
	}
	go p.run()
	return p, nil
}

func (p *Monitor) TypedName() fwkplugin.TypedName { return p.name }
func (p *Monitor) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{p.key: attrmodels.InferenceHealth{}, p.observationKey: &observation{}, fwkrh.ResponseActivityTrackingKey: false}
}
func (p *Monitor) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{Required: map[fwkplugin.DataKey]any{attrconcurrency.InFlightLoadDataKey: attrconcurrency.InFlightLoad{}}, Optional: map[fwkplugin.DataKey]any{attrmodels.ModelsAttributeKey: attrmodels.ModelDataCollection{}}}
}

func (p *Monitor) RegisterDependencies(r fwkdl.Registrar) error {
	if err := r.Register(fwkdl.PendingRegistration{Owner: p.name, SourceType: notifications.EndpointNotificationSourceType, Extractor: p,
		DefaultSource: notifications.NewEndpointDataSource(notifications.EndpointNotificationSourceType, notifications.EndpointNotificationSourceType)}); err != nil {
		return err
	}
	parameters := json.NewDecoder(strings.NewReader(fmt.Sprintf(`{"scheme":%q,"interval":%q,"insecureSkipVerify":false}`, p.cfg.Scheme, p.cfg.InactivityInterval.Duration.String())))
	source, err := srcmodels.ModelDataSourceFactory(srcmodels.ModelsDataSourceType, parameters, nil)
	if err != nil {
		return err
	}
	return r.Register(fwkdl.PendingRegistration{Owner: p.name, SourceType: srcmodels.ModelsDataSourceType, Extractor: extmodels.NewModelExtractor(), DefaultSource: source.(fwkdl.DataSource)})
}

func (p *Monitor) Extract(_ context.Context, event fwkdl.EndpointEvent) error {
	if event.Endpoint == nil || event.Endpoint.GetMetadata() == nil {
		return nil
	}
	md := event.Endpoint.GetMetadata()
	id := md.ID.String()
	p.mu.Lock()
	defer p.mu.Unlock()
	state := p.endpoints[id]
	switch event.Type {
	case fwkdl.EventDelete:
		if state != nil && state.endpoint == event.Endpoint {
			p.removeLocked(state)
		}
	case fwkdl.EventAddOrUpdate:
		if state != nil && (state.endpoint != event.Endpoint || !state.metadata.Equal(md)) {
			p.removeLocked(state)
			state = nil
		}
		// Role-specific and data-parallel checks require separate backend validation.
		if md.Labels[bylabel.RoleLabel] != "" || md.RankIndex != 0 || p.ctx.Err() != nil {
			return nil
		}
		if state == nil {
			ctx, cancel := context.WithCancel(p.ctx)
			state = &endpointState{endpoint: event.Endpoint, metadata: md.Clone(), discoveredAt: time.Now(), cancel: cancel, ctx: ctx,
				nextProbe: time.Now().Add(p.delay(p.cfg.InactivityInterval.Duration)), health: attrmodels.InferenceHealth{Reason: "model-unavailable"}}
			p.endpoints[id] = state
			p.publishLocked(state, time.Now())
		}
		captured := state
		event.Endpoint.GetAttributes().Put(p.key, &fwkdl.DynamicAttribute{Get: func() fwkdl.Cloneable {
			p.mu.Lock()
			defer p.mu.Unlock()
			if p.endpoints[id] != captured {
				return nil
			}
			return p.snapshotLocked(captured, time.Now())
		}})
	}
	return nil
}

func (p *Monitor) removeLocked(state *endpointState) {
	state.cancel()
	delete(p.endpoints, state.metadata.ID.String())
	healthUp.DeleteLabelValues(p.labels(state)...)
}

func (p *Monitor) snapshotLocked(state *endpointState, now time.Time) *attrmodels.InferenceHealth {
	p.refreshModelLocked(state)
	result := state.health
	if result.Known && now.Sub(result.ObservedAt) > p.cfg.MaxObservationAge.Duration {
		result.Known = false
		result.Up = false
		result.Reason = "stale"
	}
	return &result
}

func (p *Monitor) recordActivity(state *endpointState, model string, at time.Time) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.endpoints[state.metadata.ID.String()] != state {
		return
	}
	p.refreshModelLocked(state)
	if model == "" || state.health.Model != model || at.IsZero() {
		return
	}
	// The async response queue can deliver business output after its overlapping probe failed.
	overlappingFailure := !state.health.Up && !state.failedProbeStartedAt.IsZero() && at.After(state.failedProbeStartedAt) && at.After(state.health.LastSuccessAt)
	if !at.After(state.health.ObservedAt) && !overlappingFailure {
		return
	}
	state.revision++
	state.retryInterval = 0
	state.failedProbeStartedAt = time.Time{}
	state.health = attrmodels.InferenceHealth{Model: model, Known: true, Up: true, Reason: "business-output", ObservedAt: at, LastSuccessAt: at}
	state.nextProbe = at.Add(p.delay(p.probeInterval(state)))
	p.publishLocked(state, time.Now())
}

// Missing load is treated conservatively; in-flight requests never disable probes.
func (p *Monitor) probeInterval(state *endpointState) time.Duration {
	load, ok := fwkdl.ReadAttribute[*attrconcurrency.InFlightLoad](state.endpoint.GetAttributes(), attrconcurrency.InFlightLoadDataKey)
	if !ok || load == nil || load.Requests > 0 {
		return max(p.cfg.InactivityInterval.Duration, time.Minute)
	}
	return p.cfg.InactivityInterval.Duration
}

func (p *Monitor) delay(base time.Duration) time.Duration {
	return base + time.Duration(rand.Float64()*float64(base)/10) // #nosec G404 -- scheduling jitter, not a security decision
}

// run shares one timer across endpoints; only admitted probes create goroutines.
func (p *Monitor) run() {
	ticker := time.NewTicker(min(time.Second, max(time.Millisecond, p.cfg.InactivityInterval.Duration/10)))
	defer ticker.Stop()
	defer p.client.CloseIdleConnections()
	for {
		select {
		case <-p.ctx.Done():
			p.mu.Lock()
			for _, state := range p.endpoints {
				p.removeLocked(state)
			}
			p.mu.Unlock()
			return
		case now := <-ticker.C:
			p.mu.Lock()
			for _, state := range p.endpoints {
				p.publishLocked(state, now)
				interval := p.probeInterval(state)
				since := state.health.ObservedAt
				if since.IsZero() {
					since = state.discoveredAt
				}
				due := state.nextProbe
				if earliest := since.Add(interval); earliest.After(due) {
					due = earliest
				}
				if state.busy || state.health.Model == "" || now.Before(due) {
					continue
				}
				if p.active < p.cfg.MaxConcurrentProbes {
					p.active++
					state.busy = true
					go p.check(state, state.revision, state.health.Model)
				}
			}
			p.mu.Unlock()
		}
	}
}

func (p *Monitor) check(state *endpointState, revision uint64, model string) {
	started := time.Now()
	reason := p.probe(state.ctx, state.metadata, model)
	at := time.Now()
	p.mu.Lock()
	defer p.mu.Unlock()
	p.active--
	state.busy = false
	if p.endpoints[state.metadata.ID.String()] != state || p.ctx.Err() != nil {
		return
	}
	p.refreshModelLocked(state)
	outcome := "error"
	if reason == "success" {
		outcome = "success"
	}
	probeResults.WithLabelValues(p.name.Name, outcome, reason).Inc()
	if state.revision != revision {
		return
	}
	state.revision++
	state.health.Known = true
	state.health.Up = reason == "success"
	state.health.Reason = reason
	state.health.ObservedAt = at
	interval := p.probeInterval(state)
	if state.health.Up {
		state.health.LastSuccessAt = at
		state.failedProbeStartedAt = time.Time{}
		state.retryInterval = 0
	} else {
		state.failedProbeStartedAt = started
		state.retryInterval = min(max(state.retryInterval, interval)*2, max(interval, 60*time.Second))
		interval = state.retryInterval
	}
	delay := p.delay(interval)
	if !state.health.Up {
		delay = min(delay, max(p.probeInterval(state), 60*time.Second))
	}
	state.nextProbe = at.Add(delay)
	p.publishLocked(state, at)
}

// refreshModelLocked invalidates evidence when discovery no longer identifies one base model.
func (p *Monitor) refreshModelLocked(state *endpointState) {
	models, _ := fwkdl.ReadAttribute[attrmodels.ModelDataCollection](state.endpoint.GetAttributes(), attrmodels.ModelsAttributeKey)
	model := ""
	for _, entry := range models {
		if entry.Parent != "" {
			continue
		}
		if entry.ID == "" || len(entry.ID) > 256 || strings.TrimSpace(entry.ID) != entry.ID || (model != "" && model != entry.ID) {
			model = ""
			break
		}
		model = entry.ID
	}
	if state.health.Model == model {
		return
	}
	healthUp.DeleteLabelValues(p.labels(state)...)
	state.revision++
	state.retryInterval = 0
	state.failedProbeStartedAt = time.Time{}
	reason := "unobserved"
	if model == "" {
		reason = "model-unavailable"
	}
	state.health = attrmodels.InferenceHealth{Model: model, Reason: reason}
	state.discoveredAt = time.Now()
	state.nextProbe = state.discoveredAt.Add(p.delay(p.cfg.InactivityInterval.Duration))
}

type observation struct {
	state      *endpointState
	model      string
	attributed bool
}

func (p *Monitor) PreRequest(_ context.Context, req *fwksched.InferenceRequest, result *fwksched.SchedulingResult) error {
	if req == nil || req.Body == nil || req.Body.ChatCompletions == nil || req.TargetModel == "" || result == nil || len(result.ProfileResults) != 1 {
		return nil
	}
	profile := result.ProfileResults[result.PrimaryProfileName]
	if profile == nil || len(profile.TargetEndpoints) != 1 || profile.TargetEndpoints[0] == nil {
		return nil
	}
	md := profile.TargetEndpoints[0].GetMetadata()
	if md == nil {
		return nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	state := p.endpoints[md.ID.String()]
	if state == nil || !state.metadata.Equal(md) {
		return nil
	}
	p.refreshModelLocked(state)
	if req.TargetModel != state.health.Model {
		return nil
	}
	req.PutAttribute(p.observationKey, &observation{state: state, model: req.TargetModel})
	req.PutAttribute(fwkrh.ResponseActivityTrackingKey, true)
	return nil
}

func (p *Monitor) observation(req *fwksched.InferenceRequest) *observation {
	if req == nil {
		return nil
	}
	raw, _ := req.GetAttribute(p.observationKey)
	obs, _ := raw.(*observation)
	return obs
}

func (p *Monitor) ResponseHeader(_ context.Context, req *fwksched.InferenceRequest, resp *fwkrh.Response, target *fwkdl.EndpointMetadata) {
	obs := p.observation(req)
	if obs == nil || resp == nil || target == nil || !obs.state.metadata.Equal(target) {
		return
	}
	statusValue := resp.Headers[reqcommon.PseudoHeaderStatus]
	if statusValue == "" {
		statusValue = resp.Headers[reqcommon.HeaderStatus]
	}
	status, err := strconv.Atoi(statusValue)
	if err != nil || status < 200 || status >= 300 {
		return
	}
	lb, _ := resp.ReqMetadata[metadata.DestinationEndpointNamespace].(map[string]any)
	served, _ := lb[metadata.DestinationEndpointServedKey].(string)
	obs.attributed = served == net.JoinHostPort(target.Address, target.Port)
}

func (p *Monitor) ResponseBody(_ context.Context, req *fwksched.InferenceRequest, resp *fwkrh.Response, target *fwkdl.EndpointMetadata) {
	obs := p.observation(req)
	if obs == nil || !obs.attributed || resp == nil || resp.InferenceActivity == nil || target == nil ||
		!obs.state.metadata.Equal(target) || resp.InferenceActivity.Model != obs.model {
		return
	}
	if resp.EndOfStream && resp.TerminationCause != fwkrh.TerminationCauseNatural {
		return
	}
	p.recordActivity(obs.state, obs.model, resp.InferenceActivity.ObservedAt)
}

func (p *Monitor) probe(parent context.Context, md *fwkdl.EndpointMetadata, model string) string {
	ctx, cancel := context.WithTimeout(parent, p.cfg.RequestTimeout.Duration)
	defer cancel()
	payload, _ := json.Marshal(map[string]any{
		reqcommon.FieldModel:     model,
		reqcommon.FieldMessages:  []map[string]string{{"role": "user", "content": "Hi"}},
		reqcommon.FieldMaxTokens: 1, reqcommon.FieldStream: false,
		"ignore_eos": true, "temperature": 0,
	})
	target := url.URL{Scheme: p.cfg.Scheme, Host: net.JoinHostPort(md.Address, md.Port), Path: reqcommon.PathChatCompletions}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, target.String(), bytes.NewReader(payload))
	if err != nil {
		return "invalid-request"
	}
	request.Header.Set(reqcommon.HeaderContentType, reqcommon.ContentTypeJSON)
	response, err := p.client.Do(request)
	if err != nil {
		return probeError(ctx)
	}
	defer response.Body.Close()
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return "http-error"
	}
	body, err := io.ReadAll(io.LimitReader(response.Body, fwkrh.MaxActivityResponseBytes+1))
	if err != nil {
		return probeError(ctx)
	}
	if len(body) > fwkrh.MaxActivityResponseBytes {
		return "response-too-large"
	}
	tracker := (&openai.OpenAIParser{}).NewResponseActivityTracker()
	// A non-streaming probe must complete, not merely produce a first SSE chunk.
	headers := map[string]string{reqcommon.HeaderContentType: reqcommon.ContentTypeJSON}
	if tracker.Observe(body, headers, true) != model {
		return "invalid-response"
	}
	if ctx.Err() != nil {
		return probeError(ctx)
	}
	return "success"
}

func probeError(ctx context.Context) string {
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return "timeout"
	}
	if ctx.Err() != nil {
		return "canceled"
	}
	return "transport-error"
}

var (
	healthUp = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:      "model_inference_up",
		Help:      metricsutil.HelpMsgWithStability("Whether the latest valid model inference check passed. Unknown or expired results have no series.", compbasemetrics.ALPHA),
	}, []string{"endpoint_name", "namespace", "port", "producer_name", "target_model_name"})
	probeResults = prometheus.NewCounterVec(prometheus.CounterOpts{
		Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:      "model_inference_probes_total",
		Help:      metricsutil.HelpMsgWithStability("Total model inference probes by outcome and reason.", compbasemetrics.ALPHA),
	}, []string{"producer_name", "outcome", "reason"})
)

func registerMetrics(registerer prometheus.Registerer) error {
	if registerer == nil {
		return errors.New("health metrics registerer is required")
	}
	for _, collector := range []prometheus.Collector{healthUp, probeResults} {
		if err := registerer.Register(collector); err != nil {
			var registered prometheus.AlreadyRegisteredError
			if errors.As(err, &registered) && registered.ExistingCollector == collector {
				continue
			}
			return fmt.Errorf("register model health metric: %w", err)
		}
	}
	return nil
}
func (p *Monitor) labels(state *endpointState) []string {
	return []string{state.metadata.ID.Name, state.metadata.ID.Namespace, state.metadata.Port, p.name.Name, eppmetrics.BoundModelName(state.health.Model)}
}
func (p *Monitor) publishLocked(state *endpointState, at time.Time) {
	result := p.snapshotLocked(state, at)
	if !result.Known {
		healthUp.DeleteLabelValues(p.labels(state)...)
		return
	}
	up := 0.0
	if result.Up {
		up = 1
	}
	healthUp.WithLabelValues(p.labels(state)...).Set(up)
}
