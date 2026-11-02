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

package endpointhealth

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"strconv"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	sourcemetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/metrics"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

const (
	// EndpointHealthFilterType is the type of the EndpointHealthFilter.
	EndpointHealthFilterType = "endpoint-health-filter"

	defaultPollFailureThreshold     = 3
	defaultPollFailureDuration      = 10 * time.Second
	defaultResponseFailureThreshold = 5
	defaultEjectionDuration         = 30 * time.Second
	defaultMaxEjectionDuration      = 5 * time.Minute

	// pruneInterval is how often the state of endpoints that left the pool is dropped.
	pruneInterval = time.Minute
)

// defaultResponseFailureCodes are the statuses a proxy in front of the model
// server returns when it cannot reach it. 500 and 504 are excluded: a model
// server returns 500 for errors specific to one request, and a proxy returns
// 504 when a slow but working model server exceeds its timeout.
var defaultResponseFailureCodes = []int{502, 503}

// parameters defines the parameters for EndpointHealthFilter.
type parameters struct {
	// PollSource is the name of the polling data source whose poll outcomes
	// decide poll health. Defaults to the default metrics data source.
	PollSource string `json:"pollSource,omitempty"`

	// PollFailureThreshold is the number of consecutive failed polls after
	// which an endpoint is marked unhealthy, provided they span at least
	// PollFailureDuration. Zero disables poll health. Defaults to 3.
	PollFailureThreshold *int `json:"pollFailureThreshold,omitempty"`

	// PollFailureDuration is how long polls must keep failing before an
	// endpoint is marked unhealthy. Defaults to 10s.
	PollFailureDuration *metav1.Duration `json:"pollFailureDuration,omitempty"`

	// ResponseFailureThreshold is the number of consecutive failed responses
	// after which an endpoint is ejected. Zero disables response health.
	// Defaults to 5.
	ResponseFailureThreshold *int `json:"responseFailureThreshold,omitempty"`

	// ResponseFailureCodes are the response statuses counted as failures.
	// Defaults to 502 and 503.
	ResponseFailureCodes []int `json:"responseFailureCodes,omitempty"`

	// EjectionDuration is how long the first response ejection lasts. Each
	// further ejection without a successful response in between lasts one
	// EjectionDuration longer. Defaults to 30s.
	EjectionDuration *metav1.Duration `json:"ejectionDuration,omitempty"`

	// MaxEjectionDuration caps the length of a response ejection.
	// Defaults to 5m.
	MaxEjectionDuration *metav1.Duration `json:"maxEjectionDuration,omitempty"`
}

// config is the validated, defaulted form of parameters.
type config struct {
	pollSource               string
	pollFailureThreshold     int
	pollFailureDuration      time.Duration
	responseFailureThreshold int
	responseFailureCodes     map[int]struct{}
	ejectionDuration         time.Duration
	maxEjectionDuration      time.Duration
}

func (p *parameters) applyDefaults() {
	if p.PollSource == "" {
		p.PollSource = sourcemetrics.MetricsDataSourceType
	}
	if p.PollFailureThreshold == nil {
		p.PollFailureThreshold = ptr.To(defaultPollFailureThreshold)
	}
	if p.PollFailureDuration == nil {
		p.PollFailureDuration = &metav1.Duration{Duration: defaultPollFailureDuration}
	}
	if p.ResponseFailureThreshold == nil {
		p.ResponseFailureThreshold = ptr.To(defaultResponseFailureThreshold)
	}
	if p.ResponseFailureCodes == nil {
		p.ResponseFailureCodes = defaultResponseFailureCodes
	}
	if p.EjectionDuration == nil {
		p.EjectionDuration = &metav1.Duration{Duration: defaultEjectionDuration}
	}
	if p.MaxEjectionDuration == nil {
		p.MaxEjectionDuration = &metav1.Duration{Duration: defaultMaxEjectionDuration}
	}
}

// toConfig validates the defaulted parameters and converts them to a config.
func (p *parameters) toConfig() (config, error) {
	var errs []error
	if *p.PollFailureThreshold < 0 {
		errs = append(errs, fmt.Errorf("pollFailureThreshold must be non-negative, got %d", *p.PollFailureThreshold))
	}
	if p.PollFailureDuration.Duration < 0 {
		errs = append(errs, fmt.Errorf("pollFailureDuration must be non-negative, got %v", p.PollFailureDuration.Duration))
	}
	if *p.ResponseFailureThreshold < 0 {
		errs = append(errs, fmt.Errorf("responseFailureThreshold must be non-negative, got %d", *p.ResponseFailureThreshold))
	}
	if *p.PollFailureThreshold == 0 && *p.ResponseFailureThreshold == 0 {
		errs = append(errs, errors.New("pollFailureThreshold and responseFailureThreshold are both zero, so the filter would keep every endpoint"))
	}
	codes := make(map[int]struct{}, len(p.ResponseFailureCodes))
	for _, code := range p.ResponseFailureCodes {
		if code < 500 || code > 599 {
			errs = append(errs, fmt.Errorf("responseFailureCodes must be 5xx statuses, got %d", code))
		}
		codes[code] = struct{}{}
	}
	if *p.ResponseFailureThreshold > 0 {
		if len(codes) == 0 {
			errs = append(errs, errors.New("responseFailureCodes must not be empty when responseFailureThreshold is positive"))
		}
		if p.EjectionDuration.Duration <= 0 {
			errs = append(errs, fmt.Errorf("ejectionDuration must be positive, got %v", p.EjectionDuration.Duration))
		}
		if p.MaxEjectionDuration.Duration < p.EjectionDuration.Duration {
			errs = append(errs, fmt.Errorf("maxEjectionDuration (%v) must be at least ejectionDuration (%v)",
				p.MaxEjectionDuration.Duration, p.EjectionDuration.Duration))
		}
	}
	if len(errs) > 0 {
		return config{}, errors.Join(errs...)
	}
	return config{
		pollSource:               p.PollSource,
		pollFailureThreshold:     *p.PollFailureThreshold,
		pollFailureDuration:      p.PollFailureDuration.Duration,
		responseFailureThreshold: *p.ResponseFailureThreshold,
		responseFailureCodes:     codes,
		ejectionDuration:         p.EjectionDuration.Duration,
		maxEjectionDuration:      p.MaxEjectionDuration.Duration,
	}, nil
}

// compile-time type assertions
var _ fwksched.Filter = &EndpointHealthFilter{}
var _ fwkrc.ResponseHeaderProcessor = &EndpointHealthFilter{}
var _ fwkplugin.ConsumerPlugin = &EndpointHealthFilter{}

// Factory defines the factory function for EndpointHealthFilter.
func Factory(name string, rawParameters *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params := parameters{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' filter - %w", EndpointHealthFilterType, err)
		}
	}
	filter, err := NewEndpointHealthFilter(name, params)
	if err != nil {
		return nil, err
	}
	if handle != nil {
		if err := registerMetrics(handle.Metrics()); err != nil {
			return nil, err
		}
		go filter.pruneLoop(handle.Context(), handle)
	}
	return filter, nil
}

// NewEndpointHealthFilter validates the given parameters and returns a new
// EndpointHealthFilter with the given name.
func NewEndpointHealthFilter(name string, params parameters) (*EndpointHealthFilter, error) {
	if name == "" {
		name = EndpointHealthFilterType
	}
	params.applyDefaults()
	cfg, err := params.toConfig()
	if err != nil {
		return nil, fmt.Errorf("invalid parameters of the '%s' filter - %w", EndpointHealthFilterType, err)
	}
	return &EndpointHealthFilter{
		typedName:     fwkplugin.TypedName{Type: EndpointHealthFilterType, Name: name},
		pollStatusKey: fwkdl.PollStatusDataKey(cfg.pollSource),
		tracker:       newTracker(cfg, name),
		now:           time.Now,
	}, nil
}

// EndpointHealthFilter drops endpoints that are failing. An endpoint is marked
// unhealthy when its metric polls keep failing, or ejected for a while when it
// returns consecutive failed responses. When every candidate is unhealthy the
// filter keeps all of them.
type EndpointHealthFilter struct {
	typedName     fwkplugin.TypedName
	pollStatusKey fwkplugin.DataKey
	tracker       *tracker
	now           func() time.Time
}

// TypedName returns the typed name of the plugin.
func (f *EndpointHealthFilter) TypedName() fwkplugin.TypedName {
	return f.typedName
}

// Consumes declares the poll status of the configured source when poll health
// is enabled, so a pollSource that names no configured source fails at startup.
func (f *EndpointHealthFilter) Consumes() fwkplugin.DataDependencies {
	if f.tracker.cfg.pollFailureThreshold == 0 {
		return fwkplugin.DataDependencies{}
	}
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{f.pollStatusKey: &fwkdl.PollStatus{}},
	}
}

// Filter keeps the endpoints that are not unhealthy, or every endpoint when
// none is healthy.
func (f *EndpointHealthFilter) Filter(ctx context.Context, _ *fwksched.InferenceRequest,
	endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	logger := log.FromContext(ctx)
	now := f.now()
	pollHealth := f.tracker.cfg.pollFailureThreshold > 0
	healthy := make([]fwksched.Endpoint, 0, len(endpoints))
	for _, endpoint := range endpoints {
		var status *fwkdl.PollStatus
		if pollHealth {
			status, _ = fwkdl.ReadAttribute[*fwkdl.PollStatus](endpoint, f.pollStatusKey)
		}
		md := endpoint.GetMetadata()
		if !f.tracker.evaluate(logger, md.ID, hostPort(md), status, now) {
			healthy = append(healthy, endpoint)
		}
	}

	if len(healthy) == 0 && len(endpoints) > 0 {
		failOpens.WithLabelValues(f.typedName.Name).Inc()
		logger.V(logutil.DEBUG).Info("Every candidate endpoint is unhealthy, keeping all", "candidates", len(endpoints))
		return endpoints
	}
	return healthy
}

// ResponseHeader counts the response status toward the serving endpoint's
// response health.
func (f *EndpointHealthFilter) ResponseHeader(ctx context.Context, _ *fwksched.InferenceRequest,
	response *fwkrc.Response, targetEndpoint *fwkdl.EndpointMetadata) {
	if f.tracker.cfg.responseFailureThreshold == 0 || response == nil || targetEndpoint == nil {
		return
	}
	address := hostPort(targetEndpoint)
	// The proxy can serve a request from a later endpoint of those picked, and
	// that response says nothing about the target.
	if served, ok := servedEndpoint(response.ReqMetadata); ok && served != address {
		return
	}
	code, ok := responseStatus(response.Headers)
	if !ok {
		return
	}
	_, failed := f.tracker.cfg.responseFailureCodes[code]
	f.tracker.observeResponse(log.FromContext(ctx), targetEndpoint.ID, address, failed, f.now())
}

// hostPort returns the endpoint's address in the form the proxy reports the
// endpoint that served a request.
func hostPort(md *fwkdl.EndpointMetadata) string {
	return net.JoinHostPort(md.GetIPAddress(), md.GetPort())
}

// servedEndpoint returns the address of the endpoint the proxy reports it sent
// the request to.
func servedEndpoint(reqMetadata map[string]any) (string, bool) {
	lb, ok := reqMetadata[metadata.DestinationEndpointNamespace].(map[string]any)
	if !ok {
		return "", false
	}
	served, ok := lb[metadata.DestinationEndpointServedKey].(string)
	return served, ok
}

func responseStatus(headers map[string]string) (int, bool) {
	raw, ok := headers[reqcommon.PseudoHeaderStatus]
	if !ok {
		raw, ok = headers[reqcommon.HeaderStatus]
	}
	if !ok {
		return 0, false
	}
	code, err := strconv.Atoi(raw)
	return code, err == nil
}

// pruneLoop periodically drops the state of endpoints that left the pool.
func (f *EndpointHealthFilter) pruneLoop(ctx context.Context, handle fwkplugin.Handle) {
	ticker := time.NewTicker(pruneInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if ids := handle.PodList(); ids != nil {
				f.tracker.prune(ids)
			}
		}
	}
}
