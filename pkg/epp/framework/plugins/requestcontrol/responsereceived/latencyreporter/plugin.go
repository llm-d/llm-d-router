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

// Package latencyreporter provides a response plugin that records each completed request's
// outcome, time to first token and time per output token, labeled by priority, prompt-length
// bucket and prefill/decode path.
package latencyreporter

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const (
	// PluginType is the registered plugin type.
	PluginType = "latency-reporter"

	defaultPrefillProfile = "prefill"

	pathRemote     = "remote"
	pathLocal      = "local"
	pathAggregated = "aggregated"

	outcomeSuccess = "success"
	outcomeFailed  = "failed"
)

var (
	_ requestcontrol.ResponseHeaderProcessor = &Plugin{}
	_ requestcontrol.ResponseBodyProcessor   = &Plugin{}
)

// Parameters is the plugin configuration.
type Parameters struct {
	// PromptBuckets are increasing upper bounds, in prompt tokens. A request is labeled
	// "le<bound>" for the first bound it does not exceed, else "gt<last bound>". With no bounds
	// every request is labeled "all".
	PromptBuckets []int `json:"promptBuckets,omitempty"`
	// PrefillProfileName is the scheduling profile whose result marks a remote prefill.
	PrefillProfileName string `json:"prefillProfileName,omitempty"`
	// Priorities limits recording to these served priorities. Empty records every priority.
	Priorities []int `json:"priorities,omitempty"`
}

// Plugin records per-request outcome and latency. See the package README.
type Plugin struct {
	typedName      plugin.TypedName
	buckets        []int
	bucketLabels   []string
	prefillProfile string
	priorities     map[int]struct{} // nil records every priority

	statuses sync.Map // key: RequestID, value: HTTP status string
}

// PluginFactory builds the plugin from its parameters.
func PluginFactory(name string, rawParameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	params := Parameters{PrefillProfileName: defaultPrefillProfile}
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' plugin - %w", PluginType, err)
		}
	}
	p, err := newPlugin(name, params)
	if err != nil {
		return nil, err
	}
	if handle != nil {
		if err := registerMetrics(handle.Metrics()); err != nil {
			return nil, err
		}
	}
	return p, nil
}

func newPlugin(name string, params Parameters) (*Plugin, error) {
	if !slices.IsSorted(params.PromptBuckets) || len(slices.Compact(slices.Clone(params.PromptBuckets))) != len(params.PromptBuckets) {
		return nil, fmt.Errorf("%s plugin %q: promptBuckets must strictly increase", PluginType, name)
	}
	if params.PrefillProfileName == "" {
		params.PrefillProfileName = defaultPrefillProfile
	}
	labels := make([]string, 0, len(params.PromptBuckets)+1)
	for _, b := range params.PromptBuckets {
		if b <= 0 {
			return nil, fmt.Errorf("%s plugin %q: promptBuckets must be > 0", PluginType, name)
		}
		labels = append(labels, "le"+strconv.Itoa(b))
	}
	if n := len(params.PromptBuckets); n > 0 {
		labels = append(labels, "gt"+strconv.Itoa(params.PromptBuckets[n-1]))
	} else {
		labels = append(labels, "all")
	}
	var priorities map[int]struct{}
	if len(params.Priorities) > 0 {
		priorities = make(map[int]struct{}, len(params.Priorities))
		for _, prio := range params.Priorities {
			priorities[prio] = struct{}{}
		}
	}
	return &Plugin{
		typedName:      plugin.TypedName{Type: PluginType, Name: name},
		buckets:        params.PromptBuckets,
		bucketLabels:   labels,
		prefillProfile: params.PrefillProfileName,
		priorities:     priorities,
	}, nil
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *Plugin) TypedName() plugin.TypedName { return p.typedName }

// ResponseHeader records the response status for use at end of stream.
func (p *Plugin) ResponseHeader(_ context.Context, request *scheduling.InferenceRequest, response *requestcontrol.Response, _ *datalayer.EndpointMetadata) {
	if request == nil || response == nil {
		return
	}
	status := response.Headers[":status"]
	if status == "" {
		status = response.Headers["status"]
	}
	p.statuses.Store(request.RequestID, status)
}

// ResponseBody records the request's outcome and latency at end of stream.
func (p *Plugin) ResponseBody(_ context.Context, request *scheduling.InferenceRequest, response *requestcontrol.Response, _ *datalayer.EndpointMetadata) {
	if request == nil || response == nil || !response.EndOfStream {
		return
	}
	v, _ := p.statuses.LoadAndDelete(request.RequestID)
	status, _ := v.(string)

	priority := request.Objectives.Priority
	if p.priorities != nil {
		if _, ok := p.priorities[priority]; !ok {
			return
		}
	}
	bucket := p.bucket(promptTokens(request, response))
	path := p.path(request)
	prio := strconv.Itoa(priority)
	fairness := metricsutil.BoundFairnessID(request.FairnessID)

	success := (status == "" || status == "200") && response.TerminationCause == requestcontrol.TerminationCauseNatural
	if !success {
		latencyRequests.WithLabelValues(request.TargetModel, fairness, prio, bucket, path, outcomeFailed).Inc()
		return
	}
	if !response.FirstTokenTimestamp.IsZero() && !response.ReceivedTimestamp.IsZero() {
		latencyTTFT.WithLabelValues(prio, bucket, path).Observe(response.FirstTokenTimestamp.Sub(response.ReceivedTimestamp).Seconds())
	}
	if tpot, ok := meanTPOT(response); ok {
		latencyTPOT.WithLabelValues(prio, path).Observe(tpot.Seconds())
	}
	latencyRequests.WithLabelValues(request.TargetModel, fairness, prio, bucket, path, outcomeSuccess).Inc()
}

// meanTPOT is the per-request mean gap between output tokens. It needs a streamed response with
// at least two output tokens.
func meanTPOT(response *requestcontrol.Response) (time.Duration, bool) {
	n := response.Usage.CompletionTokens
	if !response.Streaming || n < 2 || response.FirstTokenTimestamp.IsZero() || response.CompleteTimestamp.IsZero() {
		return 0, false
	}
	return response.CompleteTimestamp.Sub(response.FirstTokenTimestamp) / time.Duration(n-1), true
}

func promptTokens(request *scheduling.InferenceRequest, response *requestcontrol.Response) int {
	if response.Usage.PromptTokens > 0 {
		return response.Usage.PromptTokens
	}
	if request.Body != nil && request.Body.TokenizedRequest != nil {
		return request.Body.TokenizedRequest.TokenCount()
	}
	return 0
}

func (p *Plugin) bucket(tokens int) string {
	for i, b := range p.buckets {
		if tokens <= b {
			return p.bucketLabels[i]
		}
	}
	return p.bucketLabels[len(p.bucketLabels)-1]
}

func (p *Plugin) path(request *scheduling.InferenceRequest) string {
	if request.SchedulingResult == nil {
		return pathAggregated
	}
	res, ok := request.SchedulingResult.ProfileResults[p.prefillProfile]
	switch {
	case !ok:
		return pathAggregated
	case res != nil && len(res.TargetEndpoints) > 0:
		return pathRemote
	default:
		return pathLocal
	}
}

func registerMetrics(registerer prometheus.Registerer) error {
	if registerer == nil {
		return nil
	}
	for _, collector := range []prometheus.Collector{latencyRequests, latencyTTFT, latencyTPOT} {
		if err := registerer.Register(collector); err != nil {
			var alreadyRegistered prometheus.AlreadyRegisteredError
			if errors.As(err, &alreadyRegistered) && alreadyRegistered.ExistingCollector == collector {
				continue
			}
			return fmt.Errorf("register latency reporter metric: %w", err)
		}
	}
	return nil
}
