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

package inflightload

import (
	"context"
	"net"
	"slices"
	"strings"
	"sync"

	"github.com/prometheus/client_golang/prometheus"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
)

const (
	// sessionIDHeader is the request header vLLM reads for the session id when
	// the body carries none. The engine echoes the id on the KV events for the
	// blocks the request stores.
	sessionIDHeader = "x-session-id"
	// sessionIDField and extraArgsField are the body locations vLLM reads for
	// the session id ahead of the header.
	sessionIDField = "session_id"
	extraArgsField = "vllm_xargs"

	signalKVEvent   = "kv_event"
	signalFirstByte = "first_byte"
)

var _ kvevents.EventConsumer = (*InFlightLoadProducer)(nil)

// pendingPrefill is a dispatched request whose prefill completion the producer
// awaits from the KV events of the engines that decode it. It references the
// request's prompt token slices and scheduling result and nothing else of the
// request.
type pendingPrefill struct {
	requestID string
	result    *fwksched.SchedulingResult
	// endpoints are the event endpoints (Address:Port) whose store events
	// complete the request: the decode engines' in disaggregated serving, the
	// prefill engine's own when the request has no other target.
	endpoints []string
	stamp     string

	mu      sync.Mutex
	prompts [][]uint32
	done    []bool
}

// markStored records a store event for the request and reports whether every
// prompt with at least one full block has had its last full block stored.
// Prompts shorter than the engine block store no prompt block; a request made
// only of such prompts never completes from events, since the blocks the
// engine then stores hold generated tokens.
func (p *pendingPrefill) markStored(tokens []uint32, blockSize int) bool {
	if blockSize <= 0 || len(tokens) < blockSize {
		return false
	}
	tail := tokens[len(tokens)-blockSize:]
	p.mu.Lock()
	defer p.mu.Unlock()
	complete := false
	for i, prompt := range p.prompts {
		if len(prompt) < blockSize {
			continue
		}
		if !p.done[i] {
			end := len(prompt) / blockSize * blockSize
			p.done[i] = slices.Equal(prompt[end-blockSize:end], tail)
		}
		if !p.done[i] {
			return false
		}
		complete = true
	}
	return complete
}

// prefillWatch indexes pending requests by event endpoint and stamp, and by
// request id for release. pending reports the number of watched requests.
type prefillWatch struct {
	mu         sync.Mutex
	byEndpoint map[string]map[string]*pendingPrefill
	byRequest  map[string]*pendingPrefill
	pending    prometheus.Gauge
}

func newPrefillWatch(pending prometheus.Gauge) *prefillWatch {
	return &prefillWatch{
		byEndpoint: map[string]map[string]*pendingPrefill{},
		byRequest:  map[string]*pendingPrefill{},
		pending:    pending,
	}
}

func (w *prefillWatch) add(pending *pendingPrefill) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.byRequest[pending.requestID] = pending
	for _, endpoint := range pending.endpoints {
		stamps := w.byEndpoint[endpoint]
		if stamps == nil {
			stamps = map[string]*pendingPrefill{}
			w.byEndpoint[endpoint] = stamps
		}
		stamps[pending.stamp] = pending
	}
	w.pending.Set(float64(len(w.byRequest)))
}

func (w *prefillWatch) lookup(endpoint, stamp string) *pendingPrefill {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.byEndpoint[endpoint][stamp]
}

// remove drops the request's watch and reports whether one was pending.
func (w *prefillWatch) remove(requestID string) bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	pending, ok := w.byRequest[requestID]
	if !ok {
		return false
	}
	delete(w.byRequest, requestID)
	for _, endpoint := range pending.endpoints {
		stamps := w.byEndpoint[endpoint]
		if stamps[pending.stamp] == pending {
			delete(stamps, pending.stamp)
		}
		if len(stamps) == 0 {
			delete(w.byEndpoint, endpoint)
		}
	}
	w.pending.Set(float64(len(w.byRequest)))
	return true
}

// reset stops matching events from an endpoint whose stream lost continuity.
// The requests stay watched, so a store event from another of their endpoints
// or the first response byte still completes them.
func (w *prefillWatch) reset(endpoint string) int {
	w.mu.Lock()
	defer w.mu.Unlock()
	dropped := len(w.byEndpoint[endpoint])
	delete(w.byEndpoint, endpoint)
	return dropped
}

// sessionStamp returns the session id the engine will echo for the request,
// following vLLM's precedence: body session_id, then the session header,
// then vllm_xargs.session_id. With none present it sets the header to the
// request id. ok is false when the request carries no headers to stamp.
func sessionStamp(request *fwksched.InferenceRequest) (stamp string, ok bool) {
	var payload fwkrequest.PayloadMap
	if request.Body != nil {
		payload, _ = request.Body.Payload.(fwkrequest.PayloadMap)
	}
	if s, ok := payload[sessionIDField].(string); ok && s != "" {
		return s, true
	}
	for key, value := range request.Headers {
		if value != "" && strings.EqualFold(key, sessionIDHeader) {
			return value, true
		}
	}
	switch xargs := payload[extraArgsField].(type) {
	case map[string]any:
		if s, ok := xargs[sessionIDField].(string); ok && s != "" {
			return s, true
		}
	case fwkrequest.PayloadMap:
		if s, ok := xargs[sessionIDField].(string); ok && s != "" {
			return s, true
		}
	}
	if request.Headers == nil {
		return "", false
	}
	request.Headers[sessionIDHeader] = request.RequestID
	return request.RequestID, true
}

// eventEndpoint is the endpoint identity under which the KV events source
// delivers an engine's events.
func eventEndpoint(endpoint fwksched.Endpoint) string {
	meta := endpoint.GetMetadata()
	if meta.Address == "" {
		return ""
	}
	return net.JoinHostPort(meta.Address, meta.Port)
}

// watchPrefill registers the dispatched request for completion from KV
// events. targets maps each profile name to the request's endpoint for that
// profile. The engines that decode the request complete it: every profile's
// endpoint other than the prefill profile's, or the prefill endpoint when the
// request has no other. The prefill engine holds the request's blocks until
// the decode engine has pulled them, so the decode engine's store event marks
// the end of the request's prefill footprint; a single engine serving both
// phases stores the blocks as it computes them.
func (p *InFlightLoadProducer) watchPrefill(ctx context.Context, request *fwksched.InferenceRequest,
	result *fwksched.SchedulingResult, stamp string, targets map[string]fwksched.Endpoint,
) {
	tokenized := request.Body.TokenizedRequest
	if tokenized == nil {
		return
	}
	pending := &pendingPrefill{
		requestID: request.RequestID,
		result:    result,
		stamp:     stamp,
		done:      make([]bool, len(tokenized.Prompts)),
	}
	for _, prompt := range tokenized.Prompts {
		pending.prompts = append(pending.prompts, prompt.TokenIDs)
	}
	for profileName, endpoint := range targets {
		if profileName == profilePrefill {
			continue
		}
		if addr := eventEndpoint(endpoint); addr != "" {
			pending.endpoints = append(pending.endpoints, addr)
		}
	}
	if len(pending.endpoints) == 0 {
		if endpoint, ok := targets[profilePrefill]; ok {
			if addr := eventEndpoint(endpoint); addr != "" {
				pending.endpoints = append(pending.endpoints, addr)
			}
		}
	}
	if len(pending.endpoints) == 0 {
		return
	}
	p.watch.add(pending)
	log.FromContext(ctx).V(logutil.TRACE).Info("Watching KV events for prefill completion",
		"requestID", request.RequestID, "stamp", stamp, "endpoints", pending.endpoints)
}

// ProcessEvents implements kvevents.EventConsumer. A store event stamped with
// a watched request's session id on one of its endpoints completes the
// request's prefill when it carries the prompt's last full block.
func (p *InFlightLoadProducer) ProcessEvents(ctx context.Context, source kvevents.EventSource, batch kvevents.EventBatch) error {
	for _, event := range batch.Events {
		stored, ok := event.(*kvevents.BlockStoredEvent)
		if !ok || stored.SessionID == nil || *stored.SessionID == "" {
			continue
		}
		pending := p.watch.lookup(source.Endpoint, *stored.SessionID)
		if pending == nil {
			continue
		}
		if pending.markStored(stored.Tokens, stored.BlockSize) {
			p.onPrefillComplete(ctx, pending.requestID, pending.result, signalKVEvent)
		}
	}
	return nil
}

// Reset implements kvevents.EventConsumer.
func (p *InFlightLoadProducer) Reset(ctx context.Context, source kvevents.EventSource) error {
	if dropped := p.watch.reset(source.Endpoint); dropped > 0 {
		log.FromContext(ctx).V(logutil.DEBUG).Info("KV event stream reset; pending prefills on the endpoint complete at first byte",
			"endpoint", source.Endpoint, "requests", dropped)
	}
	return nil
}

// onPrefillComplete frees what prefill completion frees: the prefill
// profile's entry in full, and with output tokens excluded the other
// profiles' tokens as well. It runs once per request; a later signal for the
// same request finds nothing pending and changes nothing.
func (p *InFlightLoadProducer) onPrefillComplete(ctx context.Context, requestID string,
	result *fwksched.SchedulingResult, signal string,
) {
	if p.watch.remove(requestID) {
		prefillCompletions.WithLabelValues(p.typedName.Name, signal).Inc()
		log.FromContext(ctx).V(logutil.DEBUG).Info("Prefill complete", "requestID", requestID, "signal", signal)
	}
	for profileName, profileResult := range result.ProfileResults {
		if profileResult == nil || len(profileResult.TargetEndpoints) == 0 {
			continue
		}
		endpoint := profileResult.TargetEndpoints[0]
		if endpoint == nil || endpoint.GetMetadata() == nil {
			continue
		}
		switch {
		case profileName == profilePrefill:
			p.release(endpoint, requestID, profileName)
		case !p.addEstimatedOutputTokens:
			p.releaseTokensEarly(endpoint, requestID, profileName)
		}
	}
}
