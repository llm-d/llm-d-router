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

package disagg

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"strconv"
	"sync"
	"time"

	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/log"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const (
	defaultKVCompatFailureThreshold  = 3
	defaultKVCompatExclusionDuration = 5 * time.Minute
	kvCompatEndpointPairPrefix       = "endpoint:"
)

var defaultKVCompatFailureCodes = []int{500, 502, 503}

// kvCompatExclusionConfig holds the parsed failure-based exclusion parameters.
type kvCompatExclusionConfig struct {
	threshold      int
	duration       time.Duration
	failureCodes   map[int]struct{}
	prefillProfile string
}

func (c kvCompatExclusionConfig) enabled() bool {
	return c.threshold > 0
}

func newKVCompatExclusionConfig(p KVCompatPDDeciderParameters) (kvCompatExclusionConfig, error) {
	threshold := ptr.Deref(p.FailureThreshold, defaultKVCompatFailureThreshold)
	duration := defaultKVCompatExclusionDuration
	if p.ExclusionDuration != nil {
		duration = p.ExclusionDuration.Duration
	}
	codes := p.FailureCodes
	if codes == nil {
		codes = defaultKVCompatFailureCodes
	}
	var errs []error
	if threshold < 0 {
		errs = append(errs, fmt.Errorf("failureThreshold must be non-negative, got %d", threshold))
	}
	if threshold > 0 && duration <= 0 {
		errs = append(errs, fmt.Errorf("exclusionDuration must be positive, got %v", duration))
	}
	set := make(map[int]struct{}, len(codes))
	for _, code := range codes {
		if code < 400 || code > 599 {
			errs = append(errs, fmt.Errorf("failureCodes must be 4xx or 5xx statuses, got %d", code))
		}
		set[code] = struct{}{}
	}
	if len(errs) > 0 {
		return kvCompatExclusionConfig{}, errors.Join(errs...)
	}
	profile := p.PrefillProfile
	if profile == "" {
		profile = DefaultPrefillProfile
	}
	return kvCompatExclusionConfig{threshold: threshold, duration: duration, failureCodes: set, prefillProfile: profile}, nil
}

// KVCompatExclusions lists, for one decode endpoint, the prefill pairs excluded after failed
// requests and when each exclusion ends, in Unix milliseconds. A pair is a prefill compatibility
// key, or a prefill endpoint ID prefixed with "endpoint:" when the prefill endpoint has no key.
// It is the value shared across EPP replicas.
type KVCompatExclusions struct {
	Until map[string]int64
}

// Clone implements fwkdl.Cloneable.
func (e *KVCompatExclusions) Clone() fwkdl.Cloneable {
	if e == nil {
		return nil
	}
	return &KVCompatExclusions{Until: maps.Clone(e.Until)}
}

func (e *KVCompatExclusions) excluded(pair string, nowMs int64) bool {
	return e != nil && e.Until[pair] > nowMs
}

var noKVCompatExclusions = &KVCompatExclusions{}

// kvCompatPairHealth is the local failure state of one pair.
type kvCompatPairHealth struct {
	failures int
	until    time.Time
}

// kvCompatExclusionState is the local failure state of every pair, by decode endpoint ID.
type kvCompatExclusionState struct {
	mu    sync.Mutex
	pairs map[string]map[string]*kvCompatPairHealth
}

// kvCompatRequestPair is the pair serving one disaggregated request.
type kvCompatRequestPair struct {
	decodeID string
	pair     string
}

// Clone implements plugin.StateData.
func (p *kvCompatRequestPair) Clone() plugin.StateData {
	c := *p
	return &c
}

func kvCompatPairOf(rule kvCompatRule, prefill *fwkdl.EndpointMetadata) string {
	if key := prefill.Labels[rule.keyLabel]; key != "" {
		return key
	}
	return kvCompatEndpointPairPrefix + prefill.ID.String()
}

// exclusionsFor returns the pairs excluded for a decode endpoint, merged across EPP replicas
// when a cross-replica syncer is configured. The result must not be modified.
func (d *KVCompatPDDecider) exclusionsFor(ctx context.Context, decode fwkdl.ID) *KVCompatExclusions {
	if !d.exclusion.enabled() {
		return noKVCompatExclusions
	}
	syncer := d.crossReplicaSyncer()
	if syncer == nil && !d.hasLocalExclusions() {
		return noKVCompatExclusions
	}
	decodeID := d.endpointIDString(decode)
	if syncer != nil {
		v, ok, err := syncer.Get(ctx, d.spec, decodeID)
		if err == nil && ok {
			if e, ok := v.(*KVCompatExclusions); ok && e != nil {
				return e
			}
		}
	}
	return d.localExclusions(decodeID)
}

// endpointIDString returns the endpoint ID as a string, from the index when the endpoint is in it.
func (d *KVCompatPDDecider) endpointIDString(id fwkdl.ID) string {
	d.mu.RLock()
	e, ok := d.endpoints[id]
	d.mu.RUnlock()
	if ok && e.id != "" {
		return e.id
	}
	return id.String()
}

func (d *KVCompatPDDecider) hasLocalExclusions() bool {
	d.state.mu.Lock()
	defer d.state.mu.Unlock()
	return len(d.state.pairs) > 0
}

// localExclusions returns this replica's active exclusions for a decode endpoint.
func (d *KVCompatPDDecider) localExclusions(decodeID string) *KVCompatExclusions {
	nowMs := d.now().UnixMilli()
	d.state.mu.Lock()
	defer d.state.mu.Unlock()
	var out *KVCompatExclusions
	for pair, h := range d.state.pairs[decodeID] {
		if until := h.until.UnixMilli(); until > nowMs {
			if out == nil {
				out = &KVCompatExclusions{Until: map[string]int64{}}
			}
			out.Until[pair] = until
		}
	}
	if out == nil {
		return noKVCompatExclusions
	}
	return out
}

func (d *KVCompatPDDecider) crossReplicaSyncer() fwkdl.CrossReplicaSyncer {
	if d.handle == nil {
		return nil
	}
	syncer, _ := d.handle.CrossReplicaSyncer().(fwkdl.CrossReplicaSyncer)
	return syncer
}

func (d *KVCompatPDDecider) buildSpec() fwkdl.CrossReplicaSpec {
	return fwkdl.CrossReplicaSpec{
		StateKey:     fwkdl.StateKey("kv-compat-exclusions:" + d.typedName.Name),
		AttributeKey: plugin.NewDataKey("kv-compat-exclusions", d.typedName.Name),
		SyncDisabled: !d.exclusion.enabled(),
		Read: func(endpointID string) fwkdl.Cloneable {
			return d.localExclusions(endpointID)
		},
		Aggregate: aggregateKVCompatExclusions,
	}
}

// aggregateKVCompatExclusions merges the exclusions of several replicas, keeping the latest expiry
// of each pair. When at most one value holds exclusions it is returned as is, so a decision that
// reads merged state does not allocate in the common case. The result must not be modified.
func aggregateKVCompatExclusions(values []any) any {
	var only *KVCompatExclusions
	nonEmpty := 0
	for _, v := range values {
		if e, ok := v.(*KVCompatExclusions); ok && e != nil && len(e.Until) > 0 {
			only = e
			nonEmpty++
		}
	}
	switch nonEmpty {
	case 0:
		return noKVCompatExclusions
	case 1:
		return only
	}
	merged := &KVCompatExclusions{Until: map[string]int64{}}
	for _, v := range values {
		e, ok := v.(*KVCompatExclusions)
		if !ok || e == nil {
			continue
		}
		for pair, until := range e.Until {
			merged.Until[pair] = max(merged.Until[pair], until)
		}
	}
	return merged
}

// CrossReplicaState shares the pair exclusions of each decode endpoint with peer EPP replicas.
func (d *KVCompatPDDecider) CrossReplicaState() fwkdl.CrossReplicaSpec {
	return d.spec
}

// PreRequest records which pair serves a disaggregated request, so its response can count
// toward excluding the pair.
func (d *KVCompatPDDecider) PreRequest(_ context.Context, request *scheduling.InferenceRequest, result *scheduling.SchedulingResult) error {
	if !d.exclusion.enabled() || request == nil || result == nil {
		return nil
	}
	decode := result.PrimaryEndpoint()
	prefillRes := result.ProfileResults[d.exclusion.prefillProfile]
	if decode == nil || decode.GetMetadata() == nil || prefillRes == nil {
		return nil
	}
	prefill := prefillRes.FirstEndpoint()
	if prefill == nil || prefill.GetMetadata() == nil {
		return nil
	}
	d.pluginState.Write(request.RequestID, plugin.StateKey(d.typedName.Name), &kvCompatRequestPair{
		decodeID: decode.GetMetadata().ID.String(),
		pair:     kvCompatPairOf(d.rule, prefill.GetMetadata()),
	})
	return nil
}

// ResponseHeader counts a response status in failureCodes as a failed request for the pair.
func (d *KVCompatPDDecider) ResponseHeader(ctx context.Context, request *scheduling.InferenceRequest,
	response *fwkrh.Response, _ *fwkdl.EndpointMetadata) {
	if !d.exclusion.enabled() || response == nil {
		return
	}
	raw, ok := response.Headers[reqcommon.PseudoHeaderStatus]
	if !ok {
		raw = response.Headers[reqcommon.HeaderStatus]
	}
	code, err := strconv.Atoi(raw)
	if err != nil {
		return
	}
	if _, failed := d.exclusion.failureCodes[code]; failed {
		d.finish(ctx, kvCompatRequestID(request, response), false)
	}
}

// ResponseBody records the outcome when the response ends. A stream that ends with an error, or
// that ends normally after reading the prompt without generating a token, counts as a failure.
// Client disconnects and evictions say nothing about the pair.
func (d *KVCompatPDDecider) ResponseBody(ctx context.Context, request *scheduling.InferenceRequest,
	response *fwkrh.Response, _ *fwkdl.EndpointMetadata) {
	if !d.exclusion.enabled() || response == nil || !response.EndOfStream {
		return
	}
	id := kvCompatRequestID(request, response)
	switch response.TerminationCause {
	case fwkrh.TerminationCauseNatural, "":
		d.finish(ctx, id, response.Usage.PromptTokens <= 0 || response.Usage.CompletionTokens != 0)
	case fwkrh.TerminationCauseError:
		d.finish(ctx, id, false)
	default:
		d.pluginState.Delete(id)
	}
}

func kvCompatRequestID(request *scheduling.InferenceRequest, response *fwkrh.Response) string {
	if request != nil && request.RequestID != "" {
		return request.RequestID
	}
	return response.RequestID
}

// finish applies the outcome of one disaggregated request to its pair.
func (d *KVCompatPDDecider) finish(ctx context.Context, requestID string, ok bool) {
	pair, err := plugin.ReadPluginStateKey[*kvCompatRequestPair](d.pluginState, requestID, plugin.StateKey(d.typedName.Name))
	if err != nil {
		return
	}
	d.pluginState.Delete(requestID)
	now := d.now()
	d.state.mu.Lock()
	byPair := d.state.pairs[pair.decodeID]
	if byPair == nil {
		byPair = map[string]*kvCompatPairHealth{}
		d.state.pairs[pair.decodeID] = byPair
	}
	h := byPair[pair.pair]
	if h == nil {
		h = &kvCompatPairHealth{}
		byPair[pair.pair] = h
	}
	excluded := false
	switch {
	case ok:
		h.failures = 0
	case now.Before(h.until):
		// Responses of requests scheduled before the exclusion started do not extend it.
	default:
		h.failures++
		if h.failures >= d.exclusion.threshold {
			h.failures = 0
			h.until = now.Add(d.exclusion.duration)
			excluded = true
		}
	}
	if h.failures == 0 && !now.Before(h.until) {
		delete(byPair, pair.pair)
		if len(byPair) == 0 {
			delete(d.state.pairs, pair.decodeID)
		}
	}
	d.state.mu.Unlock()
	if excluded {
		recordKVCompatExclusion(d.typedName.Name)
		log.FromContext(ctx).Info("kv-compat-pd-decider: excluding prefill pair after failed requests",
			"decode", pair.decodeID, "prefill", pair.pair, "failures", d.exclusion.threshold, "duration", d.exclusion.duration)
	}
}

// forgetEndpoint drops the local failure state of a deleted decode endpoint.
func (d *KVCompatPDDecider) forgetEndpoint(id fwkdl.ID) {
	d.state.mu.Lock()
	defer d.state.mu.Unlock()
	delete(d.state.pairs, id.String())
}
