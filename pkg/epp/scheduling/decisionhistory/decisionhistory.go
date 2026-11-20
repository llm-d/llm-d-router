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

// Package decisionhistory retains the most recent scheduling decisions for
// the /debug/scheduling/recent endpoint. A Recorder holds a fixed-capacity
// ring of Decisions; collection happens through a per-request Collector
// carried in the scheduling context. When no Recorder is configured the
// context carries no Collector and every collection call site is a no-op.
package decisionhistory

import (
	"context"
	"math"
	"sync"
	"time"
	"unicode/utf8"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// FilterStage records one filter's effect on a profile's candidate set.
type FilterStage struct {
	Type   string `json:"type"`
	Name   string `json:"name"`
	Before int    `json:"before"`
	After  int    `json:"after"`
}

// ScorerEntry records a scorer's identity and configured weight.
type ScorerEntry struct {
	Type   string  `json:"type"`
	Name   string  `json:"name"`
	Weight float64 `json:"weight"`
}

// EndpointScore records one of a profile's highest-scoring endpoints: its
// weighted total and the raw score each scorer assigned it, keyed by the
// scorer's typed name.
type EndpointScore struct {
	Endpoint      string             `json:"endpoint"`
	WeightedScore float64            `json:"weightedScore"`
	ScorerScores  map[string]float64 `json:"scorerScores"`
}

// ProfileDecision records one scheduler profile's run for a request.
type ProfileDecision struct {
	Profile                 string          `json:"profile"`
	CandidatesBeforeFilters int             `json:"candidatesBeforeFilters"`
	Filters                 []FilterStage   `json:"filters"`
	Scorers                 []ScorerEntry   `json:"scorers"`
	TopEndpoints            []EndpointScore `json:"topEndpoints"`
	TargetEndpoints         []string        `json:"targetEndpoints"`
	Error                   string          `json:"error,omitempty"`
}

// Decision records one scheduling attempt: the request identity, each
// profile's run, and the outcome. Entries hold no request payloads or
// headers. The model string is truncated to maxModelLength, and every slice
// field is non-nil so entries always serialize as JSON arrays.
type Decision struct {
	Timestamp      time.Time         `json:"timestamp"`
	RequestID      string            `json:"requestID"`
	Model          string            `json:"model"`
	Profiles       []ProfileDecision `json:"profiles"`
	TargetEndpoint string            `json:"targetEndpoint,omitempty"`
	FailureReason  string            `json:"failureReason,omitempty"`
}

// maxModelLength caps the model string stored in a decision. The model is
// request-derived and its length is unbounded under the default
// --max-request-body-size, so the cap keeps each entry's size bounded.
const maxModelLength = 1024

// truncateModel shortens model to at most maxModelLength bytes, backing off to
// a rune boundary so the stored string stays valid UTF-8.
func truncateModel(model string) string {
	if len(model) <= maxModelLength {
		return model
	}
	end := maxModelLength
	for end > 0 && !utf8.RuneStart(model[end]) {
		end--
	}
	return model[:end]
}

// Recorder retains the most recent scheduling decisions in a fixed-capacity
// ring buffer. It is safe for concurrent use.
type Recorder struct {
	mu        sync.Mutex
	decisions []Decision
	next      int
	size      int
	topK      int
}

// NewRecorder returns a Recorder retaining the last capacity decisions, with
// at most topK endpoints scored per profile. topK below 1 is treated as 1.
func NewRecorder(capacity, topK int) *Recorder {
	if topK < 1 {
		topK = 1
	}
	return &Recorder{
		decisions: make([]Decision, capacity),
		topK:      topK,
	}
}

// Begin starts collecting a decision for one request. It returns nil when
// the recorder is nil or has no capacity, disabling collection for the
// request.
func (r *Recorder) Begin(requestID, model string) *Collector {
	if r == nil || len(r.decisions) == 0 {
		return nil
	}
	return &Collector{recorder: r, decision: Decision{
		RequestID: requestID,
		Model:     truncateModel(model),
		Profiles:  []ProfileDecision{},
	}}
}

// Commit stores the collected decision. timestamp is recorded as the
// decision time; targetEndpoint and failureReason carry the attempt outcome,
// with failureReason empty on success. No-op when c is nil.
func (r *Recorder) Commit(c *Collector, timestamp time.Time, targetEndpoint, failureReason string) {
	if r == nil || c == nil {
		return
	}
	d := c.decision
	d.Timestamp = timestamp.UTC()
	d.TargetEndpoint = targetEndpoint
	d.FailureReason = failureReason

	r.mu.Lock()
	defer r.mu.Unlock()
	r.decisions[r.next] = d
	r.next = (r.next + 1) % len(r.decisions)
	if r.size < len(r.decisions) {
		r.size++
	}
}

// Snapshot returns the retained decisions, newest first. The returned
// entries share slices with the ring; entries are immutable once committed.
func (r *Recorder) Snapshot() []Decision {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]Decision, r.size)
	for i := range out {
		idx := (r.next - 1 - i + len(r.decisions)) % len(r.decisions)
		out[i] = r.decisions[idx]
	}
	return out
}

// Collector accumulates one request's decision. A nil Collector is a no-op
// for every method, so capture sites can call unconditionally.
type Collector struct {
	recorder *Recorder
	decision Decision
	current  *profileCollector
}

// profileCollector holds the profile being collected. scorerKeys and
// scorerScores are index-aligned with the profile's Scorers; scorerKeys holds
// each scorer's TypedName.String(), and scorerScores maps endpoint ID to the
// raw score that scorer assigned. Entries are trimmed to the top endpoints at
// pick time.
type profileCollector struct {
	profile      ProfileDecision
	scorerKeys   []string
	scorerScores []map[string]float64
}

// ProfileStarted begins collecting the named profile's run. A profile that
// runs in several handler iterations produces one entry per run.
func (c *Collector) ProfileStarted(profile string, candidates int) {
	if c == nil {
		return
	}
	c.current = &profileCollector{profile: ProfileDecision{
		Profile:                 profile,
		CandidatesBeforeFilters: candidates,
		Filters:                 []FilterStage{},
		Scorers:                 []ScorerEntry{},
		TopEndpoints:            []EndpointScore{},
		TargetEndpoints:         []string{},
	}}
}

// FilterRan records one filter's candidate counts.
func (c *Collector) FilterRan(pluginType, name string, before, after int) {
	if c == nil || c.current == nil {
		return
	}
	c.current.profile.Filters = append(c.current.profile.Filters, FilterStage{
		Type:   pluginType,
		Name:   name,
		Before: before,
		After:  after,
	})
}

// ScorerRan records a scorer's identity, weight, and raw per-endpoint scores
// keyed by endpoint ID. Non-finite raw scores are dropped: a scorer that
// returns NaN or Inf must not make the retained entry unmarshalable.
func (c *Collector) ScorerRan(typedName fwkplugin.TypedName, weight float64, scores map[string]float64) {
	if c == nil || c.current == nil {
		return
	}
	c.current.profile.Scorers = append(c.current.profile.Scorers, ScorerEntry{
		Type:   typedName.Type,
		Name:   typedName.Name,
		Weight: weight,
	})
	c.current.scorerKeys = append(c.current.scorerKeys, typedName.String())
	bounded := make(map[string]float64, len(scores))
	for endpoint, score := range scores {
		if math.IsNaN(score) || math.IsInf(score, 0) {
			continue
		}
		bounded[endpoint] = score
	}
	c.current.scorerScores = append(c.current.scorerScores, bounded)
}

// Picked finishes the profile: topNames and topScores are the highest
// weighted scores index-aligned in descending order, and targets holds the
// picked endpoint IDs. Per-scorer scores are kept only for the top
// endpoints. A non-finite weighted score is recorded as 0.
func (c *Collector) Picked(topNames []string, topScores []float64, targets []string) {
	if c == nil || c.current == nil {
		return
	}
	pc := c.current
	for i, name := range topNames {
		weighted := topScores[i]
		if math.IsNaN(weighted) || math.IsInf(weighted, 0) {
			weighted = 0
		}
		es := EndpointScore{
			Endpoint:      name,
			WeightedScore: weighted,
			ScorerScores:  make(map[string]float64, len(pc.profile.Scorers)),
		}
		for j := range pc.profile.Scorers {
			if score, ok := pc.scorerScores[j][name]; ok {
				es.ScorerScores[pc.scorerKeys[j]] = score
			}
		}
		pc.profile.TopEndpoints = append(pc.profile.TopEndpoints, es)
	}
	pc.profile.TargetEndpoints = append(pc.profile.TargetEndpoints, targets...)
	c.commitProfile()
}

// ProfileFailed finishes the profile with the run's error.
func (c *Collector) ProfileFailed(err string) {
	if c == nil || c.current == nil {
		return
	}
	c.current.profile.Error = err
	c.commitProfile()
}

func (c *Collector) commitProfile() {
	c.decision.Profiles = append(c.decision.Profiles, c.current.profile)
	c.current = nil
}

// TopK returns the per-profile endpoint cap the recorder was configured
// with. Zero only on a nil Collector.
func (c *Collector) TopK() int {
	if c == nil {
		return 0
	}
	return c.recorder.topK
}

type collectorContextKey struct{}

// ContextWithCollector returns a context carrying the collector. A nil
// collector returns ctx unchanged.
func ContextWithCollector(ctx context.Context, c *Collector) context.Context {
	if c == nil {
		return ctx
	}
	return context.WithValue(ctx, collectorContextKey{}, c)
}

// CollectorFromContext returns the context's collector, or nil when decision
// recording is disabled.
func CollectorFromContext(ctx context.Context) *Collector {
	c, _ := ctx.Value(collectorContextKey{}).(*Collector)
	return c
}
