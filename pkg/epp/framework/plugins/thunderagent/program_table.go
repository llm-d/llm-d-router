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

package thunderagent

import (
	"math"
	"sync"
	"time"
)

// program is one program's token footprint and where it lives. A program is an
// agent trajectory identified by the request FairnessID.
type program struct {
	// podName is the endpoint the program is bound to, set on every dispatch.
	podName string
	// inflightTokens is the sum of estimates for this program's requests
	// currently being processed.
	inflightTokens int64
	// committedTokens is the total token count reported by the usage block of
	// the program's most recent completed request. For chat-style agents each
	// request carries the full history, so the latest total approximates the
	// program's current KV footprint; values replace rather than accumulate.
	committedTokens int64
	lastResponseAt  time.Time
	lastActivity    time.Time
	// dispatchCount counts requests forwarded for this program. A program with
	// dispatchCount > 0 has KV state on its pod and classifies as REASONING
	// for admission ordering.
	dispatchCount int64
	// finalSeen records that the session-final header was observed; the
	// program is released when that turn's response completes.
	finalSeen bool
	// parentID is the parent session of a subagent, recorded for
	// observability only.
	parentID string
}

// podLoad aggregates the programs bound to one pod.
type podLoad struct {
	// tokens is the decayed working set: committed plus in-flight tokens.
	tokens float64
	// programs is the number of programs bound to the pod.
	programs int
}

// podSnapshot is one pod's capacity and working set as of the last
// Saturation refresh. The fairness policy's admission fit check reads it.
type podSnapshot struct {
	capacity  float64
	tokens    float64 // decayed working set plus per-program buffers
	updatedAt time.Time
}

// pendingAdmission reserves room for a new program the fairness policy has
// released but whose PreRequest has not yet bound it to a pod. Without it,
// every dispatch between release and binding would see the same free room and
// over-admit.
type pendingAdmission struct {
	tokens float64
	pod    string
	at     time.Time
}

// pendingAdmissionTTL bounds a reservation whose program never reached
// PreRequest, such as a request that failed after dispatch.
const pendingAdmissionTTL = 5 * time.Second

// snapshotStaleAfter drops a pod from the fit view when no Saturation call
// has reported it, keyed per pod so prefill and decode stage calls refresh
// their own pods without erasing each other's.
const snapshotStaleAfter = 5 * time.Second

// programTable is the state shared by all of the plugin's hooks: scheduling
// (scorer), flow control (saturation detector, fairness policy), and the
// request lifecycle (accounting, session-final release). All access goes
// through mu; every hook does O(1) map work except the per-pod aggregation and
// the eviction sweep, both bounded by program count.
type programTable struct {
	mu             sync.Mutex
	programs       map[string]*program
	bytesPerToken  float64
	actingHalfLife time.Duration
	ttl            time.Duration
	// snapshot is the per-pod fit view refreshed by Saturation, keyed by
	// endpoint ID.
	snapshot map[string]*podSnapshot
	// pending holds admission reservations keyed by program ID.
	pending map[string]pendingAdmission
	// shedsTotal counts programs unbound by the shed path, mirrored into the
	// state dump alongside the Prometheus counter.
	shedsTotal int64
}

func newProgramTable(cfg Config) *programTable {
	return &programTable{
		programs:       make(map[string]*program),
		bytesPerToken:  initialBytesPerToken,
		actingHalfLife: time.Duration(cfg.ActingHalfLifeSeconds * float64(time.Second)),
		ttl:            time.Duration(cfg.EvictionTTLSeconds * float64(time.Second)),
		snapshot:       make(map[string]*podSnapshot),
		pending:        make(map[string]pendingAdmission),
	}
}

// pendingOn sums live reservations charged to a pod, dropping expired ones.
// Callers must hold t.mu.
func (t *programTable) pendingOn(pod string, now time.Time) float64 {
	var total float64
	for id, p := range t.pending {
		if now.Sub(p.at) > pendingAdmissionTTL {
			delete(t.pending, id)
			continue
		}
		if p.pod == pod {
			total += p.tokens
		}
	}
	return total
}

// programTokens returns a program's current footprint: the larger of its
// in-flight estimate and its committed tokens. A turn's prefill covers the
// program's previous context (each agentic turn resends the whole history as
// a prefix), so during a turn the new estimate and the old committed total
// describe the same KV and summing them double counts -- which inflates the
// working set for exactly the duration of a turn and makes shed and
// admission chase a phantom. A program with no request in flight is between
// requests (typically running a tool); its committed tokens decay with the
// configured half-life because the engine gradually evicts its KV blocks.
// Callers must hold t.mu.
func (t *programTable) programTokens(st *program, now time.Time) float64 {
	tokens := float64(st.committedTokens)
	if f := float64(st.inflightTokens); f > tokens {
		tokens = f
	}
	if st.inflightTokens > 0 || t.actingHalfLife <= 0 || st.lastResponseAt.IsZero() {
		return tokens
	}
	elapsed := now.Sub(st.lastResponseAt)
	if elapsed <= 0 {
		return tokens
	}
	return float64(st.committedTokens) * math.Exp2(-float64(elapsed)/float64(t.actingHalfLife))
}

// podLoads aggregates the decayed footprint and program count per pod, keyed
// by endpoint ID. Callers must hold t.mu.
func (t *programTable) podLoads(now time.Time) map[string]podLoad {
	loads := make(map[string]podLoad)
	for _, st := range t.programs {
		load := loads[st.podName]
		load.tokens += t.programTokens(st, now)
		load.programs++
		loads[st.podName] = load
	}
	return loads
}

// classAndTokens returns the admission class and footprint for a program.
// REASONING requires being bound: the bypass is justified by the footprint
// already counting against a pod. A shed (unbound) program with history is
// RESUMING: fit-checked like a new program, but ranked ahead of
// never-admitted ones. Its footprint is its full committed tokens,
// undecayed, because readmission re-prefills the whole context regardless
// of how long it sat idle. An unknown program is NEW with no footprint.
// Callers must hold t.mu.
func (t *programTable) classAndTokens(id string, now time.Time) (programClass, float64) {
	st, ok := t.programs[id]
	if !ok || st.dispatchCount == 0 {
		return classNew, 0
	}
	if st.podName == "" {
		return classResuming, float64(st.committedTokens)
	}
	return classReasoning, t.programTokens(st, now)
}

// estimateTokens converts a request body size to a token estimate using the
// running bytes-per-token ratio. Callers must hold t.mu.
func (t *programTable) estimateTokens(sizeBytes int) int64 {
	if sizeBytes <= 0 {
		return 0
	}
	return int64(float64(sizeBytes) / t.bytesPerToken)
}

// refineEstimator updates the bytes-per-token ratio from an observed request
// size and its usage-reported prompt tokens. Callers must hold t.mu.
func (t *programTable) refineEstimator(sizeBytes, promptTokens int) {
	if promptTokens <= 0 || sizeBytes <= 0 {
		return
	}
	sample := float64(sizeBytes) / float64(promptTokens)
	ratio := estimatorMomentum*t.bytesPerToken + (1-estimatorMomentum)*sample
	t.bytesPerToken = math.Min(math.Max(ratio, minBytesPerToken), maxBytesPerToken)
}
