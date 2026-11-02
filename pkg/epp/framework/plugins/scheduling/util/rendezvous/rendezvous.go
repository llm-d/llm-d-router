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

// Package rendezvous keys a request on its session and picks a target from
// weighted shares with rendezvous hashing. Nothing is stored: the same key and
// shares always pick the same target, on any replica.
package rendezvous

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"

	"github.com/cespare/xxhash/v2"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/util/sessionaffinity"
)

// Source reports what a key was derived from.
type Source string

const (
	// SourceNone means the request has nothing to key on.
	SourceNone Source = ""
	// SourceSession is a session identifier resolved from the sources.
	SourceSession Source = "session"
	// SourcePrefix is the text conversation up to its first user message.
	SourcePrefix Source = "prefix"
)

// Key returns a deterministic hash key for the request. The first session
// identifier resolved from sources wins; otherwise the text conversation up to
// and including its first user message is hashed, so every turn of a
// conversation gets the same key. It returns SourceNone when neither applies.
func Key(ctx context.Context, request *fwksched.InferenceRequest, sources []sessionaffinity.SessionIDSource) (uint64, Source) {
	if request == nil {
		return 0, SourceNone
	}
	if session := sessionaffinity.ResolveSessionID(ctx, request, sources); session != "" {
		return xxhash.Sum64String(session), SourceSession
	}
	var digest xxhash.Digest
	digest.Reset()
	if !request.Body.WriteConversationPrefix(&digest) {
		return 0, SourceNone
	}
	return digest.Sum64(), SourcePrefix
}

// Pick selects a target in proportion to shares using weighted rendezvous
// hashing (-ln(u) / share, lowest wins). Changing one target's share or
// removing it only moves keys to or from that target; keys on the remaining
// targets stay put regardless of target names or count. Targets with a
// non-positive share are skipped; it returns "" when none is left.
func Pick(shares map[string]float64, key uint64) string {
	best := ""
	bestScore := math.Inf(1)
	for target, share := range shares {
		if share <= 0 {
			continue
		}
		score := Score(key, target, share)
		if best == "" || score < bestScore || (score == bestScore && target < best) {
			best = target
			bestScore = score
		}
	}
	return best
}

// Overflow returns chosen when it has room. Otherwise it returns pick over the
// other targets in shares that have room, so a key that cannot stay spreads by
// share and the same key always lands on the same alternative. When no target
// has room, chosen is kept: moving would not help.
func Overflow(chosen string, shares map[string]float64, hasRoom func(target string) bool,
	pick func(withRoom map[string]float64) string) string {
	if chosen == "" || hasRoom(chosen) {
		return chosen
	}
	withRoom := make(map[string]float64, len(shares))
	for target, share := range shares {
		if target != chosen && hasRoom(target) {
			withRoom[target] = share
		}
	}
	if next := pick(withRoom); next != "" {
		return next
	}
	return chosen
}

// ResolveGuard returns the scheduling filter named name from handle, to decide
// which targets have room. field names the configuration field in errors. The
// guard must fail closed: a guard that keeps every endpoint when it cannot
// decide would hide a full target.
func ResolveGuard(handle plugin.Handle, name, field string) (fwksched.Filter, error) {
	if handle == nil {
		return nil, fmt.Errorf("%s: no plugin handle to resolve %q", field, name)
	}
	p := handle.Plugin(name)
	if p == nil {
		return nil, fmt.Errorf("%s: plugin %q not found", field, name)
	}
	guard, ok := p.(fwksched.Filter)
	if !ok {
		return nil, fmt.Errorf("%s: plugin %q is not a scheduling filter", field, name)
	}
	if failOpen, ok := p.(interface{ FailsOpen() bool }); ok && failOpen.FailsOpen() {
		return nil, fmt.Errorf("%s: plugin %q fails open; set failOpen: false", field, name)
	}
	return guard, nil
}

// MergeGuardConsumes adds the guard's data dependencies to deps. A plugin that
// runs a guard filter on the endpoints it was handed must declare what the guard
// reads: those endpoints are scoped to the calling plugin's declaration, so an
// undeclared key reads as missing inside the guard. A key the guard requires
// becomes required.
func MergeGuardConsumes(deps plugin.DataDependencies, guard fwksched.Filter) plugin.DataDependencies {
	consumer, ok := guard.(plugin.ConsumerPlugin)
	if !ok {
		return deps
	}
	guardDeps := consumer.Consumes()
	for key, witness := range guardDeps.Required {
		if deps.Required == nil {
			deps.Required = map[plugin.DataKey]any{}
		}
		deps.Required[key] = witness
		delete(deps.Optional, key)
	}
	for key, witness := range guardDeps.Optional {
		if _, required := deps.Required[key]; required {
			continue
		}
		if deps.Optional == nil {
			deps.Optional = map[plugin.DataKey]any{}
		}
		deps.Optional[key] = witness
	}
	return deps
}

// Score is the weighted rendezvous score of key on target; lower wins.
func Score(key uint64, target string, share float64) float64 {
	var digest xxhash.Digest
	digest.Reset()
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], key)
	_, _ = digest.Write(buf[:])
	_, _ = digest.WriteString(target)
	// Map the top 53 hash bits onto (0, 1] so math.Log is finite and <= 0.
	u := float64((digest.Sum64()>>11)+1) / (1 << 53)
	return -math.Log(u) / share
}
