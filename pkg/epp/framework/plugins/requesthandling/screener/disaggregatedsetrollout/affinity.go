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

package disaggregatedsetrollout

import (
	"context"

	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/util/rendezvous"
)

// affinityKey returns a deterministic hash key for the request and the source
// it came from, or sourceRandom when affinity is off or the request has nothing
// to key on.
//
// The first session identifier resolved from Affinity.Sources takes
// precedence over the text conversation prefix.
func (c *Screener) affinityKey(ctx context.Context, request *fwksched.InferenceRequest) (uint64, selectionSource) {
	if c.config.RevisionGating.affinityMode() != AffinityModePrefix || request == nil {
		return 0, sourceRandom
	}
	key, source := rendezvous.Key(ctx, request, c.config.RevisionGating.Affinity.Sources)
	switch source {
	case rendezvous.SourceSession:
		return key, sourceSession
	case rendezvous.SourcePrefix:
		return key, sourcePrefix
	default:
		return 0, sourceRandom
	}
}

// pickRendezvousRevision selects a revision in proportion to shares; see
// rendezvous.Pick.
func pickRendezvousRevision(shares map[string]float64, key uint64) string {
	return rendezvous.Pick(shares, key)
}
