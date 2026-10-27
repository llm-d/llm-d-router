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

package redis

import (
	"context"
	"fmt"
	"testing"

	"github.com/alicebob/miniredis/v2"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

// BenchmarkSyncerSet measures redis-syncer Set (publish) latency against
// miniredis for growing PrefixHashSet payloads. Production Redis/Valkey adds
// network RTT on top of these numbers; the publisher runs this off the
// request path on the cross-replica sync interval.
func BenchmarkSyncerSet(b *testing.B) {
	for _, prompts := range []int{1, 4, 8} {
		for _, hashesPerPrompt := range []int{16, 64, 256} {
			b.Run(fmt.Sprintf("prompts=%d/hashes=%d", prompts, hashesPerPrompt), func(b *testing.B) {
				server := miniredis.RunT(b)
				syncer, _ := newTestSyncer(b, server, testReplicaID)
				local := benchPrefixHashSet(prompts, hashesPerPrompt)
				spec := fwkdl.CrossReplicaSpec{
					StateKey: "approx-prefix:bench",
					Read:     func(string) fwkdl.Cloneable { return local.Clone() },
					Aggregate: func(values []any) any {
						if len(values) == 0 {
							return &attrprefix.PrefixHashSet{}
						}
						return values[0]
					},
				}

				b.ReportAllocs()
				for b.Loop() {
					if err := syncer.Set(context.Background(), spec, testEndpointID); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

// BenchmarkSyncerGet measures Get after a peer field is already published.
func BenchmarkSyncerGet(b *testing.B) {
	server := miniredis.RunT(b)
	publisher, _ := newTestSyncer(b, server, "epp-a")
	reader, _ := newTestSyncer(b, server, "epp-b")
	local := benchPrefixHashSet(4, 64)
	spec := fwkdl.CrossReplicaSpec{
		StateKey: "approx-prefix:bench",
		Read:     func(string) fwkdl.Cloneable { return local.Clone() },
		Aggregate: func(values []any) any {
			out := &attrprefix.PrefixHashSet{}
			for _, v := range values {
				if set, ok := v.(*attrprefix.PrefixHashSet); ok && set != nil {
					out.Prompts = append(out.Prompts, set.Prompts...)
				}
			}
			return out
		},
	}
	if err := publisher.Set(context.Background(), spec, testEndpointID); err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	for b.Loop() {
		if _, _, err := reader.Get(context.Background(), spec, testEndpointID); err != nil {
			b.Fatal(err)
		}
	}
}

func benchPrefixHashSet(prompts, hashesPerPrompt int) *attrprefix.PrefixHashSet {
	out := make([][]uint64, prompts)
	for i := range out {
		h := make([]uint64, hashesPerPrompt)
		for j := range h {
			h[j] = uint64(i+1)<<32 | uint64(j+1)
		}
		out[i] = h
	}
	return &attrprefix.PrefixHashSet{Prompts: out}
}
