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

package approximateprefix

import (
	"context"
	"fmt"
	"testing"

	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/prefixhash"
)

const (
	benchBlockSize    = defaultBlockSizeTokens
	benchPromptBlocks = 512
	// benchSharedBlocks is the length of the system prompt shared by every
	// request in the sharedPrefix layout.
	benchSharedBlocks = 64
)

// benchTokens returns n token IDs derived from seed, distinct across seeds.
func benchTokens(seed uint32, n int) []uint32 {
	tokens := make([]uint32, n)
	for i := range tokens {
		tokens[i] = seed<<20 | uint32(i)
	}
	return tokens
}

// benchLayouts populate the indexer for a query prompt. Each returns the
// tokens of the query request.
var benchLayouts = map[string]func(p *dataProducer, pods []server) []uint32{
	// hotspot: every pod holds the whole query prompt.
	"hotspot": func(p *dataProducer, pods []server) []uint32 {
		query := benchTokens(1, benchPromptBlocks*benchBlockSize)
		hashes := benchHashes(p, query)
		for _, s := range pods {
			p.indexerInst.Add(hashes, s)
		}
		return query
	},
	// sharedPrefix: every pod holds a shared system prompt followed by its
	// own tail; the query shares only the system prompt.
	"sharedPrefix": func(p *dataProducer, pods []server) []uint32 {
		system := benchTokens(1, benchSharedBlocks*benchBlockSize)
		tailLen := (benchPromptBlocks - benchSharedBlocks) * benchBlockSize
		for i, s := range pods {
			prompt := append(append([]uint32{}, system...), benchTokens(uint32(100+i), tailLen)...)
			p.indexerInst.Add(benchHashes(p, prompt), s)
		}
		return append(append([]uint32{}, system...), benchTokens(2, tailLen)...)
	},
	// mixedDepth: pod i holds the first (i+1)/n of the query prompt, as in a
	// multi-turn conversation served at different turns by different pods.
	"mixedDepth": func(p *dataProducer, pods []server) []uint32 {
		query := benchTokens(1, benchPromptBlocks*benchBlockSize)
		hashes := benchHashes(p, query)
		for i, s := range pods {
			p.indexerInst.Add(hashes[:(i+1)*len(hashes)/len(pods)], s)
		}
		return query
	},
	// disjoint: every pod holds an unrelated prompt; the query misses at block 0.
	"disjoint": func(p *dataProducer, pods []server) []uint32 {
		for i, s := range pods {
			p.indexerInst.Add(benchHashes(p, benchTokens(uint32(100+i), benchPromptBlocks*benchBlockSize)), s)
		}
		return benchTokens(1, benchPromptBlocks*benchBlockSize)
	},
}

func benchHashes(p *dataProducer, tokens []uint32) []blockHash {
	req := &fwksched.InferenceRequest{TargetModel: "bench-model", Body: tokenizedBody(tokens)}
	return prefixhash.GetBlockHashes(context.Background(), req, benchBlockSize, p.resolveMaxBlocks(benchBlockSize))[0]
}

// benchPods returns n pods sized to the default LRU capacity and their
// endpoints.
func benchPods(n int) ([]server, []fwksched.Endpoint) {
	pods := make([]server, n)
	endpoints := make([]fwksched.Endpoint, n)
	for i := range pods {
		id := k8stypes.NamespacedName{Namespace: "default", Name: fmt.Sprintf("pod-%d", i)}
		pods[i] = server{ServerID: ServerID(id), NumOfGPUBlocks: defaultLRUCapacityPerServer}
		endpoints[i] = fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: id}, fwkdl.NewMetrics(), fwkdl.NewAttributes())
	}
	return pods, endpoints
}

func BenchmarkProduce(b *testing.B) {
	for _, layout := range []string{"hotspot", "sharedPrefix", "mixedDepth", "disjoint"} {
		for _, numPods := range []int{16, 96, 256} {
			b.Run(fmt.Sprintf("%s/pods=%d", layout, numPods), func(b *testing.B) {
				p, err := newDataProducer(b.Context(), ApproxPrefixCachePluginType, defaultConfig, testHandle())
				if err != nil {
					b.Fatal(err)
				}
				pods, endpoints := benchPods(numPods)
				query := benchLayouts[layout](p, pods)
				req := &fwksched.InferenceRequest{RequestID: "bench", TargetModel: "bench-model", Body: tokenizedBody(query)}

				b.ReportAllocs()
				for b.Loop() {
					if err := p.Produce(b.Context(), req, endpoints); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}
