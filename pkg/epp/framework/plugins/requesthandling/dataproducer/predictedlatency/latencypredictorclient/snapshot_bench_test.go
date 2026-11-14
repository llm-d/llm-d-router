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

package latencypredictorclient

import (
	"encoding/json"
	"strconv"
	"testing"
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/predictedlatency/latencypredictorclient/protocol"
	"google.golang.org/protobuf/proto"
)

func benchmarkSnapshot(index int) *protocol.Features {
	return &protocol.Features{
		KvCachePercentage: proto.Float64(0.5), InputTokenLength: proto.Int64(int64(100 + index)),
		NumRequestWaiting: proto.Int64(2), NumRequestRunning: proto.Int64(3),
		NumTokensGenerated: proto.Int64(1), PrefixCacheScore: proto.Float64(0.25),
		EncoderInputSize: proto.Int64(8), EncoderMatchedSize: proto.Int64(3),
		PodType: proto.String("decode"), PrefillTokensInFlight: proto.Int64(123), DecodeTokensInFlight: proto.Int64(45),
	}
}

func BenchmarkCapturedBulkPredictionWire(b *testing.B) {
	for _, count := range []int{1, 20, 100} {
		b.Run(strconv.Itoa(count), func(b *testing.B) {
			batch := BulkPredictionRequest{Requests: make([]PredictionRequest, count)}
			for index := range batch.Requests {
				batch.Requests[index] = PredictionFromFeatures(benchmarkSnapshot(index))
			}
			b.ReportAllocs()
			for b.Loop() {
				if _, err := json.Marshal(batch); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkCapturedBulkTrainingWire(b *testing.B) {
	for _, count := range []int{1, 20, 100} {
		b.Run(strconv.Itoa(count), func(b *testing.B) {
			batch := BulkTrainingRequest{Entries: make([]TrainingEntry, count)}
			for index := range batch.Entries {
				batch.Entries[index] = TrainingFromFeatures(benchmarkSnapshot(index), TrainingEntry{
					ActualTTFT: 50, Timestamp: time.Unix(100, 123).In(time.FixedZone("CST", 8*3600)),
				})
			}
			b.ReportAllocs()
			for b.Loop() {
				if _, err := json.Marshal(batch); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
