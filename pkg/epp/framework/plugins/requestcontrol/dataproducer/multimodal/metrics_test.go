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

package multimodal

import (
	"context"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	attrmm "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/multimodal"
)

func TestRecordItemLookupsMetrics(t *testing.T) {
	producer := newTestProducer(t, nil, nil)

	pod := k8stypes.NamespacedName{Namespace: "default", Name: "pod-a"}
	podKey := pod.String()
	img := string(fwkrh.ModalityImage)

	initialQueries := testutil.ToFloat64(encoderCacheQueriesTotal.WithLabelValues(ProducerType, testName, img))
	initialHits := testutil.ToFloat64(encoderCacheHitsTotal.WithLabelValues(ProducerType, testName, podKey, img))

	// Case 1: Cache Misses
	items := []attrmm.MatchItem{
		{Hash: "hash-1", Size: 1, Modality: img},
		{Hash: "hash-2", Size: 1, Modality: img},
	}
	producer.recordItemLookups(items)

	assert.Equal(t, initialQueries+2, testutil.ToFloat64(encoderCacheQueriesTotal.WithLabelValues(ProducerType, testName, img)))
	assert.Equal(t, initialHits, testutil.ToFloat64(encoderCacheHitsTotal.WithLabelValues(ProducerType, testName, podKey, img)))

	// Case 2: Cache Hits (add one to cache first)
	producer.putCacheEntry("hash-1", pod)

	items = []attrmm.MatchItem{
		{Hash: "hash-1", Size: 1, Modality: img}, // Hit
		{Hash: "hash-3", Size: 1, Modality: img}, // Miss
	}
	producer.recordItemLookups(items)

	assert.Equal(t, initialQueries+4, testutil.ToFloat64(encoderCacheQueriesTotal.WithLabelValues(ProducerType, testName, img)))
	assert.Equal(t, initialHits+1, testutil.ToFloat64(encoderCacheHitsTotal.WithLabelValues(ProducerType, testName, podKey, img)))
}

func TestRegisterEncoderCacheMetrics(t *testing.T) {
	// registerEncoderCacheMetrics uses sync.Once, so multiple calls are safe.
	// We can't easily verify registration against a mock registry because it uses the global metrics.Registry.
	// But we can verify it doesn't panic.
	assert.NotPanics(t, func() {
		registerEncoderCacheMetrics()
		registerEncoderCacheMetrics()
	})
}

func TestProduceRecordsMetrics(t *testing.T) {
	producer := newTestProducer(t, nil, nil)
	request := requestWithHashes("req-1", map[string]int{"hash-1": 1, "hash-2": 1})
	img := string(fwkrh.ModalityImage)

	initialQueries := testutil.ToFloat64(encoderCacheQueriesTotal.WithLabelValues(ProducerType, testName, img))

	// Produce should call recordItemLookups
	require.NoError(t, producer.Produce(context.Background(), request, nil))

	assert.Equal(t, initialQueries+2, testutil.ToFloat64(encoderCacheQueriesTotal.WithLabelValues(ProducerType, testName, img)))
}
