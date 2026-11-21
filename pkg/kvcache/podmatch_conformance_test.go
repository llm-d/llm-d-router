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

package kvcache_test

import (
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/llm-d/llm-d-router/pkg/kvcache"
)

// TestPodMatchFieldSet pins PodMatch's field set. Both index read paths
// (walk and materialized lookup) must populate every field, and in-tree
// consumers read a specific subset: ScoreTokens reads WeightedScore and
// MatchedBlocks telemetry, and the preciseprefixcache producer reads all
// four. A field added here is a contract change that both read paths and
// the parity assertions must carry; a field removed breaks the producer.
func TestPodMatchFieldSet(t *testing.T) {
	want := []reflect.StructField{
		{Name: "WeightedScore", Type: reflect.TypeOf(float64(0))},
		{Name: "MatchedBlocks", Type: reflect.TypeOf(int(0))},
		{Name: "ConfirmedBlocks", Type: reflect.TypeOf(int(0))},
		{Name: "BlocksByTier", Type: reflect.TypeOf(map[string]int(nil))},
	}
	got := reflect.VisibleFields(reflect.TypeOf(kvcache.PodMatch{}))
	assert.Len(t, got, len(want), "the PodMatch field set changed; update the read paths, the parity assertions, and this list deliberately")
	for i, f := range got {
		assert.Equal(t, want[i].Name, f.Name, "field %d", i)
		assert.Equal(t, want[i].Type, f.Type, "field %q type", f.Name)
	}
}
