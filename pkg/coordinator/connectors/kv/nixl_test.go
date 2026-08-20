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

package kv

import (
	"context"
	"strconv"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

func TestNixlKV_HeadersDisabledAtDPSizeOne(t *testing.T) {
	c, err := Build(NIXL, 1)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	rh, ok := c.(RankHeaderer)
	if !ok {
		t.Fatal("nixlKV does not implement RankHeaderer")
	}

	reqCtx := &pipeline.RequestContext{RequestID: "req-1"}
	if got := rh.PrefillHeaders(context.Background(), reqCtx); got != nil {
		t.Errorf("PrefillHeaders() = %v, want nil at dp_size=1", got)
	}
	if got := rh.DecodeHeaders(context.Background(), reqCtx); got != nil {
		t.Errorf("DecodeHeaders() = %v, want nil at dp_size=1", got)
	}
}

func TestNixlKV_PrefillHeadersPinsRank(t *testing.T) {
	const dpSize = 8
	c, err := Build(NIXL, dpSize)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	rh := c.(RankHeaderer)

	reqCtx := &pipeline.RequestContext{RequestID: "req-prefill-pin"}
	wantRank := pickDPRank(reqCtx.RequestID, dpSize)

	got := rh.PrefillHeaders(context.Background(), reqCtx)
	want := map[string]string{routing.DataParallelRankHeader: strconv.Itoa(wantRank)}
	if len(got) != len(want) || got[routing.DataParallelRankHeader] != want[routing.DataParallelRankHeader] {
		t.Errorf("PrefillHeaders() = %v, want %v", got, want)
	}
}

func TestNixlKV_DecodeHeadersPrefersReturnedRank(t *testing.T) {
	const dpSize = 8
	c, err := Build(NIXL, dpSize)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	rh := c.(RankHeaderer)

	t.Run("uses returned rank when valid", func(t *testing.T) {
		reqCtx := &pipeline.RequestContext{
			RequestID:        "req-decode-pin",
			KVTransferParams: map[string]any{remoteDPRankField: float64(4)},
		}
		got := rh.DecodeHeaders(context.Background(), reqCtx)
		want := map[string]string{routing.DataParallelRankHeader: "4"}
		if got[routing.DataParallelRankHeader] != want[routing.DataParallelRankHeader] {
			t.Errorf("DecodeHeaders() = %v, want %v", got, want)
		}
	})

	t.Run("falls back to the same hash prefill used when absent", func(t *testing.T) {
		reqCtx := &pipeline.RequestContext{
			RequestID:        "req-decode-fallback",
			KVTransferParams: map[string]any{},
		}
		wantRank := pickDPRank(reqCtx.RequestID, dpSize)
		got := rh.DecodeHeaders(context.Background(), reqCtx)
		want := strconv.Itoa(wantRank)
		if got[routing.DataParallelRankHeader] != want {
			t.Errorf("DecodeHeaders() = %v, want rank %s", got, want)
		}
	})
}

func TestSharedStorageAndSGLangKV_NotRankHeaderers(t *testing.T) {
	for _, name := range []string{SharedStorage, SGLang} {
		c, err := Build(name, 8)
		if err != nil {
			t.Fatalf("Build(%q): %v", name, err)
		}
		if _, ok := c.(RankHeaderer); ok {
			t.Errorf("%s: unexpectedly implements RankHeaderer", name)
		}
	}
}
