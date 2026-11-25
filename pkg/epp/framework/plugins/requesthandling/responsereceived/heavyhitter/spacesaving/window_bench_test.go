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

package spacesaving

import (
	"fmt"
	"testing"
	"time"
)

func BenchmarkAdd_Hit(b *testing.B) {
	w, err := New(Config{Window: time.Hour, Slice: 30 * time.Second, Slots: 256}, testOrigin)
	if err != nil {
		b.Fatal(err)
	}
	now := testOrigin
	w.Add(now, "user", 10)
	b.ReportAllocs()
	for b.Loop() {
		w.Add(now, "user", 10)
	}
}

func BenchmarkAdd_Takeover(b *testing.B) {
	const ids = 1000 // above Slots, so every update after warm-up is a takeover
	w, err := New(Config{Window: time.Hour, Slice: 30 * time.Second, Slots: 64}, testOrigin)
	if err != nil {
		b.Fatal(err)
	}
	keys := make([]string, ids)
	for i := range keys {
		keys[i] = fmt.Sprintf("user-%d", i)
	}
	now := testOrigin
	i := 0
	b.ReportAllocs()
	for b.Loop() {
		w.Add(now, keys[i%ids], 10)
		i++
	}
}

func BenchmarkSnapshot(b *testing.B) {
	// A full merged view at the default sizing: 120 slices of 256 slots.
	w, err := New(Config{Window: time.Hour, Slice: 30 * time.Second, Slots: 256}, testOrigin)
	if err != nil {
		b.Fatal(err)
	}
	now := testOrigin
	for sliceIdx := range 120 {
		sliceNow := now.Add(time.Duration(sliceIdx*30) * time.Second)
		for i := range 256 {
			w.Add(sliceNow, fmt.Sprintf("user-%d-%d", sliceIdx, i), 100)
		}
	}
	b.ReportAllocs()
	for b.Loop() {
		w.Snapshot(now.Add(119*30*time.Second), 10)
	}
}
