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

package hetero

import (
	"testing"
	"time"
)

func TestPairKey(t *testing.T) {
	p := Pod{Name: "p", CompatKey: "k1"}
	d := Pod{Name: "d", CompatKey: "k2"}
	if got := PairKey(p, d); got != "key:k1->k2" {
		t.Fatalf("keyed pair: %q", got)
	}
	d.CompatKey = ""
	if got := PairKey(p, d); got != "pod:p->d" {
		t.Fatalf("unkeyed pair: %q", got)
	}
}

func TestExclusions(t *testing.T) {
	t0 := time.Unix(1000, 0)
	e := NewExclusions(3, time.Minute)
	pair := "key:a->b"
	for i := 0; i < 2; i++ {
		if e.Record(pair, false, t0) {
			t.Fatalf("failure %d excluded too early", i+1)
		}
	}
	if e.Excluded(pair, t0) {
		t.Fatal("excluded before the threshold")
	}
	if !e.Record(pair, false, t0) {
		t.Fatal("third failure should start an exclusion")
	}
	if !e.Excluded(pair, t0.Add(59*time.Second)) || len(e.Active(t0)) != 1 {
		t.Fatal("pair should be excluded inside the window")
	}
	if e.Record(pair, false, t0.Add(time.Second)) {
		t.Fatal("failures during an exclusion must not restart it")
	}
	e.Record(pair, true, t0.Add(2*time.Second))
	if !e.Excluded(pair, t0.Add(3*time.Second)) {
		t.Fatal("a success during the window must not lift the exclusion")
	}
	if e.Excluded(pair, t0.Add(time.Minute)) || len(e.Active(t0.Add(time.Minute))) != 0 {
		t.Fatal("exclusion should end after the TTL")
	}
	// After expiry a success clears the state; failures count from zero again.
	e.Record(pair, true, t0.Add(2*time.Minute))
	e.Record(pair, false, t0.Add(2*time.Minute))
	e.Record(pair, true, t0.Add(2*time.Minute))
	e.Record(pair, false, t0.Add(2*time.Minute))
	e.Record(pair, false, t0.Add(2*time.Minute))
	if e.Excluded(pair, t0.Add(2*time.Minute)) {
		t.Fatal("a success should reset the consecutive count")
	}
}

func TestExclusionsDisabled(t *testing.T) {
	var nilE *Exclusions
	if nilE.Record("x", false, time.Now()) || nilE.Excluded("x", time.Now()) || nilE.Active(time.Now()) != nil {
		t.Fatal("nil tracker must be inert")
	}
	e := NewExclusions(0, time.Minute)
	for i := 0; i < 5; i++ {
		e.Record("x", false, time.Now())
	}
	if e.Excluded("x", time.Now()) {
		t.Fatal("threshold zero disables exclusion")
	}
}
