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

package capacityledger

import (
	"sync"
	"time"
)

// axis is one resource of an endpoint. Admission compares the endpoint's state and a request's
// demand with the endpoint's capacity on every axis.
type axis uint8

const (
	// axisMemory is KV cache blocks. It is the only axis whose use grows between decisions.
	axisMemory axis = iota
	// axisStep is tokens committed to the engine's next step, against its per-step token budget:
	// prompts not yet prefilled plus one decode step for every running sequence.
	axisStep
	// axisSlots is concurrent sequences.
	axisSlots
	numAxes
)

var axisNames = [numAxes]string{"memory", "step", "slots"}

func (a axis) String() string { return axisNames[a] }

// vec is a quantity on every axis.
type vec [numAxes]int64

func (v vec) add(o vec) vec {
	for i := range v {
		v[i] += o[i]
	}
	return v
}

func (v vec) sub(o vec) vec {
	for i := range v {
		v[i] -= o[i]
	}
	return v
}

func (v vec) max(o vec) vec {
	for i := range v {
		v[i] = max(v[i], o[i])
	}
	return v
}

// account is an endpoint's booked state: the sum of the contributions its leases have recorded.
// added is the running total of the increases, which the state rule reads at each scrape: a
// release after the scrape, of a request the engine had already freed before it, must not cancel a
// booking the scrape never saw. apply and markScrape are its only writers and snapshot its only
// reader.
type account struct {
	mu            sync.Mutex
	booked        vec
	added         vec
	addedAtScrape vec
	scrapeTime    time.Time
}

// accountState is a consistent read of an account.
type accountState struct {
	booked        vec
	added         vec
	addedAtScrape vec
	// scrapeTime is the metrics UpdateTime addedAtScrape was recorded for; zero if never.
	scrapeTime time.Time
}

func (a *account) apply(delta vec) {
	if delta == (vec{}) {
		return
	}
	a.mu.Lock()
	a.booked = a.booked.add(delta)
	a.added = a.added.add(delta.max(vec{}))
	a.mu.Unlock()
}

// markScrape records the additions at the scrape whose metrics carry updateTime. A poll that did
// not advance updateTime is not a new scrape and is ignored.
func (a *account) markScrape(updateTime time.Time) {
	a.mu.Lock()
	if updateTime.After(a.scrapeTime) {
		a.addedAtScrape = a.added
		a.scrapeTime = updateTime
	}
	a.mu.Unlock()
}

func (a *account) snapshot() accountState {
	a.mu.Lock()
	defer a.mu.Unlock()
	return accountState{booked: a.booked, added: a.added,
		addedAtScrape: a.addedAtScrape, scrapeTime: a.scrapeTime}
}

// usedState is the state rule. On each axis, used is the engine's report plus the increases the
// router has booked since that report, and never less than the router's bookings. A booking made
// and released between two scrapes counts until the next scrape. With no record for this scrape,
// used is the larger of booked and scraped.
func usedState(s accountState, scraped vec, scrapeTime time.Time) vec {
	if scrapeTime.IsZero() || !s.scrapeTime.Equal(scrapeTime) {
		return s.booked.max(scraped)
	}
	return s.booked.max(scraped.add(s.added.sub(s.addedAtScrape)))
}
