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
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	attrcapacity "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/capacity"
)

func TestEndpointCapacityClone(t *testing.T) {
	require.Nil(t, (*attrcapacity.EndpointCapacity)(nil).Clone())
	orig := &attrcapacity.EndpointCapacity{
		Memory:      attrcapacity.Axis{Capacity: 100, Used: 50},
		BlockSize:   16,
		Eligibility: attrcapacity.Eligible,
	}
	cp := orig.Clone().(*attrcapacity.EndpointCapacity)
	require.Equal(t, orig, cp)
	cp.Memory.Used = 80
	require.Equal(t, int64(50), orig.Memory.Used)
}

func TestVec(t *testing.T) {
	a, b := vec{1, 5, 3}, vec{4, 2, 3}
	require.Equal(t, vec{5, 7, 6}, a.add(b))
	require.Equal(t, vec{-3, 3, 0}, a.sub(b))
	require.Equal(t, vec{4, 5, 3}, a.max(b))
	require.Equal(t, []string{"memory", "step", "slots"},
		[]string{axisMemory.String(), axisStep.String(), axisSlots.String()})
}

func TestAccount(t *testing.T) {
	var a account
	a.apply(vec{})
	require.Equal(t, vec{}, a.snapshot().booked, "a zero delta changes nothing")

	a.apply(vec{3, 100, 1})
	a.apply(vec{2, -100, 1})
	s := a.snapshot()
	require.Equal(t, vec{5, 0, 2}, s.booked)
	require.Equal(t, vec{5, 100, 2}, s.added)
}

func TestUsedState(t *testing.T) {
	var a account
	a.apply(vec{7, 50, 3})
	scrape := testStart
	require.Equal(t, vec{10, 50, 3}, usedState(a.snapshot(), vec{10, 0, 2}, scrape),
		"with no bookings recorded at the scrape, each axis takes the larger of booked and scraped")

	a.markScrape(scrape)
	a.apply(vec{4, 20, 1})
	require.Equal(t, vec{14, 70, 4}, usedState(a.snapshot(), vec{10, 0, 2}, scrape),
		"scraped plus the bookings since the scrape, and never less than booked")

	a.apply(vec{-11, -70, -4})
	require.Equal(t, vec{14, 20, 3}, usedState(a.snapshot(), vec{10, 0, 2}, scrape),
		"a release after the scrape, possibly of a request the engine freed before it, does not cancel the booking the scrape never saw")

	a.markScrape(scrape)
	require.Equal(t, vec{14, 20, 3}, usedState(a.snapshot(), vec{10, 0, 2}, scrape),
		"a poll that did not advance the scrape time is not a new scrape")
	a.markScrape(scrape.Add(time.Millisecond))
	require.Equal(t, vec{10, 0, 2}, usedState(a.snapshot(), vec{10, 0, 2}, scrape.Add(time.Millisecond)),
		"a new scrape starts a new count")

	require.Equal(t, vec{10, 0, 2}, usedState(a.snapshot(), vec{10, 0, 2}, scrape.Add(time.Second)),
		"a scrape without a recorded booking falls back to the larger of the two")
}
