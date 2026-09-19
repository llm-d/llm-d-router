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

package interturnlatency

import (
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestSessionTracker_ObserveGap(t *testing.T) {
	t.Parallel()

	tracker := newSessionTracker(10, time.Hour)
	base := time.Unix(1000, 0)

	_, ok := tracker.observe("s1", base)
	assert.False(t, ok, "first sighting yields no gap")

	gap, ok := tracker.observe("s1", base.Add(30*time.Second))
	assert.True(t, ok)
	assert.Equal(t, 30*time.Second, gap)

	_, ok = tracker.observe("s1", base.Add(30*time.Second).Add(2*time.Hour))
	assert.False(t, ok, "gap beyond maxIdle is a new session")
}

func TestSessionTracker_TouchMovesGapOrigin(t *testing.T) {
	t.Parallel()

	tracker := newSessionTracker(10, time.Hour)
	base := time.Unix(1000, 0)

	tracker.observe("s1", base)
	// Response completion at base+20s: the next request's gap measures idle
	// time from completion, not from the prior arrival.
	tracker.touch("s1", base.Add(20*time.Second))

	gap, ok := tracker.observe("s1", base.Add(50*time.Second))
	assert.True(t, ok)
	assert.Equal(t, 30*time.Second, gap)
}

func TestSessionTracker_SweepDropsIdleSessions(t *testing.T) {
	t.Parallel()

	tracker := newSessionTracker(3, time.Minute)
	base := time.Unix(1000, 0)

	for i := range 3 {
		tracker.touch(fmt.Sprintf("old-%d", i), base)
	}
	assert.Equal(t, 3, tracker.size())

	// Admitting a new session past the cap sweeps entries idle beyond maxIdle.
	tracker.touch("new", base.Add(2*time.Minute))
	assert.Equal(t, 1, tracker.size())
}
