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

package groupsched

import "sync"

// THE EVENT COLLECTOR — the test-only replacement for the deleted in-memory ring.
//
// Production keeps no copy of the event stream (the adapter's V(DEBUG) log is the forensic
// record); a test that needs to read the events installs this collector on the scheduler's
// onEvent hook, which is exactly the stream the log writes.
type eventCollector struct {
	mu  sync.Mutex
	evs []Event
}

func (c *eventCollector) collect(ev Event) {
	c.mu.Lock()
	c.evs = append(c.evs, ev)
	c.mu.Unlock()
}

// events returns a copy of everything collected so far (newest last).
func (c *eventCollector) events() []Event {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]Event(nil), c.evs...)
}

// boundTo is how many live seat bindings name one engine: the invariant "bindings == seats" is
// checked against the core itself, not carried in a dump.
func boundTo(s *Scheduler, engine string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	n := 0
	for _, e := range s.bindings {
		if e == engine {
			n++
		}
	}
	return n
}

// newEventCollector installs a collector on a scheduler and returns it.
func newEventCollector(s *Scheduler) *eventCollector {
	c := &eventCollector{}
	s.onEvent = c.collect
	return c
}
