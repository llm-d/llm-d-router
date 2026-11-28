/*
Copyright 2026 The Kubernetes Authors.

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

package scheduling

import (
	"context"
	"sync/atomic"

	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// ReloadableScheduler delegates each scheduling operation to one scheduler generation.
type ReloadableScheduler struct {
	active atomic.Pointer[activeScheduler]
}

type activeScheduler struct {
	scheduler  *Scheduler
	generation uint64
}

func NewReloadableScheduler(initial *Scheduler) *ReloadableScheduler {
	s := &ReloadableScheduler{}
	s.active.Store(&activeScheduler{scheduler: initial, generation: 1})
	return s
}

func (s *ReloadableScheduler) Schedule(ctx context.Context, request *fwksched.InferenceRequest, candidateEndpoints []fwksched.Endpoint) (*fwksched.SchedulingResult, error) {
	return s.active.Load().scheduler.Schedule(ctx, request, candidateEndpoints)
}

// Replace publishes scheduler as the next active generation.
func (s *ReloadableScheduler) Replace(scheduler *Scheduler) uint64 {
	for {
		current := s.active.Load()
		next := &activeScheduler{scheduler: scheduler, generation: current.generation + 1}
		if s.active.CompareAndSwap(current, next) {
			return next.generation
		}
	}
}

// Generation returns the active scheduler generation.
func (s *ReloadableScheduler) Generation() uint64 {
	return s.active.Load().generation
}
