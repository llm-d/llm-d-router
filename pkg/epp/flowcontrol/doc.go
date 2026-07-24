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

// Package flowcontrol wires the flow control system: the FlowController data plane and the
// FlowRegistry control plane.
//
// The package tree is panic-free, enforced by a lint rule. Statistics have a single owner per
// level, so there is no duplicated state for an invariant check to guard (see
// registry/managedqueue.go and queue/priorityqueue.go); caller- and plugin-supplied values are
// validated at the boundary and surfaced as errors; long-lived goroutines defer
// utilruntime.HandleCrashWithLogger so an unknown bug is logged with component context before the
// process exits. Recover-and-continue is not used: a goroutine that panicked mid-mutation cannot
// prove its state is consistent.
//
// If SafeQueue ever becomes an injectable extension point again, queue-reported stats become
// plugin output and accounting must move to booked-charge, settle-at-most-once validation at the
// managedQueue boundary.
package flowcontrol
