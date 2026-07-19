/*
Copyright 2025 The Kubernetes Authors.

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

// Package registry provides the concrete implementation of the `contracts.FlowRegistry` interface.
//
// This package implements the flow control state machine. It separates the orchestration control plane from the
// request-processing data plane, and is composed of three core types:
//
//   - `FlowRegistry`: The top-level orchestrator and single source of truth. It manages the lifecycle of all flows and
//     priority bands, handling registration, garbage collection, and dynamic priority-band provisioning. It also
//     exposes a read-optimized, concurrent-safe data-plane view for the `controller` Processor.
//   - `priorityBand`: The runtime state for a single priority level, holding its managed queues and configuration.
//   - `managedQueue`: A stateful decorator around a SafeQueue. It is the fundamental unit of state,
//     responsible for atomically tracking statistics (e.g., length and byte size) and ensuring data consistency.
package registry
