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

// Package benchmark load-tests the Flow Control layer. It contains two families of benchmarks that
// trade off isolation against realism.
//
// # Throughput microbenchmarks (mock detector)
//
// PerformanceMatrix, TopologyChurn, and MassCancellation isolate the FlowController's CPU cost
// using a mock SaturationDetector and a synchronous, sleep-free pipeline. Simulating downstream
// inference latency with sleeps would park goroutines and measure Go scheduler overhead rather
// than computational throughput, so the harness avoids it:
//
//  1. Intentional Backpressure (W > L): Ingress Concurrency (W) is driven higher than the Capacity
//     Limit (L), forcing requests to queue.
//  2. Strict Capacity Checking: the mock SaturationDetector atomically grants capacity only when
//     evaluated, preventing races where dispatch outruns clients.
//  3. Immediate Draining: when a client unblocks it immediately frees its capacity slot, triggering
//     the next dispatch and keeping the system active.
//  4. Consistent Queue Depth: a fixed, deep queue forces continuous evaluation of fairness and
//     ordering policies at limits, isolating CPU performance.
//
// # Full-path benchmark (real components)
//
// FullPath measures the complete data path with the REAL InFlightLoadProducer and concurrency
// SaturationDetector wired into the FlowController, capturing the cost of the detector's
// per-dispatch DynamicAttribute reads against live producer state. It complements the
// microbenchmarks: they isolate flow-control CPU; it prices realistic saturation detection.
// Producer-level leak correctness is covered by the inflightload producer's own unit tests, not
// here.
//
// # Interpreting Metrics in b.RunParallel
//
// In a highly concurrent queuing system, standard Go benchmark metrics require care to interpret:
//
//  1. ns/op (system-wide amortized time): because these use b.RunParallel, ns/op represents inverse
//     throughput, not latency. The d/s custom metric converts it to dispatches per second.
//  2. ops: one "op" is the complete lifecycle of a single simulated request: ingress, queuing,
//     policy evaluation, and egress.
//  3. allocs/op and B/op (GC pressure): high allocations per request mean the garbage collector
//     thrashes under load, causing latency jitter.
//  4. Saturated Coordinates (W > L): when Concurrency exceeds Capacity, EnqueueAndWait blocks;
//     because capacity is released immediately upon dispatch, an "op" is governed by the Flow
//     Control layer's CPU overhead.
//
// # Custom Metrics Reported
//
//   - d/s:       (Dispatches/sec) the primary throughput metric (PerformanceMatrix, TopologyChurn,
//     FullPath).
//   - r/s:       (Rejects/sec) rate of requests rejected due to capacity or timeouts.
//   - errors:    total unexpected runtime errors encountered.
//   - zombies/s: rate of requests hitting context deadlines/TTLs (MassCancellation).
package benchmark
