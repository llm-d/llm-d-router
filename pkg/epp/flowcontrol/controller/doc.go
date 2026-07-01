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

// Package controller contains the implementation of the FlowController engine.
//
// # Overview
//
// The FlowController is the central processing engine of the Flow Control layer. It manages the lifecycle of all
// incoming requests from initial submission to a terminal outcome (dispatch, rejection, or eviction).
//
// # Architecture
//
// The engine is split into two cooperating roles:
//
//   - The FlowController: The public-facing API of the system. Each call to EnqueueAndWait runs on its own (caller)
//     goroutine: it acquires a flow lease from the contracts.FlowRegistry, hands the request to the Processor, and
//     blocks until the request is finalized. It owns asynchronous finalization driven by the request Context
//     (TTL/cancellation) and queue occupancy metrics.
//   - The internal.Processor (Worker): A single, stateful, single-goroutine actor that owns the request data plane. It
//     runs the dispatch loop, performs capacity checks, sweeps externally finalized items, and finalizes requests
//     synchronously (dispatch, capacity rejection, shutdown).
//
// # Concurrency Model
//
// The FlowController is designed to be highly concurrent and thread-safe.
//
//   - EnqueueAndWait: Can be called concurrently by many goroutines, each handing its item to the single Processor over
//     a buffered channel.
//   - Single-Writer Actor: Routing all state mutations through the Processor's single Run goroutine makes complex
//     transactions (such as capacity checks) inherently atomic without coarse-grained locks.
//
// # Request Lifecycle and Ownership
//
// A request (represented internally as a FlowItem) has a lifecycle managed cooperatively by the Controller and a
// Processor. Defining ownership is critical for ensuring an item is finalized exactly once.
//
//  1. Submission (Controller): The Controller attempts to hand off the item to a Processor.
//  2. Handoff:
//     - Success: Ownership transfers to the Processor, which is now responsible for Finalization.
//     - Failure: Ownership remains with the Controller, which must Finalize the item.
//  3. Processing (Processor): The Processor enqueues, manages, and eventually dispatches or rejects the item.
//  4. Finalization: The terminal outcome is set. This can happen:
//     - Synchronously: The Processor determines the outcome (e.g., Dispatch, Capacity Rejection).
//     - Asynchronously: The Controller observes the request's Context expiry (TTL/Cancellation) and calls Finalize.
//
// The FlowItem uses atomic operations to safely coordinate the Finalization state across goroutines.
package controller
