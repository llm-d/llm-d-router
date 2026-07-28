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

package datalayer

import (
	"context"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// DataSource provides raw data to registered Extractors. Concrete variants
// are PollingDispatcher (poll-driven), NotificationSource (k8s-event-driven),
// and EndpointSource (lifecycle-event-driven).
type DataSource interface {
	plugin.Plugin
}

// Extractor transforms typed input T into endpoint attributes. T pins the
// dispatch payload:
//
//   - Polling extractors:      T = PollInput[D]      (paired with a PollingDispatcher)
//   - Notification extractors: T = NotificationEvent (also implement NotificationExtractor for GVK)
//   - Endpoint extractors:     T = EndpointEvent
type Extractor[T any] interface {
	plugin.Plugin
	Extract(ctx context.Context, in T) error
}

// PollInput pairs the typed poll payload with the endpoint being polled.
// Payload is the parser's result and is expected to be usable when Poll returns a nil error.
type PollInput[D any] struct {
	Payload  D
	Endpoint Endpoint
}

// StreamInput pairs the stream payload with the endpoint being monitored.
type StreamInput[D any] struct {
	Payload  D
	Endpoint Endpoint
}

// EndpointExtractor is the typed contract for endpoint-lifecycle extractors.
type EndpointExtractor = Extractor[EndpointEvent]

// PollingExtractor is the typed contract for poll-based extractors.
type PollingExtractor[T any] = Extractor[PollInput[T]]

// StreamingExtractor is the typed contract for stream-based extractors.
type StreamingExtractor[T any] = Extractor[StreamInput[T]]

// NotificationExtractor is the typed contract for k8s-event extractors.
// GVK identifies the kind this extractor handles; it must match the paired
// NotificationSource's GVK.
type NotificationExtractor interface {
	Extractor[NotificationEvent]
	GVK() schema.GroupVersionKind
}

// PollingDispatcher is the framework's contract for polling sources. The
// source owns its extractors and runs them with typed input each tick.
//
// Contract:
//   - Dispatch runs bound extractors in AppendExtractor-insertion order.
//   - Each Poll and each Extract step runs under its own timeout.
//   - Non-nil return = poll failure; per-extractor failures record
//     DataLayerExtractErrorsTotal and do NOT surface as the return error.
//   - AppendExtractor is a pure append; duplicate-Type detection is the caller's
//     responsibility (see runtime.Configure).
type PollingDispatcher interface {
	plugin.Plugin
	Dispatch(ctx context.Context, ep Endpoint) error
	AppendExtractor(ext plugin.Plugin) error
}

// StreamingDispatcher is the framework's contract for push/stream sources (e.g. ZMQ).
// The source owns its extractors, receives continuous message streams for an endpoint,
// and dispatches incoming payloads to its bound extractors in insertion order.
type StreamingDispatcher interface {
	plugin.Plugin
	Start(ctx context.Context, ep Endpoint) error
	AppendExtractor(ext plugin.Plugin) error
}

// EventType identifies the type of mutation that triggered a notification.
type EventType int

const (
	// EventAddOrUpdate is fired when a k8s object is created or updated.
	EventAddOrUpdate EventType = iota
	// EventDelete is fired when a k8s object is deleted.
	EventDelete
)

// NotificationEvent carries the event type and the affected object.
// Object is deep-copied by the framework core before delivery.
type NotificationEvent struct {
	Type   EventType
	Object *unstructured.Unstructured
}

// NotificationSource is an event-driven DataSource for a single k8s GVK.
type NotificationSource interface {
	DataSource
	// GVK returns the GroupVersionKind this source watches.
	GVK() schema.GroupVersionKind
	// Notify is called by the framework core when a mutation event fires.
	// The event object is already deep-copied.
	// Returns the event (possibly modified) for Runtime to dispatch to extractors.
	// Returns nil event to signal Runtime to skip extractor dispatch.
	// TODO: why accept event but return *event?
	Notify(ctx context.Context, event NotificationEvent) (*NotificationEvent, error)
}

// EndpointEvent carries an endpoint lifecycle event.
type EndpointEvent struct {
	Type     EventType
	Endpoint Endpoint
}

// EndpointSource is an event-driven DataSource driven by endpoint lifecycle changes.
type EndpointSource interface {
	DataSource
	// NotifyEndpoint is called by the Runtime on each endpoint lifecycle event.
	// Returns nil event to skip extractor dispatch.
	NotifyEndpoint(ctx context.Context, event EndpointEvent) (*EndpointEvent, error)
}
