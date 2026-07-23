/*
Copyright 2025 The llm-d Authors.

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

package kvevents

import (
	"context"
	"sync"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/kvcache/metrics"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// SubscriberManager manages multiple ZMQ subscribers, one per LLM engine.
type SubscriberManager struct {
	pool        *Pool
	subscribers map[string]*subscriberEntry
	mu          sync.RWMutex
}

// subscriberEntry represents a single subscriber and its cancellation.
type subscriberEntry struct {
	subscriber *zmqSubscriber
	cancel     context.CancelFunc
	endpoint   string
	// done is closed once the subscriber's goroutine has returned.
	done chan struct{}
}

// NewSubscriberManager creates a new subscriber manager.
func NewSubscriberManager(pool *Pool) *SubscriberManager {
	return &SubscriberManager{
		pool:        pool,
		subscribers: make(map[string]*subscriberEntry),
	}
}

// EnsureSubscriber ensures a subscriber exists for the given pod.
// If the subscriber already exists with the same endpoint, it's a no-op.
// If the endpoint changed, the old subscriber is removed and a new one is created.
func (sm *SubscriberManager) EnsureSubscriber(ctx context.Context, podIdentifier, endpoint, topicFilter string,
	remoteSocket bool,
) error {
	debugLogger := log.FromContext(ctx).V(logging.DEBUG)

	sm.mu.Lock()
	defer sm.mu.Unlock()

	// Check if subscriber already exists
	if entry, exists := sm.subscribers[podIdentifier]; exists {
		if entry.endpoint == endpoint {
			// Subscriber already exists with the same endpoint, nothing to do
			debugLogger.V(logging.TRACE).Info("Subscriber already exists", "podIdentifier", podIdentifier, "endpoint", endpoint)
			return nil
		}
		// Endpoint changed, remove old subscriber
		debugLogger.Info("Endpoint changed, removing old subscriber",
			"podIdentifier", podIdentifier,
			"oldEndpoint", entry.endpoint,
			"newEndpoint", endpoint)
		entry.cancel()
		delete(sm.subscribers, podIdentifier)
		// The replacement subscriber below reuses podIdentifier, so its series
		// are kept rather than cleaned up.
	}

	// Create new subscriber
	debugLogger.Info("Creating new subscriber", "podIdentifier", podIdentifier, "endpoint", endpoint)
	subscriber := newZMQSubscriber(sm.pool, podIdentifier, endpoint, topicFilter, remoteSocket)

	// Create a context and start subscriber
	subCtx, cancel := context.WithCancel(ctx)
	done := make(chan struct{})
	go func() {
		defer close(done)
		subscriber.Start(subCtx)
	}()

	// Update subscribers
	sm.subscribers[podIdentifier] = &subscriberEntry{
		subscriber: subscriber,
		cancel:     cancel,
		endpoint:   endpoint,
		done:       done,
	}
	metrics.SubscriberActive.Set(float64(len(sm.subscribers)))

	debugLogger.Info("Subscriber created and started", "podIdentifier", podIdentifier, "endpoint", endpoint)
	return nil
}

// RemoveSubscriber removes a subscriber for the given pod identifier.
func (sm *SubscriberManager) RemoveSubscriber(ctx context.Context, podIdentifier string) {
	debugLogger := log.FromContext(ctx).V(logging.DEBUG)

	sm.mu.Lock()
	defer sm.mu.Unlock()

	entry, exists := sm.subscribers[podIdentifier]
	if !exists {
		debugLogger.Info("Subscriber does not exist, nothing to remove", "podIdentifier", podIdentifier)
		return
	}

	debugLogger.Info("Removing subscriber", "podIdentifier", podIdentifier, "endpoint", entry.endpoint)
	entry.cancel()
	delete(sm.subscribers, podIdentifier)
	metrics.SubscriberActive.Set(float64(len(sm.subscribers)))
	cleanupSubscriberMetrics(podIdentifier, entry.done)
}

// cleanupSubscriberMetrics drops the per-pod series for a removed subscriber
// once its goroutine has exited. Cancellation is asynchronous, so cleaning up
// eagerly would let a final message or error increment resurrect the series.
func cleanupSubscriberMetrics(podIdentifier string, done <-chan struct{}) {
	go func() {
		<-done
		metrics.CleanupSubscriber(podIdentifier)
	}()
}

// Shutdown shuts down all subscribers.
func (sm *SubscriberManager) Shutdown(ctx context.Context) {
	debugLogger := log.FromContext(ctx).V(logging.DEBUG)
	debugLogger.Info("Shutting down subscriber manager")

	sm.mu.Lock()
	defer sm.mu.Unlock()

	for podIdentifier, entry := range sm.subscribers {
		debugLogger.Info("Shutting down subscriber", "podIdentifier", podIdentifier)
		entry.cancel()
		cleanupSubscriberMetrics(podIdentifier, entry.done)
	}

	sm.subscribers = make(map[string]*subscriberEntry)
	metrics.SubscriberActive.Set(0)
	debugLogger.Info("All subscribers shut down")
}

// GetActiveSubscribers returns the list of active pod identifiers and their endpoints.
//
//nolint:gocritic
func (sm *SubscriberManager) GetActiveSubscribers() ([]string, []string) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	identifiers := make([]string, 0, len(sm.subscribers))
	endpoints := make([]string, 0, len(sm.subscribers))
	for id, entry := range sm.subscribers {
		identifiers = append(identifiers, id)
		endpoints = append(endpoints, entry.endpoint)
	}
	return identifiers, endpoints
}
