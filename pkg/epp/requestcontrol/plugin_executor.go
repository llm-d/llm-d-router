/*
Copyright 2025 The Kubernetes Authors.
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

package requestcontrol

import (
	"context"
	"errors"
	"fmt"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// executePluginsAsDAG executes DataProducer plugins as a DAG based on their dependencies asynchronously.
// So, a plugin is executed only after all its dependencies have been executed.
// If there is a cycle or any plugin fails with error, it returns an error.
func executePluginsAsDAG(ctx context.Context, plugins []fwkrc.DataProducer, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) error {
	return runProducers(ctx, scopeProducers(ctx, plugins, request, endpoints))
}

// producerInvocation is one DataProducer with the request and endpoints
// confined to its declarations.
type producerInvocation struct {
	plugin     fwkrc.DataProducer
	request    *fwksched.InferenceRequest
	endpoints  []fwksched.Endpoint
	violations *datalayer.Violations
}

// scopeProducers confines each producer to its declarations. Scoping copies
// the request, so it runs on the caller's goroutine rather than on one the
// timeout path may abandon while the director keeps writing request fields.
func scopeProducers(ctx context.Context, plugins []fwkrc.DataProducer, request *fwksched.InferenceRequest,
	endpoints []fwksched.Endpoint) []producerInvocation {
	logger := log.FromContext(ctx)
	invocations := make([]producerInvocation, len(plugins))
	for i, plugin := range plugins {
		scopedRequest, scopedEndpoints, violations := datalayer.ScopeInvocation(logger, fwkrc.DataProducerExtensionPoint, plugin, request, endpoints)
		invocations[i] = producerInvocation{plugin: plugin, request: scopedRequest, endpoints: scopedEndpoints, violations: violations}
	}
	return invocations
}

func runProducers(ctx context.Context, invocations []producerInvocation) error {
	for _, inv := range invocations {
		plugin := inv.plugin
		before := time.Now()
		err := plugin.Produce(ctx, inv.request, inv.endpoints)
		metrics.RecordPluginProcessingLatency(fwkrc.DataProducerExtensionPoint, plugin.TypedName().Type, plugin.TypedName().Name, time.Since(before))
		if err != nil {
			return fmt.Errorf("DataProducer %q failed: %w", plugin.TypedName().String(), err)
		}
		if err := inv.violations.Write(); err != nil {
			return fmt.Errorf("DataProducer %q failed: %w", plugin.TypedName().String(), err)
		}
	}
	return nil
}

// producerTimeout returns the producer's declared timeout when it implements
// TimeoutAwareProducer with a positive value, otherwise dataProducerTimeout.
func producerTimeout(p fwkrc.DataProducer) time.Duration {
	if tp, ok := p.(fwkrc.TimeoutAwareProducer); ok {
		if t := tp.ProduceTimeout(); t > 0 {
			return t
		}
	}
	return dataProducerTimeout
}

// dataProducerPluginsWithTimeout executes DataProducer plugins with a timeout.
// The child context is cancelled when the timeout fires so plugins can observe cancellation
// (e.g. abort outbound HTTP calls) and avoid committing state after the director has moved on.
func dataProducerPluginsWithTimeout(ctx context.Context, timeout time.Duration, plugins []fwkrc.DataProducer,
	request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) error {
	// The timeout path does not join the producer goroutine. Allocate the
	// sync.Map before launching it so any cancellation-aware producer finishing
	// a write cannot race with scheduling over lazy store initialization.
	request.InitializeAttributeStore()
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	invocations := scopeProducers(ctx, plugins, request, endpoints)
	errCh := make(chan error, 1)
	go func() {
		errCh <- runProducers(ctx, invocations)
	}()

	select {
	case err := <-errCh:
		return err
	case <-ctx.Done():
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return fmt.Errorf("DataProducer execution timed out: %w", ctx.Err())
		}
		return ctx.Err()
	}
}
