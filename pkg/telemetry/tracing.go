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

// Package telemetry exposes OpenTelemetry tracers for the kvcache and kvevents
// libraries. Tracer provider initialization is owned by the host application
// (pkg/common/observability/tracing); this package only resolves tracers from
// the global provider.
package telemetry

import (
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/llm-d/llm-d-router/version"
)

// InstrumentationName identifies this instrumentation library in traces.
const InstrumentationName = "llm-d-kv-cache"

// Tracer returns a tracer for the given instrumentation scope, defaulting to
// InstrumentationName. Build version and commit SHA are attached so every span
// in a trace carries consistent scope metadata. The host application's tracer
// provider determines the service name.
func Tracer(scope ...string) trace.Tracer {
	name := InstrumentationName
	if len(scope) > 0 && scope[0] != "" {
		name = scope[0]
	}
	return otel.Tracer(
		name,
		trace.WithInstrumentationVersion(version.BuildRef),
		trace.WithInstrumentationAttributes(
			attribute.String("commit-sha", version.CommitSHA),
		),
	)
}
