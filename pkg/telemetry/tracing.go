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

// Package telemetry provides OpenTelemetry tracing initialization and utilities
// for distributed tracing across llm-d components.
package telemetry

import (
	"context"
	"fmt"
	"os"
	"strconv"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.24.0"
	"go.opentelemetry.io/otel/trace"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/routing"
)

const (
	defaultServiceName = "llm-d-inference-scheduler"

	// instrumentationName identifies this instrumentation library in traces.
	instrumentationName = "llm-d-inference-scheduler"
)

// InitTracing initializes OpenTelemetry tracing with OTLP exporter.
// Configuration is done via environment variables:
// - OTEL_SERVICE_NAME: Service name for tracing (default: llm-d-inference-scheduler)
// - OTEL_EXPORTER_OTLP_ENDPOINT: OTLP collector endpoint (default: http://localhost:4317)
// - OTEL_TRACES_SAMPLER: Sampling strategy (default: parentbased_traceidratio)
// - OTEL_TRACES_SAMPLER_ARG: Sampling ratio (default: 0.1 for 10%)
func InitTracing(ctx context.Context) (func(context.Context) error, error) {
	logger := log.FromContext(ctx)

	// Get service name from environment, fallback to default
	serviceName := os.Getenv("OTEL_SERVICE_NAME")
	if serviceName == "" {
		serviceName = defaultServiceName
	}

	// Get OTLP endpoint from environment
	endpoint := os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
	if endpoint == "" {
		endpoint = "localhost:4317"
	}

	// Strip http:// or https:// prefix if present
	// otlptracegrpc.WithEndpoint() expects host:port only
	endpoint = routing.StripScheme(endpoint)

	logger.Info("Initializing OpenTelemetry tracing", "endpoint", endpoint, "service", serviceName)

	// Create OTLP trace exporter
	exporter, err := otlptracegrpc.New(ctx,
		otlptracegrpc.WithEndpoint(endpoint),
		otlptracegrpc.WithInsecure(), // Use WithTLSCredentials() in production
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create OTLP trace exporter: %w", err)
	}

	// Create resource with service name
	res, err := resource.New(ctx,
		resource.WithAttributes(
			semconv.ServiceName(serviceName),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create resource: %w", err)
	}

	// Get sampling ratio from environment, fallback to default
	samplingRatio := 0.1 // default 10% sampling
	if arg := os.Getenv("OTEL_TRACES_SAMPLER_ARG"); arg != "" {
		if ratio, err := strconv.ParseFloat(arg, 64); err == nil && ratio >= 0.0 && ratio <= 1.0 {
			samplingRatio = ratio
		} else {
			logger.Info("Invalid OTEL_TRACES_SAMPLER_ARG, using default", "arg", arg, "default", samplingRatio)
		}
	}

	logger.Info("Configuring trace sampling", "ratio", samplingRatio)

	// Create trace provider with parent-based sampling
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exporter),
		sdktrace.WithResource(res),
		sdktrace.WithSampler(sdktrace.ParentBased(sdktrace.TraceIDRatioBased(samplingRatio))),
	)

	// Set global trace provider
	otel.SetTracerProvider(tp)

	// Set W3C trace context propagator
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{},
		propagation.Baggage{},
	))

	logger.Info("OpenTelemetry tracing initialized successfully")

	// Return shutdown function
	return tp.Shutdown, nil
}

// Tracer returns a tracer for the inference scheduler.
// The tracer is identified by the instrumentation library name, which is
// distinct from the service name set during InitTracing().
func Tracer() trace.Tracer {
	return otel.Tracer(instrumentationName)
}
