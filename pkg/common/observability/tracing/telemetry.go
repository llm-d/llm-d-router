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

package tracing

import (
	"context"
	"fmt"
	"os"
	"strconv"

	"github.com/go-logr/logr"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/exporters/stdout/stdouttrace"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.37.0"
	"go.opentelemetry.io/otel/trace"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/version"
)

type errorHandler struct {
	logger logr.Logger
}

func (h *errorHandler) Handle(err error) {
	h.logger.V(logging.DEFAULT).Error(err, "trace error occurred")
}

func InitTracing(ctx context.Context, logger logr.Logger, defaultServiceName string) (func(context.Context) error, error) {
	logger = logger.WithName("trace")
	loggerWrap := &errorHandler{logger: logger}

	// resource.New reports malformed OTEL_RESOURCE_ATTRIBUTES entries alongside a
	// resource holding everything it could parse. Tracing degrades to that resource
	// rather than stopping process startup.
	res, err := newResource(ctx, defaultServiceName)
	if err != nil {
		loggerWrap.Handle(fmt.Errorf("%s: %v", "build trace resource degraded", err))
	}

	traceExporter, err := initTraceExporter(ctx, logger)
	if err != nil {
		loggerWrap.Handle(fmt.Errorf("%s: %v", "init trace exporter failed", err))
		return nil, err
	}

	// Go SDK doesn't have an automatic sampler, handle manually
	samplerType, ok := os.LookupEnv("OTEL_TRACES_SAMPLER")
	if !ok {
		samplerType = "parentbased_traceidratio"
	}
	samplerARG, ok := os.LookupEnv("OTEL_TRACES_SAMPLER_ARG")
	if !ok {
		samplerARG = "0.1"
	}

	sampler := sdktrace.ParentBased(sdktrace.TraceIDRatioBased(0.1))
	if samplerType == "parentbased_traceidratio" {
		fraction, err := strconv.ParseFloat(samplerARG, 64)
		if err != nil {
			fraction = 0.1
		}

		sampler = sdktrace.ParentBased(sdktrace.TraceIDRatioBased(fraction))
	} else {
		loggerWrap.Handle(fmt.Errorf("unsupported sampler type: %s, fallback to parentbased_traceidratio with 0.1 Ratio", samplerType))
	}

	opt := []sdktrace.TracerProviderOption{
		sdktrace.WithBatcher(traceExporter),
		sdktrace.WithSampler(sampler),
		sdktrace.WithResource(res),
	}

	tracerProvider := sdktrace.NewTracerProvider(opt...)
	otel.SetTracerProvider(tracerProvider)
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(propagation.TraceContext{}, propagation.Baggage{}))
	otel.SetErrorHandler(loggerWrap)

	return tracerProvider.Shutdown, nil
}

// newResource builds the resource describing this process. Detectors are applied
// in order and later ones win, so OTEL_SERVICE_NAME and OTEL_RESOURCE_ATTRIBUTES
// override the built-in defaults.
//
// resource.Default is deliberately not merged in: it carries a newer semantic
// convention schema URL than the one used here, and merging conflicting schema
// URLs drops the schema URL from the result.
func newResource(ctx context.Context, defaultServiceName string) (*resource.Resource, error) {
	return resource.New(ctx,
		resource.WithSchemaURL(semconv.SchemaURL),
		resource.WithAttributes(
			semconv.ServiceNameKey.String(defaultServiceName),
			semconv.ServiceVersionKey.String(version.BuildRef),
		),
		resource.WithFromEnv(),
	)
}

// initTraceExporter create a SpanExporter
// support exporter type
// - console: export spans in console for development use case
// - otlp: export spans through gRPC to an opentelemetry collector
//
// Transport security, authentication headers and timeouts for the otlp exporter
// come from the standard OTEL_EXPORTER_OTLP_* environment variables. Passing any
// of them as an explicit option here would override the operator's setting, since
// the exporter applies explicit options after the environment. The one exception
// is the loopback fallback in localCollectorOptions, which applies only when the
// environment sets none of them.
func initTraceExporter(ctx context.Context, logger logr.Logger) (sdktrace.SpanExporter, error) {
	var traceExporter sdktrace.SpanExporter
	traceExporter, err := stdouttrace.New(stdouttrace.WithPrettyPrint())
	if err != nil {
		return nil, fmt.Errorf("failed to create stdouttrace exporter: %w", err)
	}

	exporterType, ok := os.LookupEnv("OTEL_TRACES_EXPORTER")
	if !ok {
		exporterType = "console"
	}

	logger.Info("init OTel trace exporter", "type", exporterType)
	if exporterType == "otlp" {
		traceExporter, err = otlptracegrpc.New(ctx, localCollectorOptions()...)
		if err != nil {
			return nil, fmt.Errorf("failed to create otlp-grcp exporter: %w", err)
		}
	}

	return traceExporter, nil
}

// otlpTransportEnv are the variables that select where spans are sent and how the
// connection is secured.
var otlpTransportEnv = []string{
	"OTEL_EXPORTER_OTLP_ENDPOINT",
	"OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
	"OTEL_EXPORTER_OTLP_INSECURE",
	"OTEL_EXPORTER_OTLP_TRACES_INSECURE",
	"OTEL_EXPORTER_OTLP_CERTIFICATE",
	"OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE",
}

// localCollectorOptions targets a plaintext collector on the loopback address when
// the environment configures no transport. The SDK's own default reaches the same
// host and port but negotiates TLS, which a local collector does not serve.
//
// The options are dropped as soon as any of the transport variables is set, so an
// operator's endpoint and its scheme still decide where spans go and how the
// connection is secured.
func localCollectorOptions() []otlptracegrpc.Option {
	for _, key := range otlpTransportEnv {
		if _, ok := os.LookupEnv(key); ok {
			return nil
		}
	}

	return []otlptracegrpc.Option{
		otlptracegrpc.WithEndpoint("localhost:4317"),
		otlptracegrpc.WithInsecure(),
	}
}

const instrumentationName = "llm-d-router"

// Tracer returns a tracer for the given instrumentation scope, defaulting to
// "llm-d-router". Build version and commit SHA are attached so every span in a
// trace carries consistent scope metadata.
func Tracer(scope ...string) trace.Tracer {
	name := instrumentationName
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
