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

	sampler, err := newSampler()
	if err != nil {
		loggerWrap.Handle(fmt.Errorf("trace sampler configuration degraded: %w", err))
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

// Sampler defaults. The OpenTelemetry specification defaults OTEL_TRACES_SAMPLER
// to parentbased_always_on; this package keeps a ratio instead, so a deployment
// that sets neither variable samples a tenth of its requests rather than all of
// them.
const (
	defaultSamplerType = "parentbased_traceidratio"
	defaultSamplerArg  = 0.1
)

// newSampler builds the sampler from OTEL_TRACES_SAMPLER and OTEL_TRACES_SAMPLER_ARG.
// The Go SDK has no built-in mapping from those variables to a Sampler, so the
// specification's values are mapped here.
//
// A rejected value is returned as an error alongside a usable sampler: sampling is
// a runtime knob, and refusing to start the process over a typo in it would be
// worse than running at the documented default.
//
// jaeger_remote and parentbased_jaeger_remote are reported as unsupported. They
// need go.opentelemetry.io/contrib/samplers/jaegerremote, which is not a dependency
// of this module.
func newSampler() (sdktrace.Sampler, error) {
	samplerType, ok := os.LookupEnv("OTEL_TRACES_SAMPLER")
	if !ok {
		samplerType = defaultSamplerType
	}

	switch samplerType {
	case "always_on":
		return sdktrace.AlwaysSample(), nil
	case "always_off":
		return sdktrace.NeverSample(), nil
	case "parentbased_always_on":
		return sdktrace.ParentBased(sdktrace.AlwaysSample()), nil
	case "parentbased_always_off":
		return sdktrace.ParentBased(sdktrace.NeverSample()), nil
	case "traceidratio":
		fraction, err := samplerFraction()
		return sdktrace.TraceIDRatioBased(fraction), err
	case "parentbased_traceidratio":
		fraction, err := samplerFraction()
		return sdktrace.ParentBased(sdktrace.TraceIDRatioBased(fraction)), err
	default:
		return sdktrace.ParentBased(sdktrace.TraceIDRatioBased(defaultSamplerArg)),
			fmt.Errorf("unsupported OTEL_TRACES_SAMPLER %q, falling back to %s at %v", samplerType, defaultSamplerType, defaultSamplerArg)
	}
}

// samplerFraction reads the ratio for the traceidratio samplers, and is not consulted
// for the samplers that take no argument. The returned fraction is always usable; an
// error accompanies it when the configured value was rejected.
//
// The range check runs before the SDK sees the value. TraceIDRatioBased clamps a
// fraction outside [0, 1] to always-on or always-off, so an operator who writes 10
// meaning ten percent would otherwise sample everything with no diagnostic.
func samplerFraction() (float64, error) {
	arg, ok := os.LookupEnv("OTEL_TRACES_SAMPLER_ARG")
	if !ok {
		return defaultSamplerArg, nil
	}

	fraction, err := strconv.ParseFloat(arg, 64)
	if err != nil {
		return defaultSamplerArg, fmt.Errorf("invalid OTEL_TRACES_SAMPLER_ARG %q, falling back to %v: %w", arg, defaultSamplerArg, err)
	}
	if fraction < 0 || fraction > 1 {
		return defaultSamplerArg, fmt.Errorf("OTEL_TRACES_SAMPLER_ARG %v outside [0, 1], falling back to %v", fraction, defaultSamplerArg)
	}

	return fraction, nil
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
