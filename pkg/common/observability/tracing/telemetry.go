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
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/go-logr/logr"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/exporters/stdout/stdouttrace"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.41.0"
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

	exporterType, err := traceExporterType()
	if err != nil {
		loggerWrap.Handle(fmt.Errorf("trace exporter configuration degraded: %w", err))
	}

	sampler, err := newSampler()
	if err != nil {
		loggerWrap.Handle(fmt.Errorf("trace sampler configuration degraded: %w", err))
	}

	opt := []sdktrace.TracerProviderOption{
		sdktrace.WithSampler(sampler),
		sdktrace.WithResource(res),
	}

	// "none" registers no span processor at all. Spans are still created and
	// propagated, so instrumented code and context propagation are unaffected.
	if exporterType == exporterTypeNone {
		logger.Info("init OTel trace exporter", "type", exporterType)
	} else {
		traceExporter, err := newTraceExporter(ctx, loggerWrap, exporterType)
		if err != nil {
			loggerWrap.Handle(fmt.Errorf("%s: %v", "init trace exporter failed", err))
			return nil, err
		}
		opt = append(opt, sdktrace.WithBatcher(traceExporter))
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

// The exporter types OTEL_TRACES_EXPORTER selects between. The default sends spans
// to a collector: console pretty print writes a multi-line JSON block per span onto
// the same stream as the structured logs, so it is selected explicitly.
const (
	exporterTypeOTLP    = "otlp"
	exporterTypeConsole = "console"
	exporterTypeNone    = "none"

	defaultExporterType = exporterTypeOTLP
)

// traceExporterType resolves OTEL_TRACES_EXPORTER to one of the types
// newTraceExporter builds:
//
//   - otlp: export spans through gRPC to an opentelemetry collector
//   - console: pretty print spans on stdout, for development
//   - none: create spans but export nothing
//
// An unrecognised value is returned as an error alongside the default type, so a
// typo is reported rather than quietly selecting an exporter the operator did not
// ask for. The exporter is not worth failing startup over.
func traceExporterType() (string, error) {
	exporterType, ok := os.LookupEnv("OTEL_TRACES_EXPORTER")
	if !ok {
		return defaultExporterType, nil
	}

	switch exporterType {
	case exporterTypeOTLP, exporterTypeConsole, exporterTypeNone:
		return exporterType, nil
	default:
		return defaultExporterType, fmt.Errorf("unsupported OTEL_TRACES_EXPORTER %q, falling back to %s", exporterType, defaultExporterType)
	}
}

// The OTLP transports OTEL_EXPORTER_OTLP_PROTOCOL selects between. http/json is not
// among them: the Go SDK ships no JSON encoder for OTLP, so it is unsupported like any
// value outside this set.
const (
	protocolGRPC = "grpc"
	protocolHTTP = "http/protobuf"

	defaultProtocol = protocolGRPC
)

// otlpProtocolEnv are the variables that select the OTLP transport, per-signal first.
var otlpProtocolEnv = []string{
	"OTEL_EXPORTER_OTLP_TRACES_PROTOCOL",
	"OTEL_EXPORTER_OTLP_PROTOCOL",
}

// otlpProtocol resolves the OTLP transport. The per-signal variable wins when both name
// a supported one.
//
// The specification governs the rest: an empty value reads the same as unset, enum
// values match case-insensitively, and a value the implementation does not recognise is
// reported and then ignored, leaving the next variable to apply. gRPC applies when none
// of them resolves.
//
// Ignored values are returned alongside the protocol that won, so a misconfiguration is
// still reported even when a later variable supplies a usable transport.
func otlpProtocol() (string, error) {
	var ignored []error

	for _, key := range otlpProtocolEnv {
		protocol := strings.ToLower(strings.TrimSpace(os.Getenv(key)))
		if protocol == "" {
			continue
		}

		switch protocol {
		case protocolGRPC, protocolHTTP:
			return protocol, errors.Join(ignored...)
		default:
			ignored = append(ignored, fmt.Errorf("unsupported %s %q, ignored", key, protocol))
		}
	}

	if len(ignored) > 0 {
		ignored = append(ignored, fmt.Errorf("no supported OTLP protocol configured, falling back to %s", defaultProtocol))
	}

	return defaultProtocol, errors.Join(ignored...)
}

// newTraceExporter builds the exporter named by exporterType, which traceExporterType
// has already narrowed. Exactly one exporter is constructed; exporterTypeNone builds
// none and is handled by the caller.
//
// Transport security, authentication headers and timeouts for the otlp exporter come
// from the standard OTEL_EXPORTER_OTLP_* environment variables. Passing any of them as
// an explicit option here would override the operator's setting, since the exporter
// applies explicit options after the environment. The one exception is the loopback
// fallback, which applies only when the environment sets none of them.
func newTraceExporter(ctx context.Context, h *errorHandler, exporterType string) (sdktrace.SpanExporter, error) {
	if exporterType == exporterTypeConsole {
		h.logger.Info("init OTel trace exporter", "type", exporterType)

		traceExporter, err := stdouttrace.New(stdouttrace.WithPrettyPrint())
		if err != nil {
			return nil, fmt.Errorf("failed to create stdouttrace exporter: %w", err)
		}
		return traceExporter, nil
	}

	protocol, err := otlpProtocol()
	if err != nil {
		h.Handle(fmt.Errorf("trace exporter protocol configuration degraded: %w", err))
	}
	h.logger.Info("init OTel trace exporter", "type", exporterType, "protocol", protocol)

	if protocol == protocolHTTP {
		traceExporter, err := otlptracehttp.New(ctx, localHTTPCollectorOptions()...)
		if err != nil {
			return nil, fmt.Errorf("failed to create otlp-http exporter: %w", err)
		}
		return traceExporter, nil
	}

	traceExporter, err := otlptracegrpc.New(ctx, localGRPCCollectorOptions()...)
	if err != nil {
		return nil, fmt.Errorf("failed to create otlp-grpc exporter: %w", err)
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

// The loopback ports the two OTLP transports serve on.
const (
	localGRPCEndpoint = "localhost:4317"
	localHTTPEndpoint = "localhost:4318"
)

// transportConfigured reports whether the environment says where spans go or how the
// connection is secured. The loopback fallback applies only when it says neither, so
// an operator's endpoint and its scheme still decide both.
func transportConfigured() bool {
	for _, key := range otlpTransportEnv {
		if _, ok := os.LookupEnv(key); ok {
			return true
		}
	}

	return false
}

// localGRPCCollectorOptions targets a plaintext collector on the loopback address when
// the environment configures no transport. The SDK's own default reaches the same host
// and port but negotiates TLS, which a local collector does not serve.
func localGRPCCollectorOptions() []otlptracegrpc.Option {
	if transportConfigured() {
		return nil
	}

	return []otlptracegrpc.Option{
		otlptracegrpc.WithEndpoint(localGRPCEndpoint),
		otlptracegrpc.WithInsecure(),
	}
}

// localHTTPCollectorOptions is localGRPCCollectorOptions for the HTTP transport, which
// serves on its own port and takes an unrelated option type.
func localHTTPCollectorOptions() []otlptracehttp.Option {
	if transportConfigured() {
		return nil
	}

	return []otlptracehttp.Option{
		otlptracehttp.WithEndpoint(localHTTPEndpoint),
		otlptracehttp.WithInsecure(),
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
