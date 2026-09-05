package logging

import (
	"os"

	"github.com/go-logr/logr"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	crzap "sigs.k8s.io/controller-runtime/pkg/log/zap"
)

const (
	otelTimestampKey      = "timestamp"
	otelSeverityTextKey   = "severity_text"
	otelLoggerKey         = "logger"
	otelCallerKey         = "caller"
	otelBodyKey           = "body"
	otelStacktraceKey     = "stacktrace"
	otelServiceNameKey    = "service.name"
	otelSeverityNumberKey = "severity_number"
	otelServiceNameEnv    = "OTEL_SERVICE_NAME"

	severityTrace  = "TRACE"
	severityDebug  = "DEBUG"
	severityInfo   = "INFO"
	severityWarn   = "WARN"
	severityError  = "ERROR"
	severityDPanic = "DPANIC"
	severityPanic  = "PANIC"
	severityFatal  = "FATAL"

	severityNumberTrace  = 1
	severityNumberDebug  = 5
	severityNumberInfo   = 9
	severityNumberWarn   = 13
	severityNumberError  = 17
	severityNumberDPanic = 18
	severityNumberPanic  = 19
	severityNumberFatal  = 21
)

// NewLogger returns a logger with OpenTelemetry field names and severity
// fields. Additional options customize the logger for each service.
func NewLogger(serviceName string, opts ...crzap.Opts) logr.Logger {
	defaultOpts := []crzap.Opts{
		crzap.WriteTo(os.Stdout),
		crzap.Encoder(zapcore.NewJSONEncoder(EncoderConfig())),
		crzap.RawZapOpts(
			zap.WrapCore(WrapCore),
			zap.Fields(zap.String(otelServiceNameKey, ServiceName(serviceName))),
		),
	}
	return crzap.New(append(defaultOpts, opts...)...)
}

// EncoderConfig returns a zap encoder config that emits OTel Logs Data Model
// field names on stdout JSON records.
func EncoderConfig() zapcore.EncoderConfig {
	config := zap.NewProductionEncoderConfig()
	config.TimeKey = otelTimestampKey
	config.LevelKey = otelSeverityTextKey
	config.NameKey = otelLoggerKey
	config.CallerKey = otelCallerKey
	config.MessageKey = otelBodyKey
	config.StacktraceKey = otelStacktraceKey
	config.EncodeTime = zapcore.RFC3339NanoTimeEncoder
	config.EncodeLevel = LevelEncoder
	return config
}

// SeverityText maps zap / logr verbosity levels to OTel severity_text.
func SeverityText(l zapcore.Level) string {
	if l >= 0 {
		switch {
		case l >= zapcore.FatalLevel:
			return severityFatal
		case l >= zapcore.PanicLevel:
			return severityPanic
		case l >= zapcore.DPanicLevel:
			return severityDPanic
		case l >= zapcore.ErrorLevel:
			return severityError
		case l >= zapcore.WarnLevel:
			return severityWarn
		default:
			return severityInfo
		}
	}

	switch l {
	case zapcore.Level(-1 * DEBUG):
		return severityDebug
	case zapcore.Level(-1 * TRACE):
		return severityTrace
	default:
		if l >= zapcore.Level(-1*VERBOSE) {
			return severityInfo
		}
		return severityTrace
	}
}

// SeverityNumber maps zap / logr verbosity levels to OTel severity_number.
func SeverityNumber(l zapcore.Level) int {
	switch SeverityText(l) {
	case severityFatal:
		return severityNumberFatal
	case severityPanic:
		return severityNumberPanic
	case severityDPanic:
		return severityNumberDPanic
	case severityError:
		return severityNumberError
	case severityWarn:
		return severityNumberWarn
	case severityDebug:
		return severityNumberDebug
	case severityTrace:
		return severityNumberTrace
	default:
		return severityNumberInfo
	}
}

// ServiceName returns OTEL_SERVICE_NAME or fallback.
func ServiceName(fallback string) string {
	if name := os.Getenv(otelServiceNameEnv); name != "" {
		return name
	}
	return fallback
}

// WrapCore adds severity_number to every log record.
func WrapCore(c zapcore.Core) zapcore.Core {
	return &otelCore{Core: c}
}

type otelCore struct {
	zapcore.Core
}

func (c *otelCore) With(fields []zapcore.Field) zapcore.Core {
	return &otelCore{Core: c.Core.With(fields)}
}

func (c *otelCore) Check(ent zapcore.Entry, ce *zapcore.CheckedEntry) *zapcore.CheckedEntry {
	if c.Enabled(ent.Level) {
		return ce.AddCore(ent, c)
	}
	return ce
}

func (c *otelCore) Write(ent zapcore.Entry, fields []zapcore.Field) error {
	fields = append(fields, zap.Int(otelSeverityNumberKey, SeverityNumber(ent.Level)))
	return c.Core.Write(ent, fields)
}
