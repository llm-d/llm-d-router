package logging

import (
	"context"

	"github.com/go-logr/logr"
	uberzap "go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
)

var atomicLevel = uberzap.NewAtomicLevelAt(zapcore.InfoLevel)

func customLevelEncoder(l zapcore.Level, enc zapcore.PrimitiveArrayEncoder) {
	if l >= 0 {
		zapcore.LowercaseLevelEncoder(l, enc)
		return
	}

	switch l {
	case zapcore.Level(-1 * DEBUG):
		enc.AppendString("debug")
	case zapcore.Level(-1 * TRACE):
		enc.AppendString("trace")
	default:
		if l >= zapcore.Level(-1*VERBOSE) {
			enc.AppendString("info")
		} else {
			enc.AppendString("trace")
		}
	}
}

func InitLogging(verbosity int) {
	lvl := -1 * verbosity
	atomicLevel.SetLevel(zapcore.Level(int8(lvl)))

	config := uberzap.NewProductionEncoderConfig()
	config.EncodeLevel = customLevelEncoder

	logger := zap.New(
		zap.Level(atomicLevel),
		zap.RawZapOpts(uberzap.AddCaller()),
		zap.Encoder(zapcore.NewJSONEncoder(config)),
	)
	ctrl.SetLogger(logger)
}

func FromContext(ctx context.Context) logr.Logger {
	return log.FromContext(ctx)
}

func IntoContext(ctx context.Context, logger logr.Logger) context.Context {
	return log.IntoContext(ctx, logger)
}

func NewTestLogger() logr.Logger {
	return zap.New(
		zap.UseDevMode(true),
		zap.Level(uberzap.NewAtomicLevelAt(zapcore.Level(-1*TRACE))),
		zap.RawZapOpts(uberzap.AddCaller()),
	)
}
