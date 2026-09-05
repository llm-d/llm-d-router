package logging

import (
	"bytes"
	"encoding/json"
	"testing"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	crzap "sigs.k8s.io/controller-runtime/pkg/log/zap"
)

func TestJSONRecordUsesOTelFields(t *testing.T) {
	var buf bytes.Buffer
	core := WrapCore(zapcore.NewCore(zapcore.NewJSONEncoder(EncoderConfig()), zapcore.AddSync(&buf), zapcore.InfoLevel))
	zl := zap.New(core)
	zl.Info("request assembled")
	if err := zl.Sync(); err != nil {
		t.Fatal(err)
	}

	var rec map[string]any
	if err := json.Unmarshal(buf.Bytes(), &rec); err != nil {
		t.Fatal(err)
	}
	if rec["body"] != "request assembled" {
		t.Errorf("body = %v, want request assembled", rec["body"])
	}
	if rec["severity_text"] != "INFO" {
		t.Errorf("severity_text = %v, want INFO", rec["severity_text"])
	}
	if rec["severity_number"] != float64(9) {
		t.Errorf("severity_number = %v, want 9", rec["severity_number"])
	}
	if rec["timestamp"] == nil || rec["timestamp"] == "" {
		t.Error("timestamp missing")
	}
}

func TestNewLoggerSeparatesBodyAndHTTPBody(t *testing.T) {
	var buf bytes.Buffer
	logger := NewLogger(
		"test-service",
		crzap.WriteTo(&buf),
		crzap.Level(zapcore.InfoLevel),
	)
	logger.Info("request body", "http_body", `{"prompt":"hello"}`)

	var rec map[string]any
	if err := json.Unmarshal(buf.Bytes(), &rec); err != nil {
		t.Fatal(err)
	}
	if rec["body"] != "request body" {
		t.Errorf("body = %v, want request body", rec["body"])
	}
	if rec["http_body"] != `{"prompt":"hello"}` {
		t.Errorf("http_body = %v, want request payload", rec["http_body"])
	}
	if rec["service.name"] != "test-service" {
		t.Errorf("service.name = %v, want test-service", rec["service.name"])
	}
}

func TestSeverityNumberPreservesZapFatalLevels(t *testing.T) {
	tests := []struct {
		name     string
		level    zapcore.Level
		expected int
	}{
		{name: "dpanic", level: zapcore.DPanicLevel, expected: severityNumberDPanic},
		{name: "panic", level: zapcore.PanicLevel, expected: severityNumberPanic},
		{name: "fatal", level: zapcore.FatalLevel, expected: severityNumberFatal},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := SeverityNumber(tt.level); got != tt.expected {
				t.Errorf("SeverityNumber(%v) = %d, want %d", tt.level, got, tt.expected)
			}
		})
	}
}
