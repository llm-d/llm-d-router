/*
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
	logger.Info("request body", HTTPBodyKey, `{"prompt":"hello"}`)

	var rec map[string]any
	if err := json.Unmarshal(buf.Bytes(), &rec); err != nil {
		t.Fatal(err)
	}
	if rec["body"] != "request body" {
		t.Errorf("body = %v, want request body", rec["body"])
	}
	if rec[HTTPBodyKey] != `{"prompt":"hello"}` {
		t.Errorf("%s = %v, want request payload", HTTPBodyKey, rec[HTTPBodyKey])
	}
	if rec["service.name"] != "test-service" {
		t.Errorf("service.name = %v, want test-service", rec["service.name"])
	}
}

func TestNewLoggerWithOptionsPreservesOTelConfiguration(t *testing.T) {
	var buf bytes.Buffer
	options := &crzap.Options{
		DestWriter: &buf,
		Level:      zapcore.InfoLevel,
	}
	logger := NewLoggerWithOptions("test-service", options)
	logger.Info("request assembled")

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
	if rec["service.name"] != "test-service" {
		t.Errorf("service.name = %v, want test-service", rec["service.name"])
	}
}

func TestServiceNameReadsOTelEnvironment(t *testing.T) {
	tests := []struct {
		name     string
		service  string
		attrs    string
		fallback string
		want     string
	}{
		{name: "fallback", fallback: "default", want: "default"},
		{name: "resource attributes", attrs: "service.name=from-attrs", fallback: "default", want: "from-attrs"},
		{name: "service name takes precedence", service: "from-service-name", attrs: "service.name=from-attrs", fallback: "default", want: "from-service-name"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("OTEL_SERVICE_NAME", tt.service)
			t.Setenv("OTEL_RESOURCE_ATTRIBUTES", tt.attrs)
			if got := ServiceName(tt.fallback); got != tt.want {
				t.Errorf("ServiceName(%q) = %q, want %q", tt.fallback, got, tt.want)
			}
		})
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
