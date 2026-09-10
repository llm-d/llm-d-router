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

package payload

import (
	"os"
	"strconv"

	"github.com/go-logr/logr"
)

// Environment variables forming the Phase 1 operator surface. They are the
// env-var aliases of the canonical payloadCapture.* fields defined in the
// proposal; the Kustomize overlay surface adopts the same names.
const (
	// EnvEnabled toggles payload capture (payloadCapture.enabled). Default false.
	EnvEnabled = "LLMD_PAYLOAD_CAPTURE_ENABLED"
	// EnvBackend selects the payload backend (payloadCapture.backend).
	// Phase 1 supports "noop" and "inline". Default "noop".
	EnvBackend = "LLMD_PAYLOAD_BACKEND"
	// EnvInlineThreshold caps the size of payloads carried inline as span-event
	// attributes (payloadCapture.inlineSizeThresholdBytes). Default 4096.
	EnvInlineThreshold = "LLMD_PAYLOAD_INLINE_THRESHOLD"
)

// Backend names accepted by EnvBackend in Phase 1. gcs, s3 and filesystem are
// reserved by the proposal for Phase 2 and fall back to noop with a warning.
const (
	BackendNoop   = "noop"
	BackendInline = "inline"
)

// DefaultInlineSizeThresholdBytes is the default value of
// payloadCapture.inlineSizeThresholdBytes (4 KiB, per the proposal).
const DefaultInlineSizeThresholdBytes = 4096

// Config is the Phase 1 subset of the payloadCapture configuration block.
type Config struct {
	// Enabled is the master switch; capture is opt-in and defaults to off.
	Enabled bool
	// Backend selects the PayloadStore implementation (noop | inline).
	Backend string
	// InlineSizeThresholdBytes is the largest serialised payload attached
	// inline to a span event.
	InlineSizeThresholdBytes int
}

// LoadConfigFromEnv reads the payload-capture configuration from the
// LLMD_PAYLOAD_* environment variables, applying proposal defaults and
// falling back safely (capture disabled / noop backend) on invalid input.
func LoadConfigFromEnv(logger logr.Logger) Config {
	cfg := Config{
		Enabled:                  false,
		Backend:                  BackendNoop,
		InlineSizeThresholdBytes: DefaultInlineSizeThresholdBytes,
	}

	if v, ok := os.LookupEnv(EnvEnabled); ok {
		enabled, err := strconv.ParseBool(v)
		if err != nil {
			logger.Info("invalid value for payload capture enabled flag, capture stays disabled",
				"env", EnvEnabled, "value", v)
		} else {
			cfg.Enabled = enabled
		}
	}

	if v, ok := os.LookupEnv(EnvBackend); ok && v != "" {
		switch v {
		case BackendNoop, BackendInline:
			cfg.Backend = v
		case "gcs", "s3", "filesystem":
			logger.Info("payload backend is not implemented yet (Phase 2), falling back to noop",
				"env", EnvBackend, "value", v)
		default:
			logger.Info("unknown payload backend, falling back to noop",
				"env", EnvBackend, "value", v)
		}
	}

	if v, ok := os.LookupEnv(EnvInlineThreshold); ok {
		threshold, err := strconv.Atoi(v)
		if err != nil || threshold <= 0 {
			logger.Info("invalid payload inline threshold, using default",
				"env", EnvInlineThreshold, "value", v, "default", DefaultInlineSizeThresholdBytes)
		} else {
			cfg.InlineSizeThresholdBytes = threshold
		}
	}

	if cfg.Enabled && cfg.Backend == BackendNoop {
		logger.Info("payload capture is enabled with the noop backend; no payload events will be emitted")
	}

	return cfg
}
