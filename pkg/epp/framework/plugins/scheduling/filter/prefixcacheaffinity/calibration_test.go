/*
Copyright 2026 The Kubernetes Authors.
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

package prefixcacheaffinity

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// measuredFingerprint is the deployment identity a record in these tests was
// measured on.
const measuredFingerprint = "sha256:measured-deployment"

// Parameter names and record fields that the cases below repeat.
const (
	paramPeakPrefillThroughput         = "peakPrefillThroughput"
	paramPrefillCalibrationFile        = "prefillCalibrationFile"
	paramPrefillCalibrationFingerprint = "prefillCalibrationFingerprint"
	paramPrefillCalibrationRequired    = "prefillCalibrationRequired"
	paramMaxTTFTPenaltyMs              = "maxTTFTPenaltyMs"
	keySchemaVersion                   = "schema_version"
	keyStatus                          = "status"
	keyFingerprint                     = "fingerprint"
	keyModelRevision                   = "model_revision"
	keyEngineImageDigest               = "engine_image_digest"
	keyChunkTokens                     = "chunk_tokens"
	keyValidSamples                    = "valid_samples"
	keyMedianTTFTSeconds               = "median_ttft_seconds"
	keyPeakPrefillTokensPerSecond      = "peak_prefill_tokens_per_second"
	keyApplied                         = "applied"
)

// recordFieldOrder keeps the rendered record byte-stable, so a failure names
// the one field the test changed.
var recordFieldOrder = []string{
	keySchemaVersion, keyStatus, keyFingerprint, keyModelRevision, keyEngineImageDigest,
	keyChunkTokens, keyValidSamples, keyMedianTTFTSeconds, keyPeakPrefillTokensPerSecond, keyApplied,
}

// calibrationRecordJSON renders a complete, applicable record. An empty
// override value drops the field, an override value is raw JSON.
func calibrationRecordJSON(overrides map[string]string) string {
	fields := map[string]string{
		keySchemaVersion:              "1",
		keyStatus:                     `"OK"`,
		keyFingerprint:                strconv.Quote(measuredFingerprint),
		keyModelRevision:              `"Qwen/Qwen3-32B@revision"`,
		keyEngineImageDigest:          `"sha256:engine-image"`,
		keyChunkTokens:                "2048",
		keyValidSamples:               "20",
		keyMedianTTFTSeconds:          "0.128",
		keyPeakPrefillTokensPerSecond: "16000",
		keyApplied:                    "false",
	}
	for key, value := range overrides {
		if value == "" {
			delete(fields, key)
			continue
		}
		fields[key] = value
	}
	var body strings.Builder
	body.WriteString("{")
	first := true
	for _, key := range recordFieldOrder {
		value, ok := fields[key]
		if !ok {
			continue
		}
		if !first {
			body.WriteString(",")
		}
		body.WriteString(strconv.Quote(key))
		body.WriteByte(':')
		body.WriteString(value)
		first = false
	}
	body.WriteString("}")
	return body.String()
}

func writeRecord(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "prefill-calibration.json")
	require.NoError(t, os.WriteFile(path, []byte(body), 0o600))
	return path
}

func calibrationParams(recordPath string) map[string]any {
	return map[string]any{
		paramPrefillCalibrationFile:        recordPath,
		paramPrefillCalibrationFingerprint: measuredFingerprint,
	}
}

// newFactoryPlugin builds the plugin the way the config loader does, through
// the registered factory, so parameter decoding and presence detection are the
// ones under test.
func newFactoryPlugin(t *testing.T, params map[string]any) (*Plugin, error) {
	t.Helper()
	raw, err := json.Marshal(params)
	require.NoError(t, err)
	plugin, err := Factory("test", fwkplugin.StrictDecoder(raw), nil)
	if plugin == nil {
		return nil, err
	}
	return plugin.(*Plugin), err
}

func float64Ptr(v float64) *float64 { return &v }
func intPtr(v int) *int             { return &v }

// A1-U01: an absent key leaves room for a calibrated value, an explicit value is
// the operator's choice even when it repeats the built-in default, and an
// invalid explicit value is reported rather than papered over by a fallback.
func TestA1U01_PeakPrefillThroughputPrecedence(t *testing.T) {
	record := writeRecord(t, calibrationRecordJSON(nil))

	tests := []struct {
		name       string
		params     map[string]any
		want       float64
		wantSource PrefillThroughputSource
	}{
		{
			name:       "absent key keeps the built-in default",
			params:     map[string]any{},
			want:       DefaultConfig.PeakPrefillThroughput,
			wantSource: SourceDefault,
		},
		{
			name:       "absent key applies a fingerprint-matched measurement",
			params:     calibrationParams(record),
			want:       16000,
			wantSource: SourceCalibrated,
		},
		{
			name: "explicit value equal to the default is still the operator's choice",
			params: map[string]any{
				paramPeakPrefillThroughput:         DefaultConfig.PeakPrefillThroughput,
				paramPrefillCalibrationFile:        record,
				paramPrefillCalibrationFingerprint: measuredFingerprint,
			},
			want:       DefaultConfig.PeakPrefillThroughput,
			wantSource: SourceUserConfigured,
		},
		{
			name: "explicit value wins over the measurement",
			params: map[string]any{
				paramPeakPrefillThroughput:         12345,
				paramPrefillCalibrationFile:        record,
				paramPrefillCalibrationFingerprint: measuredFingerprint,
			},
			want:       12345,
			wantSource: SourceUserConfigured,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			plugin, err := newFactoryPlugin(t, tt.params)
			require.NoError(t, err)
			assert.Equal(t, tt.want, plugin.config.PeakPrefillThroughput)
			assert.Equal(t, tt.wantSource, plugin.config.PeakPrefillThroughputSource)
		})
	}
}

func TestA1U01_InvalidExplicitValueIsNotMaskedByCalibration(t *testing.T) {
	record := writeRecord(t, calibrationRecordJSON(nil))

	_, err := newFactoryPlugin(t, map[string]any{
		paramPeakPrefillThroughput:         -1,
		paramPrefillCalibrationFile:        record,
		paramPrefillCalibrationFingerprint: measuredFingerprint,
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "peakPrefillThroughput must be >= 0")

	_, err = newFactoryPlugin(t, map[string]any{
		paramPeakPrefillThroughput:         0,
		paramPrefillCalibrationFile:        record,
		paramPrefillCalibrationFingerprint: measuredFingerprint,
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "peakPrefillThroughput must be > 0")
}

// A1-U02: zero, negative, non-finite, sample-poor and malformed records never
// reach the effective config, and the reason stays visible.
func TestA1U02_UnusableRecordsNeverReachTheConfig(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{"zero throughput", calibrationRecordJSON(map[string]string{keyPeakPrefillTokensPerSecond: "0"})},
		{"negative throughput", calibrationRecordJSON(map[string]string{keyPeakPrefillTokensPerSecond: "-16000"})},
		{"missing throughput", calibrationRecordJSON(map[string]string{keyPeakPrefillTokensPerSecond: ""})},
		{"zero median ttft", calibrationRecordJSON(map[string]string{keyMedianTTFTSeconds: "0"})},
		{"missing median ttft", calibrationRecordJSON(map[string]string{keyMedianTTFTSeconds: ""})},
		{"missing chunk size", calibrationRecordJSON(map[string]string{keyChunkTokens: ""})},
		{"zero chunk size", calibrationRecordJSON(map[string]string{keyChunkTokens: "0"})},
		{"too few samples", calibrationRecordJSON(map[string]string{keyValidSamples: "9"})},
		{"placeholder not run", calibrationRecordJSON(map[string]string{
			keyStatus: "NOT_RUN", keyChunkTokens: "", keyValidSamples: "0",
			keyMedianTTFTSeconds: "", keyPeakPrefillTokensPerSecond: "",
		})},
		{"failed run", calibrationRecordJSON(map[string]string{keyStatus: `"FAILED"`})},
		{"unknown schema version", calibrationRecordJSON(map[string]string{keySchemaVersion: "2"})},
		{"truncated write", `{"schema_version":1,"status":"OK","peak_prefill`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := writeRecord(t, tt.body)

			plugin, err := newFactoryPlugin(t, calibrationParams(path))
			require.NoError(t, err, "an optional calibration failure must not stop the plugin")
			assert.Equal(t, DefaultConfig.PeakPrefillThroughput, plugin.config.PeakPrefillThroughput,
				"the built-in default is the only value in effect")
			assert.Equal(t, SourceDefault, plugin.config.PeakPrefillThroughputSource)
			require.NoError(t, plugin.config.validate())

			_, err = newFactoryPlugin(t, map[string]any{
				paramPrefillCalibrationFile:        path,
				paramPrefillCalibrationFingerprint: measuredFingerprint,
				paramPrefillCalibrationRequired:    true,
			})
			assert.Error(t, err, "required calibration must refuse to start on the same record")
		})
	}
}

func TestA1U02_MissingRecordFile(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "absent.json")

	plugin, err := newFactoryPlugin(t, calibrationParams(missing))
	require.NoError(t, err)
	assert.Equal(t, DefaultConfig.PeakPrefillThroughput, plugin.config.PeakPrefillThroughput)
	assert.Equal(t, SourceDefault, plugin.config.PeakPrefillThroughputSource)

	_, err = newFactoryPlugin(t, map[string]any{
		paramPrefillCalibrationFile:        missing,
		paramPrefillCalibrationFingerprint: measuredFingerprint,
		paramPrefillCalibrationRequired:    true,
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "calibration not applied")
}

// Non-finite throughput has no JSON spelling, so it is rejected where the value
// is validated: no code path may put it into the config.
func TestA1U02_NonFiniteValuesAreRejected(t *testing.T) {
	for _, value := range []float64{math.NaN(), math.Inf(1), math.Inf(-1), 0, -1} {
		record := CalibrationRecord{
			SchemaVersion:              CalibrationSchemaVersion,
			Status:                     CalibrationStatusOK,
			Fingerprint:                measuredFingerprint,
			ChunkTokens:                intPtr(2048),
			ValidSamples:               20,
			MedianTTFTSeconds:          float64Ptr(0.128),
			PeakPrefillTokensPerSecond: float64Ptr(value),
		}
		assert.Error(t, record.usable(), "throughput %v must be rejected", value)
	}

	record := CalibrationRecord{
		SchemaVersion:              CalibrationSchemaVersion,
		Status:                     CalibrationStatusOK,
		Fingerprint:                measuredFingerprint,
		ChunkTokens:                intPtr(2048),
		ValidSamples:               20,
		MedianTTFTSeconds:          float64Ptr(math.NaN()),
		PeakPrefillTokensPerSecond: float64Ptr(16000),
	}
	assert.Error(t, record.usable(), "a NaN median must be rejected")

	// A producer that writes the Python/Go non-standard NaN literal produces a
	// record that does not even parse, which is also a refusal to apply.
	path := writeRecord(t, `{"schema_version":1,"status":"OK","fingerprint":"sha256:measured-deployment","chunk_tokens":2048,"valid_samples":20,"median_ttft_seconds":0.128,"peak_prefill_tokens_per_second":NaN}`)
	plugin, err := newFactoryPlugin(t, calibrationParams(path))
	require.NoError(t, err)
	assert.Equal(t, SourceDefault, plugin.config.PeakPrefillThroughputSource)
}

// A1-U03: a model, TP, batch-limit or image change produces a different
// fingerprint, so the previous measurement is not silently reused.
func TestA1U03_FingerprintChangeInvalidatesTheRecord(t *testing.T) {
	tests := []struct {
		name       string
		recordBody string
		params     map[string]any
	}{
		{
			name:       "measured on another deployment",
			recordBody: calibrationRecordJSON(map[string]string{keyFingerprint: `"sha256:other-deployment"`}),
			params:     calibrationParams("PLACEHOLDER"),
		},
		{
			name:       "record carries no fingerprint",
			recordBody: calibrationRecordJSON(map[string]string{keyFingerprint: ""}),
			params:     calibrationParams("PLACEHOLDER"),
		},
		{
			name:       "deployment fingerprint not configured",
			recordBody: calibrationRecordJSON(nil),
			params:     map[string]any{paramPrefillCalibrationFile: "PLACEHOLDER"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := writeRecord(t, tt.recordBody)
			params := tt.params
			if _, ok := params[paramPrefillCalibrationFile]; ok {
				params[paramPrefillCalibrationFile] = path
			}
			plugin, err := newFactoryPlugin(t, params)
			require.NoError(t, err)
			assert.Equal(t, DefaultConfig.PeakPrefillThroughput, plugin.config.PeakPrefillThroughput)
			assert.Equal(t, SourceDefault, plugin.config.PeakPrefillThroughputSource)
		})
	}
}

// A1-U04: optional calibration failures stay local to the calibration, required
// ones stop the plugin, and neither leaves a half-applied value behind.
func TestA1U04_OptionalVersusRequiredCalibrationFailure(t *testing.T) {
	broken := writeRecord(t, `{"schema_version":1,"status":"OK","peak_prefill`)
	missing := filepath.Join(t.TempDir(), "absent.json")

	for _, path := range []string{broken, missing} {
		plugin, err := newFactoryPlugin(t, calibrationParams(path))
		require.NoError(t, err, "optional calibration failure must not stop the plugin")
		assert.Equal(t, DefaultConfig.PeakPrefillThroughput, plugin.config.PeakPrefillThroughput)
		assert.Equal(t, SourceDefault, plugin.config.PeakPrefillThroughputSource)

		_, err = newFactoryPlugin(t, map[string]any{
			paramPrefillCalibrationFile:        path,
			paramPrefillCalibrationFingerprint: measuredFingerprint,
			paramPrefillCalibrationRequired:    true,
		})
		require.Error(t, err)
	}

	// Required with no calibration configured at all is a configuration error,
	// not a silent default.
	_, err := newFactoryPlugin(t, map[string]any{paramPrefillCalibrationRequired: true})
	require.Error(t, err)
	assert.Contains(t, err.Error(), paramPrefillCalibrationRequired)

	// Optional failure alongside an explicit value leaves that value intact.
	record := writeRecord(t, `{"schema_version":1,"status":"OK","peak_prefill`)
	plugin, err := newFactoryPlugin(t, map[string]any{
		paramPeakPrefillThroughput:         12345,
		paramPrefillCalibrationFile:        record,
		paramPrefillCalibrationFingerprint: measuredFingerprint,
	})
	require.NoError(t, err)
	assert.Equal(t, 12345.0, plugin.config.PeakPrefillThroughput)
	assert.Equal(t, SourceUserConfigured, plugin.config.PeakPrefillThroughputSource)
}

// A1-U06: with the latency predictor as the TTFT source no prefill constant is
// in the path, so calibration is neither required nor consulted.
func TestA1U06_LatencyPredictorDoesNotRequireCalibration(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "absent.json")

	plugin, err := newFactoryPlugin(t, map[string]any{
		"ttftSource":                       string(TTFTSourceLatencyPredictor),
		paramPrefillCalibrationFile:        missing,
		paramPrefillCalibrationFingerprint: measuredFingerprint,
		paramPrefillCalibrationRequired:    true,
	})
	require.NoError(t, err, "a required calibration must not block the latency predictor path")
	assert.Equal(t, SourceNotUsed, plugin.config.PeakPrefillThroughputSource)

	// The gate is on and the value is unused, so even an unset throughput is a
	// valid configuration on this path.
	plugin, err = newFactoryPlugin(t, map[string]any{
		"ttftSource":                    string(TTFTSourceLatencyPredictor),
		paramPeakPrefillThroughput:      0,
		paramPrefillCalibrationRequired: true,
	})
	require.NoError(t, err)
	assert.Equal(t, SourceNotUsed, plugin.config.PeakPrefillThroughputSource)
}

// A1-U06: a record measured on one pool is not reused for another pool, so a
// single measurement can never stand in for a heterogeneous deployment.
func TestA1U06_HeterogeneousPoolRecordIsNotReused(t *testing.T) {
	record := writeRecord(t, calibrationRecordJSON(map[string]string{keyFingerprint: `"sha256:pool-a"`}))

	plugin, err := newFactoryPlugin(t, map[string]any{
		paramPrefillCalibrationFile:        record,
		paramPrefillCalibrationFingerprint: "sha256:pool-b",
	})
	require.NoError(t, err)
	assert.Equal(t, DefaultConfig.PeakPrefillThroughput, plugin.config.PeakPrefillThroughput)
	assert.Equal(t, SourceDefault, plugin.config.PeakPrefillThroughputSource)
}

// The setup injects prefillCalibrationFile and prefillCalibrationFingerprint by
// name, so a renamed or mistyped parameter must fail loudly at startup instead
// of being ignored and silently falling back to the default.
func TestA1U02_UnknownCalibrationParameterIsRejected(t *testing.T) {
	record := writeRecord(t, calibrationRecordJSON(nil))
	_, err := newFactoryPlugin(t, map[string]any{
		"prefillCalibrationFil":            record,
		paramPrefillCalibrationFingerprint: measuredFingerprint,
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unknown field")
}
