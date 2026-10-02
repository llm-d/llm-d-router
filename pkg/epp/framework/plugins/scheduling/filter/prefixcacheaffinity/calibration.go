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
	"errors"
	"fmt"
	"math"
	"os"
)

// CalibrationSchemaVersion is the only calibration record schema this consumer
// understands; a record carrying another version is not applied.
const CalibrationSchemaVersion = 1

// CalibrationStatusOK is the status of a completed measurement. Any other
// status (NOT_RUN, FAILED, BLOCKED, ...) carries no value to apply.
const CalibrationStatusOK = "OK"

// MinCalibrationSamples is how many valid samples a record needs before its
// median is trusted.
const MinCalibrationSamples = 10

// CalibrationRecord is the structured output of the peak prefill calibration
// recipe, which writes raw samples, rejection reasons and this summary
// separately. The field names are the recipe's schema, not EPP config.
type CalibrationRecord struct {
	SchemaVersion              int      `json:"schema_version"`
	Status                     string   `json:"status"`
	Fingerprint                string   `json:"fingerprint"`
	ModelRevision              string   `json:"model_revision"`
	EngineImageDigest          string   `json:"engine_image_digest"`
	ChunkTokens                *int     `json:"chunk_tokens"`
	ValidSamples               int      `json:"valid_samples"`
	MedianTTFTSeconds          *float64 `json:"median_ttft_seconds"`
	PeakPrefillTokensPerSecond *float64 `json:"peak_prefill_tokens_per_second"`
	Applied                    bool     `json:"applied"`
}

// LoadCalibrationRecord reads a calibration record from path.
func LoadCalibrationRecord(path string) (*CalibrationRecord, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var record CalibrationRecord
	if err := json.Unmarshal(raw, &record); err != nil {
		return nil, fmt.Errorf("malformed calibration record %s: %w", path, err)
	}
	return &record, nil
}

// usable explains why the record cannot be applied, or returns nil when it can.
// Only a completed measurement of a positive, finite throughput derived from
// enough samples qualifies: a placeholder, a truncated write or a failed run
// must never reach the effective config.
func (r *CalibrationRecord) usable() error {
	if r.SchemaVersion != CalibrationSchemaVersion {
		return fmt.Errorf("schema_version is %d, want %d", r.SchemaVersion, CalibrationSchemaVersion)
	}
	if r.Status != CalibrationStatusOK {
		return fmt.Errorf("status is %q, want %q", r.Status, CalibrationStatusOK)
	}
	if r.PeakPrefillTokensPerSecond == nil {
		return errors.New("peak_prefill_tokens_per_second is missing")
	}
	if v := *r.PeakPrefillTokensPerSecond; math.IsNaN(v) || math.IsInf(v, 0) || v <= 0 {
		return fmt.Errorf("peak_prefill_tokens_per_second is not a positive finite number: %v", v)
	}
	if r.ChunkTokens == nil || *r.ChunkTokens <= 0 {
		return errors.New("chunk_tokens is missing or not positive")
	}
	if r.ValidSamples < MinCalibrationSamples {
		return fmt.Errorf("valid_samples is %d, want at least %d", r.ValidSamples, MinCalibrationSamples)
	}
	if r.MedianTTFTSeconds == nil {
		return errors.New("median_ttft_seconds is missing")
	}
	if v := *r.MedianTTFTSeconds; math.IsNaN(v) || math.IsInf(v, 0) || v <= 0 {
		return fmt.Errorf("median_ttft_seconds is not a positive finite number: %v", v)
	}
	return nil
}

// matchesFingerprint reports whether the record was measured on the deployment
// described by want. The producer composes the fingerprint from the identity of
// the measured deployment — model revision, engine image digest, dtype or
// quantization, accelerator, parallelism, batch and chunk limits, and the pool
// the measurement targeted — so an equal string means the measurement still
// describes this deployment. A model, TP, batch-limit or image change therefore
// invalidates the record instead of silently reusing it.
func (r *CalibrationRecord) matchesFingerprint(want string) error {
	if r.Fingerprint == "" {
		return errors.New("record carries no fingerprint")
	}
	if want == "" {
		return errors.New("no prefillCalibrationFingerprint is configured for this deployment")
	}
	if r.Fingerprint != want {
		return fmt.Errorf("record was measured on a different deployment: fingerprint %q, want %q", r.Fingerprint, want)
	}
	return nil
}

// PrefillThroughputSource records where the effective peakPrefillThroughput
// came from, so the resolved value is visible in logs and setup dry-runs.
type PrefillThroughputSource string

const (
	// SourceUserConfigured: peakPrefillThroughput was present in the plugin
	// parameters, so the operator's value is in effect.
	SourceUserConfigured PrefillThroughputSource = "userConfigured"
	// SourceCalibrated: a fingerprint-matched calibration record was applied.
	SourceCalibrated PrefillThroughputSource = "calibrated"
	// SourceDefault: the built-in default is in effect.
	SourceDefault PrefillThroughputSource = "default"
	// SourceNotUsed: ttftSource is latencyPredictor, so no prefill throughput
	// constant is in the path and no calibration is required or consulted.
	SourceNotUsed PrefillThroughputSource = "notUsed"
)

// applyCalibration resolves the effective PeakPrefillThroughput and its source
// following
//
//	explicit user value > fingerprint-matched calibration > built-in default
//
// explicit reports whether peakPrefillThroughput was present in the plugin
// parameters: a value the operator wrote wins even when it equals the built-in
// default, while an absent key leaves room for a calibrated value. The returned
// error is the reason a configured calibration was skipped, and is nil when one
// was applied or when none was configured; the caller decides whether that
// reason is fatal, which it is only under prefillCalibrationRequired.
func (c *Config) applyCalibration(explicit bool) error {
	if c.usesLatencyPredictor() {
		c.PeakPrefillThroughputSource = SourceNotUsed
		return nil
	}
	if explicit {
		c.PeakPrefillThroughputSource = SourceUserConfigured
		return nil
	}
	c.PeakPrefillThroughputSource = SourceDefault
	if c.PrefillCalibrationFile == "" {
		if c.PrefillCalibrationRequired {
			return errors.New("prefillCalibrationRequired is set but prefillCalibrationFile is empty")
		}
		return nil
	}
	record, err := LoadCalibrationRecord(c.PrefillCalibrationFile)
	if err == nil {
		err = record.usable()
	}
	if err == nil {
		err = record.matchesFingerprint(c.PrefillCalibrationFingerprint)
	}
	if err != nil {
		return fmt.Errorf("calibration not applied: %w", err)
	}
	c.PeakPrefillThroughput = *record.PeakPrefillTokensPerSecond
	c.PeakPrefillThroughputSource = SourceCalibrated
	return nil
}

// parameterPresent reports whether the raw plugin parameters carry key, which
// distinguishes a value the operator wrote from the built-in default the
// decoder leaves in place for an absent one.
func parameterPresent(raw json.RawMessage, key string) bool {
	var parameters map[string]json.RawMessage
	if err := json.Unmarshal(raw, &parameters); err != nil {
		return false
	}
	_, present := parameters[key]
	return present
}
