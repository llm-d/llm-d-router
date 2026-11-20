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

package crossbandfairness

import (
	"errors"
	"fmt"

	"k8s.io/utils/ptr"
)

// schedule constants select how dispatch slots are spread over time.
const (
	// scheduleInterleaved spreads slots evenly using virtual-time scheduling.
	scheduleInterleaved = "interleaved"
)

// distribution constants select how a weight is derived for each priority band.
const (
	// distributionLinear decreases weights linearly by ordinal position, ignoring priority values.
	distributionLinear = "linear"
	// distributionExplicit uses operator-supplied weight anchors, interpolating missing priorities by
	// priority value and clamping outside the anchor range.
	distributionExplicit = "explicit"
)

const (
	defaultSchedule     = scheduleInterleaved
	defaultDistribution = distributionLinear
)

// apiConfig represents the external configuration schema for the cross-band fairness policy.
// It is designed to be deserialized from JSON via the plugin's raw parameters.
type apiConfig struct {
	// ActivationThreshold is the saturation level at or above which weighted fair queuing engages.
	// Below it, bands are offered a dispatch opportunity from highest to lowest priority.
	//
	// Required. Must be in [0.0, 1.0).
	ActivationThreshold *float64 `json:"activationThreshold"`

	// Schedule selects the mechanism that spreads dispatch slots over time.
	//   - "interleaved": virtual-time scheduling, spreading slots evenly across bands.
	//
	// Optional, defaults to "interleaved".
	Schedule *string `json:"schedule,omitempty"`

	// Distribution selects how a weight is derived for each priority band.
	//   - "linear": weights decrease linearly by ordinal position, ignoring priority values.
	//   - "explicit": priority levels map to weights via operator-supplied anchors with value-based
	//     interpolation.
	//
	// Optional, defaults to "linear".
	Distribution *string `json:"distribution,omitempty"`

	// Ratios maps configured priority levels (anchors) to their dispatch weights. Only the proportions
	// between weights matter, so they need not sum to any particular total. Missing priorities
	// interpolate linearly between anchors over priority value, or clamp to boundary anchors.
	//
	// Required when distribution is "explicit". Every weight must be strictly positive.
	// Unused and rejected when distribution is not "explicit".
	Ratios map[int]float64 `json:"ratios,omitempty"`
}

// config is the internal, fully-validated configuration used by the policy.
type config struct {
	activationThreshold float64
	distribution        string
	ratios              map[int]float64
}

// buildConfig applies the configuration lifecycle (defaulting and validation) and translates the
// external schema into the internal domain model.
// The provided apiConfig is copied to prevent mutation side-effects.
func buildConfig(apiCfg *apiConfig) (*config, error) {
	var safeCfg apiConfig
	if apiCfg != nil {
		safeCfg = *apiCfg
	}

	if err := checkRequired(&safeCfg); err != nil {
		return nil, fmt.Errorf("invalid cross-band fairness policy configuration: %w", err)
	}

	applyDefaults(&safeCfg)

	if err := validateConfig(&safeCfg); err != nil {
		return nil, fmt.Errorf("invalid cross-band fairness policy configuration: %w", err)
	}

	cfg := &config{
		activationThreshold: *safeCfg.ActivationThreshold,
		distribution:        *safeCfg.Distribution,
	}
	if cfg.distribution == distributionExplicit {
		cfg.ratios = safeCfg.Ratios
	}

	return cfg, nil
}

// checkRequired verifies that mandatory fields are present before defaulting.
func checkRequired(cfg *apiConfig) error {
	var errs []error

	if cfg.ActivationThreshold == nil {
		errs = append(errs, errors.New("activationThreshold is required"))
	}
	if cfg.Distribution != nil && *cfg.Distribution == distributionExplicit && len(cfg.Ratios) == 0 {
		errs = append(errs, errors.New("ratios is required when distribution is \"explicit\""))
	}

	return errors.Join(errs...)
}

// applyDefaults populates unset optional fields with their standard defaults.
func applyDefaults(cfg *apiConfig) {
	if cfg.Schedule == nil {
		cfg.Schedule = ptr.To(defaultSchedule)
	}
	if cfg.Distribution == nil {
		cfg.Distribution = ptr.To(defaultDistribution)
	}
}

// validateConfig checks the constraints of the fully defaulted configuration.
// It aggregates all validation failures rather than failing on the first error.
func validateConfig(cfg *apiConfig) error {
	var errs []error

	if cfg.ActivationThreshold != nil && (*cfg.ActivationThreshold < 0.0 || *cfg.ActivationThreshold >= 1.0) {
		errs = append(errs, fmt.Errorf("activationThreshold must be in [0.0, 1.0), got %f", *cfg.ActivationThreshold))
	}

	switch *cfg.Schedule {
	case scheduleInterleaved:
	default:
		errs = append(errs, fmt.Errorf("unsupported schedule %q, must be %q", *cfg.Schedule, scheduleInterleaved))
	}

	switch *cfg.Distribution {
	case distributionLinear:
		if len(cfg.Ratios) > 0 {
			errs = append(errs, errors.New("ratios must not be set when distribution is not \"explicit\""))
		}
	case distributionExplicit:
		for p, w := range cfg.Ratios {
			if w <= 0.0 {
				errs = append(errs, fmt.Errorf("ratio for priority %d must be strictly positive, got %f", p, w))
			}
		}
	default:
		errs = append(errs, fmt.Errorf("unsupported distribution %q, must be one of: %q, %q",
			*cfg.Distribution, distributionLinear, distributionExplicit))
	}

	return errors.Join(errs...)
}
