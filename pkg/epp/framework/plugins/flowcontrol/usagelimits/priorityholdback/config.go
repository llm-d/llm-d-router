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

package priorityholdback

import (
	"errors"
	"fmt"

	"k8s.io/utils/ptr"
)

// strategy constants define the supported gating strategies.
const (
	// strategyStepwiseSpread divides the [minCeiling, maxCeiling] range into equal steps based on
	// the count of active priorities, ignoring their absolute numerical difference.
	strategyStepwiseSpread = "stepwise-spread"

	// strategyLinearProportional scales ceilings linearly based on numerical priority values
	// relative to the observed active range.
	strategyLinearProportional = "linear-proportional"
)

const (
	// defaultMaxCeiling allows the highest priority to use full capacity.
	defaultMaxCeiling float64 = 1.0
)

// apiConfig represents the external configuration schema for the priority holdback policy.
// It is designed to be deserialized from JSON via the plugin's raw parameters.
type apiConfig struct {
	// Strategy selects the gating algorithm used to compute per-priority admission ceilings.
	//
	// Required. Valid values: "stepwise-spread", "linear-proportional".
	Strategy *string `json:"strategy"`

	// MinCeiling is the admission ceiling assigned to the lowest-priority traffic.
	// Determines how aggressively the lowest priority is gated as saturation rises.
	//
	// Required. Must be in [0.0, 1.0) and strictly less than MaxCeiling.
	MinCeiling *float64 `json:"minCeiling"`

	// MaxCeiling is the admission ceiling assigned to the highest-priority traffic.
	// A value of 1.0 means the highest priority is only gated at full saturation.
	//
	// Defaults to 1.0 if unset. Must be in (0.0, 1.0] and strictly greater than MinCeiling.
	MaxCeiling *float64 `json:"maxCeiling,omitempty"`
}

// config is the internal, fully-validated configuration used by the policy.
type config struct {
	strategy   string
	minCeiling float64
	maxCeiling float64
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
		return nil, fmt.Errorf("invalid priority holdback policy configuration: %w", err)
	}

	applyDefaults(&safeCfg)

	if err := validateConfig(&safeCfg); err != nil {
		return nil, fmt.Errorf("invalid priority holdback policy configuration: %w", err)
	}

	return &config{
		strategy:   *safeCfg.Strategy,
		minCeiling: *safeCfg.MinCeiling,
		maxCeiling: *safeCfg.MaxCeiling,
	}, nil
}

// checkRequired verifies that mandatory fields are present before defaulting.
func checkRequired(cfg *apiConfig) error {
	var errs []error

	if cfg.Strategy == nil {
		errs = append(errs, fmt.Errorf("strategy is required"))
	}
	if cfg.MinCeiling == nil {
		errs = append(errs, fmt.Errorf("minCeiling is required"))
	}

	return errors.Join(errs...)
}

// applyDefaults populates unset optional fields with their standard defaults.
func applyDefaults(cfg *apiConfig) {
	if cfg.MaxCeiling == nil {
		cfg.MaxCeiling = ptr.To(defaultMaxCeiling)
	}
}

// validateConfig checks the constraints of the fully defaulted configuration.
// It aggregates all validation failures rather than failing on the first error.
func validateConfig(cfg *apiConfig) error {
	var errs []error

	if cfg.Strategy != nil {
		switch *cfg.Strategy {
		case strategyStepwiseSpread, strategyLinearProportional:
		default:
			errs = append(errs, fmt.Errorf("unsupported strategy %q, must be one of: %q, %q",
				*cfg.Strategy, strategyStepwiseSpread, strategyLinearProportional))
		}
	}

	if cfg.MinCeiling != nil && (*cfg.MinCeiling < 0.0 || *cfg.MinCeiling >= 1.0) {
		errs = append(errs, fmt.Errorf("minCeiling must be in [0.0, 1.0), got %f", *cfg.MinCeiling))
	}

	if cfg.MaxCeiling != nil && (*cfg.MaxCeiling <= 0.0 || *cfg.MaxCeiling > 1.0) {
		errs = append(errs, fmt.Errorf("maxCeiling must be in (0.0, 1.0], got %f", *cfg.MaxCeiling))
	}

	if cfg.MinCeiling != nil && cfg.MaxCeiling != nil && *cfg.MinCeiling >= *cfg.MaxCeiling {
		errs = append(errs, fmt.Errorf("minCeiling (%f) must be strictly less than maxCeiling (%f)",
			*cfg.MinCeiling, *cfg.MaxCeiling))
	}

	return errors.Join(errs...)
}
