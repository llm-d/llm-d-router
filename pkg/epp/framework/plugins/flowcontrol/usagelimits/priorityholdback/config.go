/*
Copyright 2026 The Kubernetes Authors.

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
	"sort"

	"k8s.io/utils/ptr"
)

// shape constants define the interpolation curve applied across the ceiling range.
// Only "linear" is supported; additional shapes (sigmoid, exponential, step, etc.) may be added.
const (
	shapeLinear = "linear"
)

// domain constants define how priority levels are mapped to positions in the ceiling range.
const (
	// domainRank maps by ordinal rank, ignoring numerical priority values.
	domainRank = "rank"
	// domainValue maps proportionally to numerical priority values.
	domainValue = "value"
	// domainExplicit uses a caller-supplied map of priority level to ceiling value directly.
	domainExplicit = "explicit"
)

const (
	defaultShape              = shapeLinear
	defaultDomain             = domainRank
	defaultMaxCeiling float64 = 1.0
)

// apiConfig represents the external configuration schema for the priority holdback policy.
// It is designed to be deserialized from JSON via the plugin's raw parameters.
type apiConfig struct {
	// Shape selects the interpolation curve used to distribute ceilings across the range.
	//
	// Optional, defaults to "linear". Currently only "linear" is supported.
	// Unused and rejected when domain is "explicit".
	Shape *string `json:"shape,omitempty"`

	// Domain selects how priority levels are mapped to positions in the ceiling range.
	//   - "rank": equal spacing by ordinal rank, ignoring numerical values.
	//   - "value": spacing proportional to numerical priority differences.
	//   - "explicit": each priority level's ceiling is taken directly from the Ceilings map.
	//
	// Optional, defaults to "rank".
	Domain *string `json:"domain,omitempty"`

	// MinCeiling is the admission ceiling assigned to the lowest-priority traffic.
	// Determines how aggressively the lowest priority is gated as saturation rises.
	//
	// Required when domain is not "explicit". Must be in [0.0, 1.0) and strictly less than MaxCeiling.
	// Unused and rejected when domain is "explicit".
	MinCeiling *float64 `json:"minCeiling"`

	// MaxCeiling is the admission ceiling assigned to the highest-priority traffic.
	// A value of 1.0 means the highest priority is only gated at full saturation.
	//
	// Defaults to 1.0 if unset. Must be in (0.0, 1.0] and strictly greater than MinCeiling.
	// Unused and rejected when domain is "explicit".
	MaxCeiling *float64 `json:"maxCeiling,omitempty"`

	// Ceilings maps each priority level to its admission ceiling in [0.0, 1.0].
	//
	// Required when domain is "explicit". Ceilings must be monotonically non-increasing
	// when priorities are sorted highest-first. Unused when domain is not "explicit".
	Ceilings map[int]float64 `json:"ceilings,omitempty"`
}

// config is the internal, fully-validated configuration used by the policy.
type config struct {
	shape      string
	domain     string
	minCeiling float64
	maxCeiling float64
	ceilings   map[int]float64
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

	cfg := &config{
		domain: *safeCfg.Domain,
	}
	if *safeCfg.Domain == domainExplicit {
		cfg.shape = defaultShape
		cfg.ceilings = safeCfg.Ceilings
	} else {
		cfg.shape = *safeCfg.Shape
		cfg.minCeiling = *safeCfg.MinCeiling
		cfg.maxCeiling = *safeCfg.MaxCeiling
	}

	return cfg, nil
}

// checkRequired verifies that mandatory fields are present before defaulting.
func checkRequired(cfg *apiConfig) error {
	if cfg.Domain != nil && *cfg.Domain == domainExplicit {
		if len(cfg.Ceilings) == 0 {
			return errors.New("ceilings is required when domain is \"explicit\"")
		}
		return nil
	}
	if cfg.MinCeiling == nil {
		return errors.New("minCeiling is required")
	}
	return nil
}

// applyDefaults populates unset optional fields with their standard defaults.
func applyDefaults(cfg *apiConfig) {
	if cfg.Domain == nil {
		cfg.Domain = ptr.To(defaultDomain)
	}
	if *cfg.Domain == domainExplicit {
		// shape and maxCeiling defaults do not apply for the explicit domain.
		return
	}
	if cfg.Shape == nil {
		cfg.Shape = ptr.To(defaultShape)
	}
	if cfg.MaxCeiling == nil {
		cfg.MaxCeiling = ptr.To(defaultMaxCeiling)
	}
}

// validateConfig checks the constraints of the fully defaulted configuration.
// It aggregates all validation failures rather than failing on the first error.
func validateConfig(cfg *apiConfig) error {
	if cfg.Domain != nil && *cfg.Domain == domainExplicit {
		return validateExplicitConfig(cfg)
	}

	var errs []error

	if cfg.Shape != nil {
		switch *cfg.Shape {
		case shapeLinear:
		default:
			errs = append(errs, fmt.Errorf("unsupported shape %q, must be %q",
				*cfg.Shape, shapeLinear))
		}
	}

	if cfg.Domain != nil {
		switch *cfg.Domain {
		case domainRank, domainValue:
		default:
			errs = append(errs, fmt.Errorf("unsupported domain %q, must be one of: %q, %q, %q",
				*cfg.Domain, domainRank, domainValue, domainExplicit))
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

// validateExplicitConfig validates the constraints specific to domain "explicit".
func validateExplicitConfig(cfg *apiConfig) error {
	var errs []error

	if cfg.Shape != nil {
		errs = append(errs, errors.New("shape must not be set when domain is \"explicit\""))
	}
	if cfg.MinCeiling != nil {
		errs = append(errs, errors.New("minCeiling must not be set when domain is \"explicit\""))
	}
	if cfg.MaxCeiling != nil {
		errs = append(errs, errors.New("maxCeiling must not be set when domain is \"explicit\""))
	}

	// Validate individual ceiling values.
	for p, c := range cfg.Ceilings {
		if c < 0.0 || c > 1.0 {
			errs = append(errs, fmt.Errorf("ceiling for priority %d must be in [0.0, 1.0], got %f", p, c))
		}
	}

	// Validate monotonicity: ceilings must be non-increasing as priorities decrease.
	// Only check after individual value errors are clear to avoid misleading messages.
	if len(errs) == 0 {
		priorities := make([]int, 0, len(cfg.Ceilings))
		for p := range cfg.Ceilings {
			priorities = append(priorities, p)
		}
		sort.Sort(sort.Reverse(sort.IntSlice(priorities)))
		for i := 1; i < len(priorities); i++ {
			if cfg.Ceilings[priorities[i]] > cfg.Ceilings[priorities[i-1]] {
				errs = append(errs, fmt.Errorf(
					"ceilings must be monotonically non-increasing (highest priority first): "+
						"priority %d has ceiling %f which exceeds priority %d ceiling %f",
					priorities[i], cfg.Ceilings[priorities[i]],
					priorities[i-1], cfg.Ceilings[priorities[i-1]]))
				break
			}
		}
	}

	return errors.Join(errs...)
}
