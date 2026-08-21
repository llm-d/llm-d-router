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

// Package compatibilityfilter provides a request-control Screener that groups
// endpoints by a metrics-derived compatibility value (e.g. a NIXL config hash)
// and ensures that only mutually compatible endpoints survive screening.
//
// This is useful during rolling upgrades of disaggregated prefill/decode
// deployments: pods running different model configs produce different hashes,
// and KV transfer between incompatible pods will fail. The screener picks one
// compatible group per request so the scheduler only sees endpoints that can
// exchange KV cache data.
package compatibilityfilter

import (
	"errors"
	"fmt"
	"strings"
)

// Config is the parameters block of a compatibility-screener plugin.
type Config struct {
	// AttributeKey is the endpoint attribute key that holds the compatibility
	// value (e.g. a hash extracted from a Prometheus metric label via the
	// customMetrics extractor with labelName).
	AttributeKey string `json:"attributeKey"`

	// HeaderName is the HTTP header used to propagate the chosen compatibility
	// value. On prefill responses it is stamped; on decode requests, if present,
	// it pins the screener to a specific value (strict matching).
	HeaderName string `json:"headerName"`

	// RoleLabelKey is the pod label key that identifies an endpoint's role
	// (e.g. "prefill", "decode"). Used to verify that a compatibility group
	// has coverage across all required roles before selecting it.
	// Optional: if empty, role coverage checking is skipped.
	RoleLabelKey string `json:"roleLabelKey,omitempty"`

	// RequireRoles lists the roles that must each have at least one endpoint
	// in a compatibility group for that group to be eligible. Ignored when
	// RoleLabelKey is empty.
	RequireRoles []string `json:"requireRoles,omitempty"`
}

// Validate performs static config checks.
func (c *Config) Validate() error {
	if c.AttributeKey == "" {
		return errors.New("attributeKey is required")
	}
	if c.HeaderName == "" {
		return errors.New("headerName is required")
	}
	c.HeaderName = strings.ToLower(c.HeaderName)

	if c.RoleLabelKey != "" && len(c.RequireRoles) == 0 {
		return errors.New("requireRoles must contain at least one role when roleLabelKey is set")
	}
	if c.RoleLabelKey == "" && len(c.RequireRoles) > 0 {
		return errors.New("roleLabelKey is required when requireRoles is set")
	}

	seen := make(map[string]struct{}, len(c.RequireRoles))
	for i, role := range c.RequireRoles {
		if role == "" {
			return fmt.Errorf("requireRoles[%d] must not be empty", i)
		}
		if _, exists := seen[role]; exists {
			return fmt.Errorf("requireRoles contains duplicate role %q", role)
		}
		seen[role] = struct{}{}
	}
	return nil
}
