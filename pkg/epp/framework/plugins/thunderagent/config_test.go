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

package thunderagent

import (
	"math"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// The defaults are valid, zero boundaries on duration fields are allowed,
// and out-of-range values are rejected.
func TestConfigValidate(t *testing.T) {
	cfg := defaultConfig()
	require.NoError(t, cfg.validate())
	require.Equal(t, int64(4194304), cfg.FallbackCapacityTokens)
	require.Equal(t, 1.0, cfg.UsableCapacityRatio)
	require.Equal(t, 30*time.Second, cfg.IdleGracePeriod.Duration)
	require.Equal(t, 5*time.Minute, cfg.ForceAdmitTimeout.Duration)

	for name, mutate := range map[string]func(*Config){
		"fallbackCapacityTokens zero":     func(c *Config) { c.FallbackCapacityTokens = 0 },
		"fallbackCapacityTokens negative": func(c *Config) { c.FallbackCapacityTokens = -1 },
		"usableCapacityRatio zero":        func(c *Config) { c.UsableCapacityRatio = 0 },
		"usableCapacityRatio negative":    func(c *Config) { c.UsableCapacityRatio = -0.1 },
		"usableCapacityRatio above one":   func(c *Config) { c.UsableCapacityRatio = 1.1 },
		"usableCapacityRatio NaN":         func(c *Config) { c.UsableCapacityRatio = math.NaN() },
		"negative idleGracePeriod":        func(c *Config) { c.IdleGracePeriod = metav1.Duration{Duration: -time.Second} },
		"negative forceAdmitTimeout":      func(c *Config) { c.ForceAdmitTimeout = metav1.Duration{Duration: -time.Second} },
	} {
		c := defaultConfig()
		mutate(&c)
		require.Error(t, c.validate(), name)
	}

	cfg.IdleGracePeriod = metav1.Duration{Duration: 0}
	cfg.ForceAdmitTimeout = metav1.Duration{Duration: 0}
	require.NoError(t, cfg.validate(), "0s idleGracePeriod and 0s forceAdmitTimeout are valid")
}
