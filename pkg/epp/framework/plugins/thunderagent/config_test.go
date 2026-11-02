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
	"testing"

	"github.com/stretchr/testify/require"
)

// The defaults are valid; each gate parameter rejects values outside its
// range, and the idle TTL must outlast the starvation deadline.
func TestConfigValidate(t *testing.T) {
	require.NoError(t, defaultConfig().validate())

	for name, mutate := range map[string]func(*Config){
		"utilThreshold zero":            func(c *Config) { c.UtilThreshold = 0 },
		"utilThreshold above one":       func(c *Config) { c.UtilThreshold = 1.1 },
		"negative idleLeaseSeconds":     func(c *Config) { c.IdleLeaseSeconds = -1 },
		"negative headWaitStarvationMs": func(c *Config) { c.HeadWaitStarvationMs = -1 },
		"TTL not above starvation":      func(c *Config) { c.EvictionTTLSeconds, c.HeadWaitStarvationMs = 60, 60000 },
		"capacityTokens zero":           func(c *Config) { c.CapacityTokens = 0 },
		"evictionTtlSeconds zero":       func(c *Config) { c.EvictionTTLSeconds = 0 },
	} {
		cfg := defaultConfig()
		mutate(&cfg)
		require.Error(t, cfg.validate(), name)
	}

	cfg := defaultConfig()
	cfg.IdleLeaseSeconds = 0
	require.NoError(t, cfg.validate(), "a zero lease reclaims idle sessions at once")
}

// The default starvation deadline fires before the flow control default
// request TTL (60s) sheds the request.
func TestDefaultStarvationBelowRequestTTL(t *testing.T) {
	require.Less(t, defaultConfig().HeadWaitStarvationMs, float64(60000))
}
