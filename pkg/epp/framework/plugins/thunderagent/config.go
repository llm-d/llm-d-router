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
	"fmt"
)

// Config holds the configuration for the ThunderAgent plugin.
type Config struct {
	// CapacityTokens is the per-endpoint KV cache capacity in tokens used when
	// the endpoint does not report cache_config_info through the datalayer.
	CapacityTokens int64 `json:"capacityTokens"`
	// EvictionTTLSeconds is how long a session with no in-flight request and
	// no activity is kept before its state is dropped.
	EvictionTTLSeconds float64 `json:"evictionTtlSeconds"`
}

func defaultConfig() Config {
	return Config{
		CapacityTokens:     4194304,
		EvictionTTLSeconds: 3600,
	}
}

func (c Config) validate() error {
	if c.CapacityTokens <= 0 {
		return fmt.Errorf("capacityTokens must be > 0, got %d", c.CapacityTokens)
	}
	if c.EvictionTTLSeconds <= 0 {
		return fmt.Errorf("evictionTtlSeconds must be > 0, got %v", c.EvictionTTLSeconds)
	}
	return nil
}
