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
	// UtilThreshold is the fraction of a pod's KV capacity at which the pod
	// is treated as full (1.0 = 100%).
	UtilThreshold float64 `json:"utilThreshold"`
	// IdleLeaseSeconds is how long an idle session keeps its room after its
	// last response. Past the lease, a paused or new session that does not
	// otherwise fit may take the room, pausing the idle session. Set it to
	// about a typical tool-call duration: shorter pauses sessions that are
	// about to return, longer holds admissions behind long tool calls.
	IdleLeaseSeconds float64 `json:"idleLeaseSeconds"`
	// HeadWaitStarvationMs is the maximum queue wait: a request waiting
	// this long is admitted even when no pod has room. 0 disables it.
	HeadWaitStarvationMs float64 `json:"headWaitStarvationMs"`
}

func defaultConfig() Config {
	return Config{
		CapacityTokens:       4194304,
		EvictionTTLSeconds:   3600,
		UtilThreshold:        1.0,
		IdleLeaseSeconds:     30,
		HeadWaitStarvationMs: 1800000,
	}
}

func (c Config) validate() error {
	if c.CapacityTokens <= 0 {
		return fmt.Errorf("capacityTokens must be > 0, got %d", c.CapacityTokens)
	}
	if c.EvictionTTLSeconds <= 0 {
		return fmt.Errorf("evictionTtlSeconds must be > 0, got %v", c.EvictionTTLSeconds)
	}
	if c.UtilThreshold <= 0 || c.UtilThreshold > 1 {
		return fmt.Errorf("utilThreshold must be in (0, 1], got %v", c.UtilThreshold)
	}
	if c.IdleLeaseSeconds < 0 {
		return fmt.Errorf("idleLeaseSeconds must be >= 0, got %v", c.IdleLeaseSeconds)
	}
	if c.HeadWaitStarvationMs < 0 {
		return fmt.Errorf("headWaitStarvationMs must be >= 0, got %v", c.HeadWaitStarvationMs)
	}
	if c.EvictionTTLSeconds*1000 <= c.HeadWaitStarvationMs {
		return fmt.Errorf("evictionTtlSeconds (%v s) must exceed headWaitStarvationMs (%v ms), or a held session is evicted mid-wait and re-enters as new", c.EvictionTTLSeconds, c.HeadWaitStarvationMs)
	}
	return nil
}
