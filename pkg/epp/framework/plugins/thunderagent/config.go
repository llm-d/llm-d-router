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
	"errors"
	"fmt"
	"strings"
)

// Config holds the configuration for the ThunderAgent plugin.
type Config struct {
	// CapacityTokens is the per-endpoint KV cache capacity in tokens used when
	// the endpoint does not report cache_config_info (block_size *
	// num_gpu_blocks) through the datalayer.
	CapacityTokens int64 `json:"capacityTokens"`
	// RequireRealCapacity, when true, treats an endpoint without scraped
	// capacity as having no room for new programs instead of falling back to
	// CapacityTokens.
	RequireRealCapacity bool `json:"requireRealCapacity"`
	// UtilThreshold is the admission fit ceiling: a new program is admitted
	// only onto a pod whose working set plus the program's projected footprint
	// stays within this fraction of the pod's capacity. Turns of programs
	// already admitted bypass this ceiling; their footprint is already
	// counted.
	UtilThreshold float64 `json:"utilThreshold"`
	// ActingHalfLifeSeconds is the half-life applied to the committed tokens
	// of a program with no request in flight. 0 disables decay.
	ActingHalfLifeSeconds float64 `json:"actingHalfLifeSeconds"`
	// BufferTokensPerProgram is added to a program's footprint in the working
	// set and in the admission fit check, reserving growth room for the
	// context a session accumulates across its turns.
	BufferTokensPerProgram int64 `json:"bufferTokensPerProgram"`
	// ShedIdleSeconds enables proactive shedding: when a pod's working set
	// exceeds the fit ceiling, its idle programs (no request in flight, idle
	// at least this long) are unbound smallest first until the pod is back
	// under the ceiling. A shed program's next turn re-enters admission as a
	// new program: fit-checked, re-placed, held if nothing fits. 0 disables
	// shedding; over-admitted pods then rely on the engine's own eviction.
	ShedIdleSeconds float64 `json:"shedIdleSeconds"`
	// HeadWaitStarvationMs promotes any queue whose head has waited at least
	// this long ahead of class and size order. 0 disables the guard, which
	// lets a large program starve behind smaller ones.
	HeadWaitStarvationMs float64 `json:"headWaitStarvationMs"`
	// EvictionTTLSeconds is how long a program with no in-flight request and
	// no activity is kept before its state is dropped.
	EvictionTTLSeconds float64 `json:"evictionTtlSeconds"`
	// EvictionSweepSeconds is how often expired program state is swept.
	EvictionSweepSeconds float64 `json:"evictionSweepSeconds"`
	// SessionFinalHeader is the request header that marks the session's last
	// turn. The program's state is released when that turn completes instead
	// of waiting for the eviction TTL.
	SessionFinalHeader string `json:"sessionFinalHeader"`
	// ParentSessionHeader carries the parent session ID of a subagent. It is
	// recorded for observability only; accounting is not shared.
	ParentSessionHeader string `json:"parentSessionHeader"`
	// ProfileName selects which profile's picked endpoint a program is
	// attributed to. When empty, the primary (decode) profile is used.
	ProfileName string `json:"profileName"`
}

func defaultConfig() Config {
	return Config{
		CapacityTokens:         4194304,
		UtilThreshold:          0.9,
		BufferTokensPerProgram: 100,
		HeadWaitStarvationMs:   30000,
		EvictionTTLSeconds:     3600,
		EvictionSweepSeconds:   300,
		SessionFinalHeader:     "x-session-final",
		ParentSessionHeader:    "x-parent-session-id",
	}
}

func (c Config) validate() error {
	if c.CapacityTokens <= 0 {
		return fmt.Errorf("capacityTokens must be > 0, got %d", c.CapacityTokens)
	}
	if c.UtilThreshold <= 0 || c.UtilThreshold > 1 {
		return fmt.Errorf("utilThreshold must be in (0, 1], got %v", c.UtilThreshold)
	}
	if c.ActingHalfLifeSeconds < 0 {
		return fmt.Errorf("actingHalfLifeSeconds must be >= 0, got %v", c.ActingHalfLifeSeconds)
	}
	if c.BufferTokensPerProgram < 0 {
		return fmt.Errorf("bufferTokensPerProgram must be >= 0, got %d", c.BufferTokensPerProgram)
	}
	if c.ShedIdleSeconds < 0 {
		return fmt.Errorf("shedIdleSeconds must be >= 0, got %v", c.ShedIdleSeconds)
	}
	if c.HeadWaitStarvationMs < 0 {
		return fmt.Errorf("headWaitStarvationMs must be >= 0, got %v", c.HeadWaitStarvationMs)
	}
	if c.EvictionTTLSeconds <= 0 {
		return fmt.Errorf("evictionTtlSeconds must be > 0, got %v", c.EvictionTTLSeconds)
	}
	if c.EvictionSweepSeconds <= 0 {
		return fmt.Errorf("evictionSweepSeconds must be > 0, got %v", c.EvictionSweepSeconds)
	}
	if strings.TrimSpace(c.SessionFinalHeader) == "" {
		return errors.New("sessionFinalHeader must not be empty")
	}
	return nil
}
