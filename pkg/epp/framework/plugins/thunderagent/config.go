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
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const defaultForceAdmitTimeout = 5 * time.Minute

// sweepInterval is how often idle sessions and stale endpoints are swept.
var sweepInterval = 10 * time.Second

// Config holds the configuration for the ThunderAgent plugin.
type Config struct {
	// FallbackCapacityTokens is the per-endpoint KV cache capacity in tokens
	// used when the endpoint does not report cache_config_info through the
	// datalayer.
	FallbackCapacityTokens int64 `json:"fallbackCapacityTokens"`
	// UsableCapacityRatio is the fraction of a pod's KV capacity available to
	// sessions (1.0 = 100%).
	UsableCapacityRatio float64 `json:"usableCapacityRatio"`
	// IdleGracePeriod is how long an idle session keeps its room after its
	// last response before another session can reclaim it. 0s reclaims idle
	// sessions immediately.
	IdleGracePeriod metav1.Duration `json:"idleGracePeriod"`
	// ForceAdmitTimeout is the maximum queue wait before a request is admitted
	// even when no pod has room. 0s disables forced admission.
	ForceAdmitTimeout metav1.Duration `json:"forceAdmitTimeout"`
}

func defaultConfig() Config {
	return Config{
		FallbackCapacityTokens: 4194304,
		UsableCapacityRatio:    1.0,
		IdleGracePeriod:        metav1.Duration{Duration: 30 * time.Second},
		ForceAdmitTimeout:      metav1.Duration{Duration: defaultForceAdmitTimeout},
	}
}

func (c Config) validate() error {
	if c.FallbackCapacityTokens <= 0 {
		return fmt.Errorf("fallbackCapacityTokens must be > 0, got %d", c.FallbackCapacityTokens)
	}
	if !(c.UsableCapacityRatio > 0 && c.UsableCapacityRatio <= 1) {
		return fmt.Errorf("usableCapacityRatio must be in (0, 1], got %v", c.UsableCapacityRatio)
	}
	if c.IdleGracePeriod.Duration < 0 {
		return fmt.Errorf("idleGracePeriod must be >= 0, got %v", c.IdleGracePeriod.Duration)
	}
	if c.ForceAdmitTimeout.Duration < 0 {
		return fmt.Errorf("forceAdmitTimeout must be >= 0, got %v", c.ForceAdmitTimeout.Duration)
	}
	return nil
}
