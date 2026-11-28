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

package qoscontract

import (
	"errors"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/prometheus/common/model"
)

const (
	defaultValidationInterval = "1m"
	defaultValidationWindow   = "5m"
	defaultMinRequests        = 20
)

// PromiseConfig is a team's latency promise. Each set objective holds for the given percentile of
// the team's requests served within its contract, over the validation window.
type PromiseConfig struct {
	// TTFT is the time to first token, from receipt at the EPP, including queue wait.
	TTFT *LatencyObjective `json:"ttft,omitempty"`
	// TPOT is the mean time per output token of each streamed request.
	TPOT *LatencyObjective `json:"tpot,omitempty"`
	// ITL is the time between consecutive streamed chunks.
	ITL *LatencyObjective `json:"itl,omitempty"`
}

// LatencyObjective is one latency promise: Percentile of requests complete within ThresholdMs.
type LatencyObjective struct {
	// Percentile is the share of requests that must meet ThresholdMs, as "p95" or "p99.9".
	Percentile string `json:"percentile"`
	// ThresholdMs is the latency bound, in milliseconds.
	ThresholdMs float64 `json:"thresholdMs"`
}

// SLOValidationConfig configures how team promises are validated against the latency histograms
// this EPP records. Every field has a default.
type SLOValidationConfig struct {
	// Interval is how often the promises are validated. Default "1m".
	Interval string `json:"interval,omitempty"`
	// Window is the stretch of recent traffic a promise is evaluated over. It must be at least
	// Interval. Default "5m".
	Window string `json:"window,omitempty"`
	// MinRequests is the fewest requests in the window for a verdict. Below it the status is
	// insufficient_data. Default 20.
	MinRequests int `json:"minRequests,omitempty"`
}

func (p *PromiseConfig) validate() error {
	objectives := p.objectives()
	if len(objectives) == 0 {
		return errors.New("at least one of ttft, tpot and itl is required")
	}
	for _, o := range objectives {
		if _, err := parsePercentile(o.Percentile); err != nil {
			return fmt.Errorf("%s: %w", o.slo, err)
		}
		if !(o.ThresholdMs > 0) {
			return fmt.Errorf("%s: thresholdMs must be > 0, got %v", o.slo, o.ThresholdMs)
		}
	}
	return nil
}

var percentilePattern = regexp.MustCompile(`^p[0-9]{1,2}(\.[0-9]{1,3})?$`)

// parsePercentile converts "p95" or "p99.9" to a quantile in (0, 1).
func parsePercentile(s string) (float64, error) {
	if !percentilePattern.MatchString(s) {
		return 0, fmt.Errorf("percentile must look like p95 or p99.9, got %q", s)
	}
	// Shift the decimal point in the string rather than divide by 100, so p99.9 is exactly the
	// float nearest 0.999.
	whole, frac, _ := strings.Cut(s[1:], ".")
	if len(whole) == 1 {
		whole = "0" + whole
	}
	v, err := strconv.ParseFloat("0."+whole+frac, 64)
	if err != nil || v <= 0 {
		return 0, fmt.Errorf("percentile must be above p0, got %q", s)
	}
	return v, nil
}

// sloValidation is the resolved SLOValidationConfig.
type sloValidation struct {
	interval    time.Duration
	window      time.Duration
	minRequests float64
}

// resolve validates the configuration and applies the defaults.
func (c *SLOValidationConfig) resolve() (sloValidation, error) {
	out := sloValidation{minRequests: defaultMinRequests}
	for _, d := range []struct {
		field, value, def string
		dst               *time.Duration
	}{
		{"interval", c.Interval, defaultValidationInterval, &out.interval},
		{"window", c.Window, defaultValidationWindow, &out.window},
	} {
		v := d.value
		if v == "" {
			v = d.def
		}
		// Prometheus duration syntax, which also accepts days and weeks (1d, 1w).
		parsed, err := model.ParseDuration(v)
		if err != nil || parsed <= 0 {
			return out, fmt.Errorf("%s must be a positive duration such as 5m, got %q", d.field, v)
		}
		*d.dst = time.Duration(parsed)
	}
	if out.window < out.interval {
		return out, fmt.Errorf("window %s must be at least interval %s", out.window, out.interval)
	}
	if c.MinRequests < 0 {
		return out, fmt.Errorf("minRequests must be >= 0, got %d", c.MinRequests)
	}
	if c.MinRequests > 0 {
		out.minRequests = float64(c.MinRequests)
	}
	return out, nil
}
