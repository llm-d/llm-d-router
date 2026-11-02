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

package prefixcacheaware

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// Config holds the parameters of the prefix-cache-aware admitter.
type Config struct {
	// SaturationDetectorRef names the saturation detector plugin that supplies the
	// pool saturation S, typically the concurrency-detector flow control already
	// uses. Any plugin that implements flowcontrol.SaturationDetector works. Required.
	SaturationDetectorRef string `json:"saturationDetectorRef" pluginRef:""`

	// ShedThreshold is T. Shedding starts when S >= T. Range (0, 1]. Default 0.85.
	ShedThreshold float64 `json:"shedThreshold,omitempty"`

	// MinHitRatio is H. While shedding, a request is admitted only if some
	// endpoint reports a prefix cache hit ratio >= H for it. Range [0, 1]. Default 0.6.
	MinHitRatio float64 `json:"minHitRatio,omitempty"`

	// MinUncachedTokens exempts small requests. A request whose smallest uncached
	// token count across the endpoints is below this value is never shed.
	// 0 disables the exemption. Default 0.
	MinUncachedTokens int64 `json:"minUncachedTokens,omitempty"`

	// ShedStatusCode is the HTTP status of a shed response: 429 (the code flow
	// control uses for a saturated pool, so clients see one kind of backpressure)
	// or 503. Default 429.
	ShedStatusCode int `json:"shedStatusCode,omitempty"`

	// PrefixMatchInfoProducerName selects the producer of PrefixCacheMatchInfo.
	// Empty uses the approximate prefix cache producer.
	PrefixMatchInfoProducerName string `json:"prefixMatchInfoProducerName,omitempty"`

	// InFlightLoadProducerName selects the producer of InFlightLoad and
	// UncachedRequestTokens. It must match the detector's producer. Empty uses
	// the in-flight load producer.
	InFlightLoadProducerName string `json:"inFlightLoadProducerName,omitempty"`
}

// DefaultConfig holds the defaults applied before decoding parameters.
var DefaultConfig = Config{
	ShedThreshold:  0.85,
	MinHitRatio:    0.6,
	ShedStatusCode: http.StatusTooManyRequests,
}

func (c *Config) validate() error {
	if c.SaturationDetectorRef == "" {
		return errors.New("saturationDetectorRef is required")
	}
	if c.ShedThreshold <= 0 || c.ShedThreshold > 1.0 {
		return fmt.Errorf("shedThreshold must be in (0, 1], got %f", c.ShedThreshold)
	}
	if c.MinHitRatio < 0 || c.MinHitRatio > 1.0 {
		return fmt.Errorf("minHitRatio must be in [0, 1], got %f", c.MinHitRatio)
	}
	if c.MinUncachedTokens < 0 {
		return fmt.Errorf("minUncachedTokens must be >= 0, got %d", c.MinUncachedTokens)
	}
	if _, err := c.errorCode(); err != nil {
		return err
	}
	return nil
}

// errorCode maps ShedStatusCode to the canonical error code the Director turns
// into that HTTP status.
func (c *Config) errorCode() (string, error) {
	switch c.ShedStatusCode {
	case http.StatusServiceUnavailable:
		return errcommon.ServiceUnavailable, nil
	case http.StatusTooManyRequests:
		return errcommon.ResourceExhausted, nil
	default:
		return "", fmt.Errorf("shedStatusCode must be 429 or 503, got %d", c.ShedStatusCode)
	}
}

// ConfigParser decodes and validates the parameters. It is registered with the
// plugin registry so the config loader instantiates the referenced detector
// before this admitter.
func ConfigParser(rawParameters *json.Decoder, _ fwkplugin.Handle) (any, error) {
	config := DefaultConfig
	if rawParameters != nil {
		if err := rawParameters.Decode(&config); err != nil {
			return nil, fmt.Errorf("failed to unmarshal config: %w", err)
		}
	}
	if err := config.validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}
	return &config, nil
}
