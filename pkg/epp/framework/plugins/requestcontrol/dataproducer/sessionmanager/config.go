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

package sessionmanager

import (
	"encoding/base64"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"
)

const (
	defaultBindingTTL      = 5 * time.Minute
	defaultMaxBindings     = 100_000
	maxBindingsLimit       = 1_000_000
	maxConfigurationLength = 1024
	sessionManagerKeyBytes = 32
)

// Config configures the session-manager plugin.
type Config struct {
	DeploymentID            string `json:"deploymentID"`
	HMACKeyFile             string `json:"hmacKeyFile"`
	TokenProducer           string `json:"tokenProducer,omitempty"`
	EventCorrelationEnabled bool   `json:"eventCorrelationEnabled,omitempty"`
	BindingTTL              string `json:"bindingTTL,omitempty"`
	MaxBindings             int    `json:"maxBindings,omitempty"`
}

type resolvedConfig struct {
	deploymentID            string
	hmacKey                 []byte
	tokenProducer           string
	eventCorrelationEnabled bool
	bindingTTL              time.Duration
	maxBindings             int
}

func (c Config) resolve() (resolvedConfig, error) {
	cfg := resolvedConfig{
		deploymentID:            c.DeploymentID,
		tokenProducer:           strings.TrimSpace(c.TokenProducer),
		eventCorrelationEnabled: c.EventCorrelationEnabled,
		bindingTTL:              defaultBindingTTL,
		maxBindings:             defaultMaxBindings,
	}
	if strings.TrimSpace(cfg.deploymentID) == "" ||
		cfg.deploymentID != strings.TrimSpace(cfg.deploymentID) ||
		len(cfg.deploymentID) > maxConfigurationLength {
		return resolvedConfig{}, errors.New("deploymentID must be non-empty, have no surrounding whitespace, and be at most 1024 bytes")
	}
	keyPath := strings.TrimSpace(c.HMACKeyFile)
	if keyPath == "" {
		return resolvedConfig{}, errors.New("hmacKeyFile is required")
	}
	encoded, err := os.ReadFile(keyPath)
	if err != nil {
		return resolvedConfig{}, fmt.Errorf("read hmacKeyFile: %w", err)
	}
	cfg.hmacKey, err = base64.RawURLEncoding.DecodeString(strings.TrimSpace(string(encoded)))
	if err != nil || len(cfg.hmacKey) != sessionManagerKeyBytes {
		return resolvedConfig{}, errors.New("hmacKeyFile must contain unpadded base64url for exactly 32 bytes")
	}
	if c.BindingTTL != "" {
		cfg.bindingTTL, err = time.ParseDuration(c.BindingTTL)
		if err != nil {
			return resolvedConfig{}, fmt.Errorf("parse bindingTTL: %w", err)
		}
	}
	if cfg.bindingTTL <= 0 || cfg.bindingTTL > time.Hour {
		return resolvedConfig{}, errors.New("bindingTTL must be greater than zero and at most 1h")
	}
	if c.MaxBindings != 0 {
		cfg.maxBindings = c.MaxBindings
	}
	if cfg.maxBindings <= 0 || cfg.maxBindings > maxBindingsLimit {
		return resolvedConfig{}, fmt.Errorf("maxBindings must be between 1 and %d", maxBindingsLimit)
	}
	if cfg.eventCorrelationEnabled && cfg.tokenProducer == "" {
		return resolvedConfig{}, errors.New("tokenProducer is required when eventCorrelationEnabled is true")
	}
	return cfg, nil
}
