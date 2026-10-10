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
)

const (
	maxConfigurationLength = 1024
	sessionManagerKeyBytes = 32
)

// Config configures the session-manager plugin.
type Config struct {
	DeploymentID string `json:"deploymentID"`
	HMACKeyFile  string `json:"hmacKeyFile"`
}

type resolvedConfig struct {
	deploymentID string
	hmacKey      []byte
}

func (c Config) resolve() (resolvedConfig, error) {
	cfg := resolvedConfig{
		deploymentID: c.DeploymentID,
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
	return cfg, nil
}
