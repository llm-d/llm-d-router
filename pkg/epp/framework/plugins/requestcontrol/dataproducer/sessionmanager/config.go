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
	"errors"
	"strings"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/sessionidentity"
)

const maxConfigurationLength = 1024

// Config configures the session-manager plugin.
type Config struct {
	SessionIdentityProducer string `json:"sessionIdentityProducer,omitempty"`
}

type resolvedConfig struct {
	sessionIdentityProducer string
}

func (c Config) resolve() (resolvedConfig, error) {
	cfg := resolvedConfig{
		sessionIdentityProducer: strings.TrimSpace(c.SessionIdentityProducer),
	}
	if cfg.sessionIdentityProducer == "" {
		cfg.sessionIdentityProducer = sessionidentity.PluginType
	}
	if c.SessionIdentityProducer != "" && c.SessionIdentityProducer != strings.TrimSpace(c.SessionIdentityProducer) {
		return resolvedConfig{}, errors.New("sessionIdentityProducer must have no surrounding whitespace")
	}
	if len(cfg.sessionIdentityProducer) > maxConfigurationLength {
		return resolvedConfig{}, errors.New("sessionIdentityProducer must be at most 1024 bytes")
	}
	return cfg, nil
}
