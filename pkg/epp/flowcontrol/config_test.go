/*
Copyright 2025 The Kubernetes Authors.

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

package flowcontrol

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/controller"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/registry"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/usagelimits"
)

func TestNewConfig(t *testing.T) {
	t.Parallel()

	t.Run("all fields are correctly assigned", func(t *testing.T) {
		t.Parallel()

		ctrl := &controller.Config{EnqueueChannelBufferSize: 42}
		reg := &registry.Config{MaxBytes: 1024}
		ulp := usagelimits.DefaultPolicy()

		cfg := NewConfig(ctrl, reg, ulp)

		assert.NotNil(t, cfg, "NewConfig should return a non-nil Config")
		assert.Same(t, ctrl, cfg.Controller, "Controller should be the same pointer passed in")
		assert.Same(t, reg, cfg.Registry, "Registry should be the same pointer passed in")
		assert.Same(t, ulp, cfg.UsageLimitPolicy, "UsageLimitPolicy should be the same pointer passed in")
	})

	t.Run("nil values are handled gracefully", func(t *testing.T) {
		t.Parallel()

		cfg := NewConfig(nil, nil, nil)

		assert.NotNil(t, cfg, "NewConfig should return a non-nil Config even when all arguments are nil")
		assert.Nil(t, cfg.Controller, "Controller should be nil when nil was passed")
		assert.Nil(t, cfg.Registry, "Registry should be nil when nil was passed")
		assert.Nil(t, cfg.UsageLimitPolicy, "UsageLimitPolicy should be nil when nil was passed")
	})
}
