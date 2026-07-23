// Copyright 2025 The Kubernetes Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

package runner

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	ctrl "sigs.k8s.io/controller-runtime"

	"github.com/llm-d/llm-d-router/pkg/epp/disaggregation"
)

// registerDisaggregation instantiates the disaggregation controller if the
// rawConfig contains a disaggregation block and hands it to WireInto, which
// does the correct-position registration in the scheduler and request-control
// pipelines. No-op when the block is absent or disabled.
func (r *Runner) registerDisaggregation(ctx context.Context, mgr ctrl.Manager, namespace string) error {
	if r.rawConfig == nil || r.rawConfig.Disaggregation == nil {
		return nil
	}

	var config disaggregation.Config
	if err := json.Unmarshal(*r.rawConfig.Disaggregation, &config); err != nil {
		return fmt.Errorf("parse disaggregation config: %w", err)
	}
	if !config.Enabled {
		return nil
	}

	controller, err := disaggregation.Register(ctx, mgr, namespace, config)
	if err != nil {
		if errors.Is(err, disaggregation.ErrDisabled) {
			return nil
		}
		return fmt.Errorf("register disaggregation controller: %w", err)
	}
	return disaggregation.WireInto(r.schedulerConfig, r.requestControlConfig, controller)
}
