/*
Copyright 2026 The Kubernetes Authors.
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

package runner

import (
	"context"
	"encoding/hex"
	"errors"
	"fmt"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	configapiv1 "github.com/llm-d/llm-d-router/apix/config/v1"
	"github.com/llm-d/llm-d-router/pkg/epp/config"
	"github.com/llm-d/llm-d-router/pkg/epp/config/loader"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
	"github.com/llm-d/llm-d-router/pkg/epp/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/scheduling"
	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"
)

func (r *Runner) newRuntimeScheduler(opts *runserver.Options, initial *scheduling.Scheduler) requestcontrol.Scheduler {
	if !opts.WatchConfigFile {
		return initial
	}

	reloadable := scheduling.NewReloadableScheduler(initial)
	r.reloadableScheduler = reloadable
	metrics.RecordConfigGeneration(reloadable.Generation())
	return reloadable
}

func (r *Runner) startConfigWatcher(ctx context.Context, opts *runserver.Options) error {
	if r.reloadableScheduler == nil {
		return nil
	}

	reloadable := r.reloadableScheduler
	profiles := append([]configapiv1.SchedulingProfile(nil), r.rawConfig.SchedulingProfiles...)
	logger := log.FromContext(ctx).WithName("config-reloader").WithValues("path", opts.ConfigFile)
	watcher, err := config.NewFileWatcher(opts.ConfigFile, r.startupConfigBytes, func(attempt config.FileAttempt) {
		started := time.Now()
		if attempt.Err != nil {
			result := "read_error"
			if attempt.WatchError {
				result = "watch_error"
			}
			metrics.RecordConfigReload(result)
			logger.Error(attempt.Err, "config reload failed", "result", result, "generation", reloadable.Generation(), "duration", time.Since(started))
			return
		}

		hash := hex.EncodeToString(attempt.Hash[:6])
		candidate, buildErr := loader.BuildSchedulerForReload(attempt.Content, r.startupSourceConfig, profiles, r.PluginHandle, opts.FeatureGates...)
		if buildErr != nil {
			result := "invalid"
			if errors.Is(buildErr, loader.ErrUnsupportedReload) {
				result = "unsupported"
			}
			metrics.RecordConfigReload(result)
			logger.Error(buildErr, "config reload failed", "result", result, "generation", reloadable.Generation(), "duration", time.Since(started), "contentHash", hash)
			return
		}
		if ctx.Err() != nil {
			return
		}

		generation := reloadable.Replace(candidate)
		metrics.RecordConfigReload("success")
		metrics.RecordConfigReloadSuccess(generation)
		logger.Info("config reload completed", "result", "success", "generation", generation, "duration", time.Since(started), "contentHash", hash)
	})
	if err != nil {
		return fmt.Errorf("failed to watch config file: %w", err)
	}
	go watcher.Run(ctx)
	logger.Info("watching config file for scheduling profile changes")
	return nil
}
