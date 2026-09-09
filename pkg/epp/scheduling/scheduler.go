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

// Package scheduling implements request scheduling algorithms.
package scheduling

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

const (
	profilePickerExtensionPoint          = "ProfilePicker"
	filterExtensionPoint                 = "Filter"
	scorerExtensionPoint                 = "Scorer"
	pickerExtensionPoint                 = "Picker"
	processProfilesResultsExtensionPoint = "ProcessProfilesResults"
)

// NewSchedulerWithConfig returns a new scheduler with the given scheduler plugins configuration.
func NewSchedulerWithConfig(config *SchedulerConfig) *Scheduler {
	return &Scheduler{
		profileHandler: config.profileHandler,
		profiles:       config.profiles,
	}
}

type Scheduler struct {
	profileHandler fwksched.ProfileHandler
	profiles       map[string]fwksched.SchedulerProfile
}

// Schedule finds the target pod based on metrics and the requested lora adapter.
func (s *Scheduler) Schedule(ctx context.Context, request *fwksched.InferenceRequest, candidateEndpoints []fwksched.Endpoint) (result *fwksched.SchedulingResult, err error) {
	loggerVerbose := log.FromContext(ctx).V(logutil.VERBOSE)
	verboseEnabled := loggerVerbose.Enabled()
	handlerName := s.profileHandler.TypedName()

	scheduleStart := time.Now()
	defer func() {
		metrics.RecordSchedulerE2ELatency(time.Since(scheduleStart))
		metrics.RecordSchedulerAttempt(err, request.TargetModel, result)
	}()

	profileRunResults := map[string]*fwksched.ProfileRunResult{}
	// Keyed like profileRunResults so a profile that fails in one iteration and
	// succeeds in a later one leaves no stale error behind. Nil until a profile
	// fails: delete and len are no-ops on a nil map, and the happy path skips
	// the allocation.
	var profileRunErrors map[string]error

	for { // get the next set of profiles to run iteratively based on the request and the previous execution results
		if verboseEnabled {
			loggerVerbose.Info("Running profile handler, Pick profiles", "plugin", handlerName)
		}
		before := time.Now()
		profiles := s.profileHandler.Pick(ctx, request, s.profiles, profileRunResults)
		metrics.RecordPluginProcessingLatency(profilePickerExtensionPoint, handlerName.Type, handlerName.Name, time.Since(before))
		if verboseEnabled {
			loggerVerbose.Info("Completed running profile handler Pick profiles successfully", "plugin", handlerName, "result", profiles)
		}
		if len(profiles) == 0 { // profile picker didn't pick any profile to run
			break
		}

		for name, profile := range profiles {
			if verboseEnabled {
				loggerVerbose.Info("Running scheduler profile", "profile", name)
			}
			// run the selected profiles and collect results (current code runs all profiles)
			profileRunResult, err := runSchedulerProfile(ctx, name, profile, request, candidateEndpoints)
			if err != nil {
				if verboseEnabled {
					loggerVerbose.Info("failed to run scheduler profile", "profile", name, "error", err.Error())
				}
				if profileRunErrors == nil {
					profileRunErrors = map[string]error{}
				}
				profileRunErrors[name] = fmt.Errorf("profile %q: %w", name, err)
			} else {
				if verboseEnabled {
					loggerVerbose.Info("Completed running scheduler profile successfully", "profile", name)
				}
				delete(profileRunErrors, name)
			}

			profileRunResults[name] = profileRunResult // if profile failed to run, the run result is nil
		}
	}

	if len(profileRunResults) == 0 {
		err = fmt.Errorf("failed to run any scheduler profile for request %s", request.RequestID)
		return nil, err
	}

	if verboseEnabled {
		loggerVerbose.Info("Running profile handler, ProcessResults", "plugin", handlerName)
	}
	before := time.Now()
	result, err = s.profileHandler.ProcessResults(ctx, request, profileRunResults)
	metrics.RecordPluginProcessingLatency(processProfilesResultsExtensionPoint, handlerName.Type, handlerName.Name, time.Since(before))
	if verboseEnabled {
		loggerVerbose.Info("Completed running profile handler ProcessResults successfully", "plugin", handlerName)
	}

	// Profile handlers see failed profiles only as nil results and report them
	// with fresh untyped errors. Join the retained profile errors so a typed
	// errcommon.Error raised inside a profile run (e.g. filters draining the
	// candidate set) stays reachable via errors.As in the caller.
	if err != nil && len(profileRunErrors) > 0 {
		errs := make([]error, 0, len(profileRunErrors)+1)
		errs = append(errs, err)
		// Sorted so error composition, and therefore errors.As selection when
		// profiles fail with different typed codes, is deterministic.
		for _, name := range slices.Sorted(maps.Keys(profileRunErrors)) {
			errs = append(errs, profileRunErrors[name])
		}
		err = errors.Join(errs...)
	}

	return result, err
}

func runSchedulerProfile(ctx context.Context, name string, profile fwksched.SchedulerProfile,
	request *fwksched.InferenceRequest, candidateEndpoints []fwksched.Endpoint,
) (*fwksched.ProfileRunResult, error) {
	profileCtx, span := tracing.Tracer(TracerScope).Start(ctx, "run_scheduler_profile",
		trace.WithSpanKind(trace.SpanKindInternal),
		trace.WithAttributes(attribute.String("llm_d.epp.scheduling.profile.name", name)),
	)
	defer span.End()

	return profile.Run(profileCtx, request, candidateEndpoints)
}
