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
	"fmt"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-inference-scheduler/pkg/common/observability/logging"
	fwksched "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/metrics"
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
		fallbacks:      config.fallbacks,
	}
}

type Scheduler struct {
	profileHandler fwksched.ProfileHandler
	profiles       map[string]fwksched.SchedulerProfile
	// fallbacks maps a primary profile name to the SchedulerProfile that should run
	// in its place when the primary's Run reports NoSignal=true. See #1139.
	fallbacks map[string]fwksched.SchedulerProfile
}

// Schedule finds the target pod based on metrics and the requested lora adapter.
func (s *Scheduler) Schedule(ctx context.Context, request *fwksched.InferenceRequest, candidateEndpoints []fwksched.Endpoint) (result *fwksched.SchedulingResult, err error) {
	loggerVerbose := log.FromContext(ctx).V(logutil.VERBOSE)

	scheduleStart := time.Now()
	defer func() {
		metrics.RecordSchedulerE2ELatency(time.Since(scheduleStart))
		metrics.RecordSchedulerAttempt(err, request.TargetModel, result)
	}()

	profileRunResults := map[string]*fwksched.ProfileRunResult{}
	cycleState := fwksched.NewCycleState()

	for { // get the next set of profiles to run iteratively based on the request and the previous execution results
		loggerVerbose.Info("Running profile handler, Pick profiles", "plugin", s.profileHandler.TypedName())
		before := time.Now()
		profiles := s.profileHandler.Pick(ctx, cycleState, request, s.profiles, profileRunResults)
		metrics.RecordPluginProcessingLatency(profilePickerExtensionPoint, s.profileHandler.TypedName().Type, s.profileHandler.TypedName().Name, time.Since(before))
		loggerVerbose.Info("Completed running profile handler Pick profiles successfully", "plugin", s.profileHandler.TypedName(), "result", profiles)
		if len(profiles) == 0 { // profile picker didn't pick any profile to run
			break
		}

		for name, profile := range profiles {
			loggerVerbose.Info("Running scheduler profile", "profile", name)
			// run the selected profiles and collect results (current code runs all profiles)
			profileRunResult, err := profile.Run(ctx, request, cycleState, candidateEndpoints)
			if err != nil {
				loggerVerbose.Info("failed to run scheduler profile", "profile", name, "error", err.Error())
			} else {
				loggerVerbose.Info("Completed running scheduler profile succuessfully", "profile", name)
			}

			// Fallback chaining (#1139): if the profile reported no usable signal
			// and a fallback is configured for it, run the fallback and substitute
			// its result. ProfileHandler stays unaware that fallback was used.
			if profileRunResult != nil && profileRunResult.NoSignal {
				if fb, ok := s.fallbacks[name]; ok {
					loggerVerbose.Info("Primary profile produced no usable signal; running fallback", "primary", name)
					fbResult, fbErr := fb.Run(ctx, request, cycleState, candidateEndpoints)
					if fbErr != nil {
						loggerVerbose.Info("fallback profile failed to run", "primary", name, "error", fbErr.Error())
					} else {
						profileRunResult = fbResult
					}
				}
			}

			profileRunResults[name] = profileRunResult // if profile failed to run, the run result is nil
		}
	}

	if len(profileRunResults) == 0 {
		err = fmt.Errorf("failed to run any scheduler profile for request %s", request.RequestID)
		return nil, err
	}

	loggerVerbose.Info("Running profile handler, ProcessResults", "plugin", s.profileHandler.TypedName())
	before := time.Now()
	result, err = s.profileHandler.ProcessResults(ctx, cycleState, request, profileRunResults)
	metrics.RecordPluginProcessingLatency(processProfilesResultsExtensionPoint, s.profileHandler.TypedName().Type, s.profileHandler.TypedName().Name, time.Since(before))
	loggerVerbose.Info("Completed running profile handler ProcessResults successfully", "plugin", s.profileHandler.TypedName())

	return result, err
}
