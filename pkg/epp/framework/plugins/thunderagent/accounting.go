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

package thunderagent

import (
	"context"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// PreRequest attributes the request's estimated prompt tokens to the picked
// endpoint's program, binds (or rebinds) the program to that endpoint, and
// stashes the estimate on the request for ResponseBody to remove. The
// director guarantees ResponseBody runs (with EndOfStream) for every request
// that picked a pod, including aborted ones, so the estimate cannot leak.
// The session-final and parent-session headers are read here so no separate
// header hook is needed.
func (a *ThunderAgent) PreRequest(ctx context.Context, request *fwksched.InferenceRequest, schedulingResult *fwksched.SchedulingResult) error {
	id := programID(request)
	if id == "" {
		return nil
	}
	podName := a.pickedPodName(schedulingResult)
	if podName == "" {
		return nil
	}

	now := time.Now()
	t := a.table
	t.mu.Lock()
	estimate := t.estimateTokens(request.RequestSizeBytes)
	st, ok := t.programs[id]
	if !ok {
		st = &program{}
		t.programs[id] = st
	}
	delete(t.pending, id) // the program is bound and counted; drop its admission reservation
	rebound := st.podName != "" && st.podName != podName
	st.podName = podName
	st.inflightTokens += estimate
	st.dispatchCount++
	st.lastActivity = now
	if isSessionFinal(request, a.sessionFinalHeader) {
		st.finalSeen = true
	}
	if parent := request.Headers[a.parentSessionHeader]; parent != "" && st.parentID == "" {
		st.parentID = parent
	}
	t.mu.Unlock()

	if rebound {
		a.metrics.rebinds.Inc()
		log.FromContext(ctx).V(logutil.DEBUG).Info("thunderagent.rebind", "pod", podName)
	}
	request.PutAttribute(inflightEstimateKey, estimate)
	return nil
}

// ResponseBody acts on the final stream chunk only. It removes the in-flight
// estimate recorded by PreRequest, replaces the program's committed footprint
// with the usage-reported total, refines the size-to-token estimator, and
// releases the program when its session-final turn completes.
func (a *ThunderAgent) ResponseBody(ctx context.Context, request *fwksched.InferenceRequest, response *fwkrc.Response, _ *datalayer.EndpointMetadata) {
	if request == nil || response == nil || !response.EndOfStream {
		return
	}
	id := programID(request)
	if id == "" {
		return
	}
	estimate, _ := fwksched.ReadRequestAttribute[int64](request, inflightEstimateKey)

	now := time.Now()
	t := a.table
	t.mu.Lock()

	t.refineEstimator(request.RequestSizeBytes, response.Usage.PromptTokens)

	st, ok := t.programs[id]
	if !ok {
		t.mu.Unlock()
		return
	}
	st.inflightTokens -= estimate
	if st.inflightTokens < 0 {
		st.inflightTokens = 0
	}
	switch {
	case response.Usage.TotalTokens > 0:
		st.committedTokens = int64(response.Usage.TotalTokens)
	case estimate > st.committedTokens:
		st.committedTokens = estimate
	}
	st.lastResponseAt = now
	st.lastActivity = now

	release := st.finalSeen && st.inflightTokens == 0
	var freed int64
	if release {
		freed = st.committedTokens
		delete(t.programs, id)
	}
	t.mu.Unlock()

	if release {
		a.metrics.sessionFinalReleases.Inc()
		log.FromContext(ctx).Info("thunderagent.release_final", "freed_tokens", freed)
	}
}

// pickedPodName returns the pod chosen for this instance's profile, or ""
// when that profile was not scheduled. An empty profileName selects the
// primary (decode) profile.
func (a *ThunderAgent) pickedPodName(schedulingResult *fwksched.SchedulingResult) string {
	if schedulingResult == nil {
		return ""
	}
	profileName := a.profileName
	if profileName == "" {
		profileName = schedulingResult.PrimaryProfileName
	}
	result, ok := schedulingResult.ProfileResults[profileName]
	if !ok || result == nil || len(result.TargetEndpoints) == 0 || result.TargetEndpoints[0] == nil {
		return ""
	}
	if md := result.TargetEndpoints[0].GetMetadata(); md != nil {
		return md.ID.String()
	}
	return ""
}
