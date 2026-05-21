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

package scheduling

import (
	fwkrhapi "github.com/llm-d/llm-d-router/pkg/epp/framework/requesthandler/types"
	"context"

)

const nilString = "<nil>"

type Modality = fwkrhapi.Modality

const ModalityImage = fwkrhapi.ModalityImage

type TokenizedPrompt = fwkrhapi.TokenizedPrompt

type MultiModalFeature = fwkrhapi.MultiModalFeature

// RequestObjectives represents the scheduling objectives parsed from the InferenceObjectiveSpec, to be used in scheduling decisions.
type RequestObjectives = fwkrhapi.RequestObjectives

// InferenceRequest is a structured representation of the fields we parse out of the InferenceRequest body.
type InferenceRequest = fwkrhapi.InferenceRequest

type Endpoint = fwkrhapi.Endpoint

type ScoredEndpoint = fwkrhapi.ScoredEndpoint

// ProfileRunResult captures the profile run result.
type ProfileRunResult = fwkrhapi.ProfileRunResult

// SchedulingResult captures the result of the scheduling cycle.
type SchedulingResult = fwkrhapi.SchedulingResult

type SchedulerProfile interface {
	Run(ctx context.Context, request *InferenceRequest, cycleState *CycleState, candidateEndpoints []Endpoint) (*ProfileRunResult, error)
}

var NewEndpoint = fwkrhapi.NewEndpoint
var EndpointComparer = fwkrhapi.EndpointComparer
var ScoredEndpointComparer = fwkrhapi.ScoredEndpointComparer
