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

// Package servedmodel provides a filter that keeps only the endpoints whose
// model server lists the requested model at GET /v1/models.
//
// The list comes from the models-data-extractor. A vLLM endpoint lists its
// base model and every LoRA adapter registered with it.
package servedmodel

import (
	"context"
	"encoding/json"
	"fmt"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmodels "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/models"
)

const (
	ServedModelFilterType = "served-model-filter"

	onMissingPass = "Pass"
	onMissingFail = "Fail"
)

type parameters struct {
	// OnMissing controls endpoints without a stored model list, which is the
	// case until the first successful poll after the endpoint is added.
	// "Pass" (default) keeps them, except for an adapter request that an
	// endpoint with a list serves. "Fail" excludes them.
	OnMissing       string `json:"onMissing"`
	FallbackOnEmpty bool   `json:"fallbackOnEmpty"`
}

var (
	_ scheduling.Filter     = &ServedModelFilter{}
	_ plugin.ConsumerPlugin = &ServedModelFilter{}
)

func ServedModelFilterFactory(name string, rawParameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	var params parameters
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' filter - %w", ServedModelFilterType, err)
		}
	}
	if handle != nil {
		if err := registerMetrics(handle.Metrics()); err != nil {
			return nil, err
		}
	}
	return NewServedModelFilter(name, params)
}

func NewServedModelFilter(name string, params parameters) (*ServedModelFilter, error) {
	if name == "" {
		name = ServedModelFilterType
	}
	switch params.OnMissing {
	case "":
		params.OnMissing = onMissingPass
	case onMissingPass, onMissingFail:
	default:
		return nil, fmt.Errorf("%s onMissing must be %q or %q, got %q",
			ServedModelFilterType, onMissingPass, onMissingFail, params.OnMissing)
	}
	initMetrics(name)
	return &ServedModelFilter{
		typedName:       plugin.TypedName{Type: ServedModelFilterType, Name: name},
		dataKey:         attrmodels.ModelsAttributeKey,
		passOnMissing:   params.OnMissing == onMissingPass,
		fallbackOnEmpty: params.FallbackOnEmpty,
	}, nil
}

type ServedModelFilter struct {
	typedName       plugin.TypedName
	dataKey         plugin.DataKey
	passOnMissing   bool
	fallbackOnEmpty bool
}

func (f *ServedModelFilter) TypedName() plugin.TypedName {
	return f.typedName
}

// Consumes declares the models attribute as required. models-data-extractor is
// not a default producer, so a configuration without it fails at start-up.
func (f *ServedModelFilter) Consumes() plugin.DataDependencies {
	return plugin.DataDependencies{
		Required: map[plugin.DataKey]any{
			f.dataKey: attrmodels.ModelDataCollection{},
		},
	}
}

func (f *ServedModelFilter) Filter(ctx context.Context, request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) []scheduling.Endpoint {
	if request == nil || request.TargetModel == "" {
		recordDecision(f.typedName.Name, outcomeNotApplicable)
		return endpoints
	}

	var serving, unlisted []scheduling.Endpoint
	adapter := false
	for _, endpoint := range endpoints {
		models, ok := fwkdl.ReadAttribute[attrmodels.ModelDataCollection](endpoint, f.dataKey)
		if !ok {
			unlisted = append(unlisted, endpoint)
			continue
		}
		if model, ok := find(models, request.TargetModel); ok {
			serving = append(serving, endpoint)
			// vLLM sets parent on LoRA adapter entries only.
			adapter = adapter || model.Parent != ""
		}
	}
	recordCandidates(f.typedName.Name, len(endpoints), len(unlisted))

	if len(serving) > 0 {
		recordDecision(f.typedName.Name, outcomeListed)
		// An endpoint that has not reported a list is unlikely to have an adapter
		// registered that another endpoint serves. Every endpoint serving a base
		// model lists it, so dropping unlisted endpoints for a base model would
		// only concentrate load on the endpoints that have reported.
		if f.passOnMissing && !adapter {
			return append(serving, unlisted...)
		}
		return serving
	}

	log.FromContext(ctx).V(logutil.DEBUG).Info("ServedModelFilter: no endpoint lists the target model",
		"model", request.TargetModel, "candidates", len(endpoints), "withoutModelList", len(unlisted))
	switch {
	case f.passOnMissing && len(unlisted) > 0:
		recordDecision(f.typedName.Name, outcomeUnlisted)
		return unlisted
	case f.fallbackOnEmpty:
		recordDecision(f.typedName.Name, outcomeFallback)
		return endpoints
	default:
		recordDecision(f.typedName.Name, outcomeEmpty)
		return nil
	}
}

func find(models attrmodels.ModelDataCollection, target string) (attrmodels.ModelData, bool) {
	for _, model := range models {
		if model.ID == target {
			return model, true
		}
	}
	return attrmodels.ModelData{}, false
}
