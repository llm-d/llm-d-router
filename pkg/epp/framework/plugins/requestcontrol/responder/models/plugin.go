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

package models

import (
	"context"
	"encoding/json"
	"net/http"
	"slices"
	"strings"

	"sigs.k8s.io/controller-runtime/pkg/log"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmodels "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/models"
	extmodels "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/models"
	srcmodels "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/models"
)

const ModelsResponderType = "models-responder"

// openAIModel is a public model item returned by the /v1/models API.
type openAIModel struct {
	ID           string `json:"id"`
	Object       string `json:"object,omitempty"`
	Created      int64  `json:"created,omitempty"`
	OwnedBy      string `json:"owned_by,omitempty"`
	ShutdownDate string `json:"shutdown_date,omitempty"`
}

// openAIModelListResponse is the public response returned by /v1/models.
type openAIModelListResponse struct {
	Object string        `json:"object"`
	Data   []openAIModel `json:"data"`
}

var (
	_ fwkrc.Responder          = &Responder{}
	_ fwkrc.Screener           = &Responder{}
	_ fwkplugin.ConsumerPlugin = &Responder{}
	_ fwkdl.Registrant         = &Responder{}
)

// Responder answers GET /v1/models from model data stored in EPP. It does not
// route the request to a model-server endpoint.
type Responder struct {
	typedName fwkplugin.TypedName
}

func Factory(name string, _ *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	return New().WithName(name), nil
}

// New returns a Responder for the model discovery endpoint.
func New() *Responder {
	return &Responder{
		typedName: fwkplugin.TypedName{Type: ModelsResponderType, Name: ModelsResponderType},
	}
}

func (p *Responder) TypedName() fwkplugin.TypedName {
	return p.typedName
}

func (p *Responder) WithName(name string) *Responder {
	p.typedName.Name = name
	return p
}

// Consumes declares the per-endpoint model list. It is optional rather than required
// because RegisterDependencies supplies the producer itself, and required keys are
// resolved before plugins get to register their dependencies.
func (p *Responder) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Optional: map[fwkplugin.DataKey]any{
			attrmodels.ModelsAttributeKey: attrmodels.ModelDataCollection{},
		},
	}
}

// RegisterDependencies creates the default models-data-source and models-data-extractor
// when models-responder is configured, so users do not need to configure them separately.
// An explicitly configured models-data-source takes precedence over the default.
func (p *Responder) RegisterDependencies(r fwkdl.Registrar) error {
	source, err := srcmodels.NewDefaultHTTPModelsDataSource(srcmodels.ModelsDataSourceType)
	if err != nil {
		return err
	}
	return r.Register(fwkdl.PendingRegistration{
		Owner:         p.typedName,
		SourceType:    srcmodels.ModelsDataSourceType,
		Extractor:     extmodels.NewModelExtractor(),
		DefaultSource: source,
	})
}

// Screen keeps the endpoints whose collected /v1/models list contains the target model.
// For a base model it also keeps endpoints that have not reported a list yet. For a LoRA
// adapter it does not, because an unreported endpoint is unlikely to have the adapter.
// When no endpoint lists the target, it keeps the unreported endpoints, or all endpoints
// if every endpoint has reported.
func (p *Responder) Screen(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	if request == nil || request.TargetModel == "" {
		return endpoints
	}

	serving := make([]fwksched.Endpoint, 0, len(endpoints))
	unlisted := make([]fwksched.Endpoint, 0, len(endpoints))
	adapterListed := false
	for _, endpoint := range endpoints {
		models, ok := fwkdl.ReadAttribute[attrmodels.ModelDataCollection](endpoint, attrmodels.ModelsAttributeKey)
		if !ok {
			unlisted = append(unlisted, endpoint)
			continue
		}
		for _, model := range models {
			if model.ID != request.TargetModel {
				continue
			}
			serving = append(serving, endpoint)
			adapterListed = adapterListed || model.Parent != ""
			break
		}
	}

	logger := log.FromContext(ctx).V(logutil.DEBUG).WithValues(
		"plugin", p.typedName, "model", request.TargetModel)
	switch {
	case len(serving) > 0 && adapterListed:
		if len(unlisted) > 0 {
			logger.Info("Returning LoRA adapter request candidates only from endpoints that list the adapter. "+
				"Endpoints whose model list is not collected yet are skipped.",
				"endpointsWithAdapter", len(serving),
				"endpointsSkipped", len(unlisted))
		}
		return serving
	case len(serving) > 0:
		return append(serving, unlisted...)
	case len(unlisted) > 0:
		logger.Info("No endpoint lists the requested model. "+
			"Returning only endpoints whose model list is not collected yet.",
			"endpointsUsed", len(unlisted),
			"endpointsTotal", len(endpoints))
		return unlisted
	default:
		logger.Info("No endpoint lists the requested model. "+
			"Returning all endpoints because the model lists may be incomplete.",
			"endpointsTotal", len(endpoints))
		return endpoints
	}
}

// Respond answers GET /v1/models and declines everything else.
func (p *Responder) Respond(ctx context.Context, request *fwkrc.RequestLine, endpoints []fwkdl.Endpoint) (*fwkrc.LocalResponse, error) {
	if request == nil || request.Method != http.MethodGet {
		return nil, nil //nolint:nilnil
	}
	if strings.TrimSuffix(request.Path, "/") != attrmodels.OpenAIModelsPath {
		return nil, nil //nolint:nilnil
	}

	body, collected, err := aggregate(endpoints)
	if err != nil {
		log.FromContext(ctx).Error(err, "failed to encode /v1/models response")
		return nil, errcommon.Error{
			Code: errcommon.Internal,
			Msg:  "failed to encode model response",
		}
	}
	if collected == 0 {
		// No endpoint has reported its model list: either the pool is empty, or the endpoints
		// are still being scraped after a cold start or scale-up. Fail so the client retries
		// instead of caching a misleading empty list.
		return nil, errcommon.Error{
			Code: errcommon.ServiceUnavailable,
			Msg:  "no model data collected yet from any endpoint",
		}
	}

	return &fwkrc.LocalResponse{
		StatusCode: http.StatusOK,
		Headers:    map[string]string{fwkrequest.HeaderContentType: "application/json"},
		Body:       body,
	}, nil
}

// aggregate collects the models from all endpoints and returns one entry per unique model
// ID, sorted alphabetically so the response is the same every time. The second return value
// is how many endpoints have reported a model list, letting the caller distinguish "nothing
// collected yet" from a genuinely empty result.
func aggregate(endpoints []fwkdl.Endpoint) (json.RawMessage, int, error) {
	// Sort endpoints first so the same endpoint's copy of a duplicated model ID always wins.
	ordered := slices.Clone(endpoints)
	slices.SortFunc(ordered, func(a, b fwkdl.Endpoint) int {
		return strings.Compare(a.GetMetadata().ID.String(), b.GetMetadata().ID.String())
	})

	seen := make(map[string]struct{})
	data := make([]openAIModel, 0)
	collected := 0 // endpoints that have reported their model list at least once

	for _, ep := range ordered {
		c, ok := fwkdl.ReadAttribute[attrmodels.ModelDataCollection](ep.GetAttributes(), attrmodels.ModelsAttributeKey)
		if !ok {
			continue // registered but not scraped yet; do not count it
		}
		collected++
		for _, model := range c {
			if model.ID == "" {
				continue
			}
			if _, dup := seen[model.ID]; dup {
				continue
			}
			seen[model.ID] = struct{}{}
			data = append(data, openAIModel{
				ID:           model.ID,
				Object:       model.Object,
				Created:      model.Created,
				OwnedBy:      model.OwnedBy,
				ShutdownDate: model.ShutdownDate,
			})
		}
	}
	if collected == 0 {
		return nil, 0, nil
	}
	slices.SortFunc(data, func(a, b openAIModel) int { return strings.Compare(a.ID, b.ID) })

	body, err := json.Marshal(openAIModelListResponse{Object: "list", Data: data})
	if err != nil {
		return nil, 0, err
	}
	return body, collected, nil
}
