/*
Copyright 2026 The Kubernetes Authors.

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

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	attrmodels "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/models"
	extmodels "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/models"
	srcmodels "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/models"
)

const ModelsResponderType = "models-responder"

// openAIModelsPath is the route this plugin answers.
const openAIModelsPath = "/v1/models"

// defaultScrapePath is where the auto-created source collects each pod's model list.
const defaultScrapePath = "/v1/models"

// defaultScheme is the scheme used for the auto-created models source.
const defaultScheme = "http"

var (
	_ fwkrc.Responder          = &Responder{}
	_ fwkplugin.ConsumerPlugin = &Responder{}
	_ fwkdl.Registrant         = &Responder{}
)

// Responder answers GET /v1/models from the model lists the data layer collects per
// endpoint. A single model server only knows the models loaded on itself, so routing the
// request to one of them omits adapters loaded elsewhere in the pool.
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

// RegisterDependencies auto-creates a models-data-source and models-data-extractor so
// declaring this plugin is sufficient to serve /v1/models without config the other two.
// If the user has already declared a models-data-source in EPPconfig (e.g. with custom TLS or scrape path), it takes
// precedence over the auto-created one.
func (p *Responder) RegisterDependencies(r fwkdl.Registrar) error {
	source, err := srcmodels.NewHTTPModelsDataSource(defaultScheme, defaultScrapePath, srcmodels.ModelsDataSourceType)
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

// Respond answers GET /v1/models and declines everything else.
func (p *Responder) Respond(_ context.Context, request *fwkrc.RequestLine, endpoints []fwkdl.Endpoint) (*fwkrc.LocalResponse, error) {
	if request == nil || request.Method != http.MethodGet {
		return nil, nil //nolint:nilnil
	}
	if strings.TrimSuffix(request.Path, "/") != openAIModelsPath {
		return nil, nil //nolint:nilnil
	}

	body, collected := aggregate(endpoints)
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
		Headers:    map[string]string{"content-type": "application/json"},
		Body:       body,
	}, nil
}

// aggregate collects the models from all endpoints and returns one entry per unique model
// ID, sorted alphabetically so the response is the same every time. The second return value
// is how many endpoints have reported a model list, letting the caller distinguish "nothing
// collected yet" from a genuinely empty result.
func aggregate(endpoints []fwkdl.Endpoint) (json.RawMessage, int) {
	// Sort endpoints first so the same endpoint's copy of a duplicated model ID always wins.
	ordered := slices.Clone(endpoints)
	slices.SortFunc(ordered, func(a, b fwkdl.Endpoint) int {
		return strings.Compare(a.GetMetadata().ID.String(), b.GetMetadata().ID.String())
	})

	seen := make(map[string]struct{})
	data := make([]attrmodels.ModelData, 0)
	collected := 0 // endpoints that have reported their model list at least once

	for _, ep := range ordered {
		c, ok := fwkdl.ReadAttribute[attrmodels.ModelDataCollection](ep.GetAttributes(), attrmodels.ModelsAttributeKey)
		if !ok {
			continue // registered but not scraped yet; do not count it
		}
		collected++
		for _, model := range c {
			if _, dup := seen[model.ID]; dup {
				continue
			}
			seen[model.ID] = struct{}{}
			data = append(data, model)
		}
	}
	slices.SortFunc(data, func(a, b attrmodels.ModelData) int { return strings.Compare(a.ID, b.ID) })

	body, err := json.Marshal(attrmodels.ModelResponse{Object: "list", Data: data})
	if err != nil {
		return nil, 0
	}
	return body, collected
}
