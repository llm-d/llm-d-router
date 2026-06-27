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
	"fmt"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	attrmodels "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/models"
	srchttp "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/http"
	srcnotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
)

// ModelsEndpointExtractorType identifies the extractor that fetches /v1/models
// once per endpoint, as opposed to the polling models-data-extractor.
const ModelsEndpointExtractorType = "models-endpoint-extractor"

// fetchTimeout bounds a single /v1/models fetch so a slow or unresponsive model
// server cannot keep the background fetch goroutine alive indefinitely.
const fetchTimeout = 10 * time.Second

// Model-server connection defaults, matching the polling models source so both
// reach the same endpoint with the same TLS handling.
const (
	defaultModelsScheme             = "http"
	defaultModelsPath               = "/v1/models"
	defaultModelsInsecureSkipVerify = true
)

// Assert the extractor produces an attribute, runs on endpoint lifecycle events,
// and self-wires its source dependency.
var (
	_ fwkplugin.ProducerPlugin = (*ModelEndpointExtractor)(nil)
	_ fwkdl.EndpointExtractor  = (*ModelEndpointExtractor)(nil)
	_ fwkdl.Registrant         = (*ModelEndpointExtractor)(nil)
)

// modelsEndpointExtractorParams configures the model-server endpoint and TLS
// handling. It mirrors the polling models source so operators can point the
// extractor at an https model server.
type modelsEndpointExtractorParams struct {
	Scheme             string `json:"scheme"`
	Path               string `json:"path"`
	InsecureSkipVerify bool   `json:"insecureSkipVerify"`
}

// ModelEndpointExtractor fetches /v1/models once when an endpoint is added and
// stores the parsed model list (including max_model_len) as an endpoint
// attribute. The model list is fixed for an endpoint's lifetime, so it does not
// re-fetch on a timer the way the polling extractor does.
type ModelEndpointExtractor struct {
	typedName fwkplugin.TypedName
	dk        fwkplugin.DataKey
	// fetcher performs the one-shot GET and JSON parse, reusing the HTTP source's
	// scheme/TLS handling. Only its Poll method is used; the polling Dispatch loop
	// is never driven.
	fetcher *srchttp.HTTPDataSource[*ModelResponse]
}

// NewModelEndpointExtractor builds an extractor that fetches from
// scheme://<endpoint>/path, verifying the server certificate unless insecure is set.
func NewModelEndpointExtractor(name, scheme, path string, insecure bool) (*ModelEndpointExtractor, error) {
	fetcher, err := srchttp.NewHTTPDataSource(scheme, path, insecure, ModelsEndpointExtractorType, name, ParseModels)
	if err != nil {
		return nil, fmt.Errorf("failed to create models fetcher: %w", err)
	}
	return &ModelEndpointExtractor{
		typedName: fwkplugin.TypedName{Type: ModelsEndpointExtractorType, Name: name},
		dk:        attrmodels.ModelsAttributeKey,
		fetcher:   fetcher,
	}, nil
}

// TypedName returns the plugin type and name.
func (me *ModelEndpointExtractor) TypedName() fwkplugin.TypedName { return me.typedName }

// Produces declares the /v1/models attribute this extractor populates so the
// framework can wire it to consumers of that attribute.
func (me *ModelEndpointExtractor) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{me.dk: attrmodels.ModelDataCollection{}}
}

// Extract records the model list for an added endpoint. The fetch runs in the
// background and returns immediately: endpoint add/update is processed serially,
// so a slow or unreachable model server must not stall endpoint provisioning.
// Until the fetch completes the attribute stays unset and the consumer falls
// back to its default. Delete events need no work; the attribute leaves with the
// endpoint.
func (me *ModelEndpointExtractor) Extract(ctx context.Context, event fwkdl.EndpointEvent) error {
	// Only an add/update carries model data to extract; deletes carry none.
	if event.Type != fwkdl.EventAddOrUpdate {
		return nil
	}
	ep := event.Endpoint
	if ep == nil || ep.GetMetadata() == nil {
		return nil
	}
	// The model list is fixed for an endpoint's lifetime, so fetch only once even
	// though add/update may fire repeatedly for the same endpoint.
	if _, ok := ep.GetAttributes().Get(me.dk.String()); ok {
		return nil
	}
	go me.fetchAndStore(ctx, ep)
	return nil
}

// fetchAndStore fetches /v1/models once, under a bounded timeout, and records the
// model list (including max_model_len) as an endpoint attribute. A failed fetch
// is logged and dropped, leaving the attribute unset so the consumer falls back
// to its default.
func (me *ModelEndpointExtractor) fetchAndStore(ctx context.Context, ep fwkdl.Endpoint) {
	ctx, cancel := context.WithTimeout(ctx, fetchTimeout)
	defer cancel()
	resp, err := me.fetcher.Poll(ctx, ep)
	if err != nil {
		log.FromContext(ctx).V(logging.DEBUG).Info("failed to fetch /v1/models",
			"endpoint", ep.GetMetadata().NamespacedName, "err", err)
		return
	}
	ep.GetAttributes().Put(me.dk.String(), attrmodels.ModelDataCollection(resp.Data))
}

// RegisterDependencies binds this extractor to an endpoint-notification-source,
// auto-creating that source when the configuration does not already define one.
// This is what makes the extractor wire itself in as a default source.
func (me *ModelEndpointExtractor) RegisterDependencies(r fwkdl.Registrar) error {
	return r.Register(fwkdl.PendingRegistration{
		Owner:      me.TypedName(),
		SourceType: srcnotifications.EndpointNotificationSourceType,
		Extractor:  me,
		DefaultSource: srcnotifications.NewEndpointDataSource(
			srcnotifications.EndpointNotificationSourceType, srcnotifications.EndpointNotificationSourceType),
	})
}

// ModelEndpointExtractorFactory instantiates the extractor from optional JSON
// parameters, defaulting to an http model server with TLS verification skipped.
func ModelEndpointExtractorFactory(name string, parameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params := modelsEndpointExtractorParams{
		Scheme:             defaultModelsScheme,
		Path:               defaultModelsPath,
		InsecureSkipVerify: defaultModelsInsecureSkipVerify,
	}
	// Overlay the defaults with any configured values.
	if parameters != nil {
		if err := parameters.Decode(&params); err != nil {
			return nil, err
		}
	}
	return NewModelEndpointExtractor(name, params.Scheme, params.Path, params.InsecureSkipVerify)
}
