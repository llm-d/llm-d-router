package models

import (
	"context"
	"encoding/json"
	"fmt"
	"io"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	attrmodels "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/models"
)

// ModelExtractor is not a requesthandling.DataProducer as data here is produced
// asynchronously and not tied to incoming requests
var _ fwkplugin.ProducerPlugin = &ModelExtractor{}

var _ fwkdl.PollingExtractor[*ModelResponse] = &ModelExtractor{}

// ModelResponse is the response from /v1/models API.
type ModelResponse struct {
	Object string                 `json:"object"`
	Data   []attrmodels.ModelData `json:"data"`
}

// ParseModels decodes a /v1/models response body into a ModelResponse. It is the
// shared parser for both the polling source and the once-per-endpoint extractor.
func ParseModels(data io.Reader) (*ModelResponse, error) {
	body, err := io.ReadAll(data)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %v", err)
	}
	var modelsResponse ModelResponse
	if err := json.Unmarshal(body, &modelsResponse); err != nil {
		return nil, err
	}
	return &modelsResponse, nil
}

// ModelExtractor implements the models extraction.
type ModelExtractor struct {
	typedName fwkplugin.TypedName
	dk        fwkplugin.DataKey
}

// NewModelExtractor returns a new model extractor.
func NewModelExtractor() *ModelExtractor {
	return &ModelExtractor{
		typedName: fwkplugin.TypedName{
			Type: attrmodels.ModelsExtractorType,
			Name: attrmodels.ModelsExtractorType,
		},
		dk: attrmodels.ModelsAttributeKey,
	}
}

// TypedName returns the type and name of the ModelExtractor.
func (me *ModelExtractor) TypedName() fwkplugin.TypedName {
	return me.typedName
}

// ModelServerExtractorFactory is a factory function used to instantiate data layer's
// models extractor plugins specified in a configuration.
func ModelServerExtractorFactory(name string, _ *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	extractor := NewModelExtractor()
	extractor.typedName.Name = name
	return extractor, nil
}

// Extract stores the model list as an endpoint attribute.
func (me *ModelExtractor) Extract(_ context.Context, in fwkdl.PollInput[*ModelResponse]) error {
	in.Endpoint.GetAttributes().Put(me.dk.String(), attrmodels.ModelDataCollection(in.Payload.Data))
	return nil
}

// Produces returns data produced by the producer.
func (me *ModelExtractor) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{me.dk: attrmodels.ModelDataCollection{}}
}
