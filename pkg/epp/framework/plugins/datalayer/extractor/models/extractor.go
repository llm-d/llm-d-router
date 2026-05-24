package models

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	attrmodels "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/models"
)

// ModelExtractor is not a requesthandling.DataProducer as data here is produced
// asynchronously and not tied to incoming requests
var _ fwkplugin.ProducerPlugin = &ModelExtractor{}

var _ fwkdl.Extractor = &ModelExtractor{}

// ModelResponse is the response from /v1/models API
type ModelResponse struct {
	Object string                 `json:"object"`
	Data   []attrmodels.ModelData `json:"data"`
}

// ModelsResponseType is the type of models response
var (
	ModelsResponseType = reflect.TypeOf(ModelResponse{})
)

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

// ExpectedInputType defines the type expected by ModelExtractor.
func (me *ModelExtractor) ExpectedInputType() reflect.Type {
	return ModelsResponseType
}

// ModelServerExtractorFactory is a factory function used to instantiate data layer's
// models extractor plugins specified in a configuration.
func ModelServerExtractorFactory(name string, _ json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	extractor := NewModelExtractor()
	extractor.typedName.Name = name
	return extractor, nil
}

// Extract transforms the data source output into a concrete attribute that
// is stored on the given endpoint.
func (me *ModelExtractor) Extract(_ context.Context, data any, ep fwkdl.Endpoint) error {
	models, ok := data.(*ModelResponse)
	if !ok {
		return fmt.Errorf("unexpected input in Extract: %T", data)
	}

	ep.GetAttributes().Put(me.dk.String(), attrmodels.ModelDataCollection(models.Data))
	return nil
}

// Produces returns data produced by the producer.
// This is a map from data key (string) produced to
// the data type of the key (represented as data with default value casted as any field).
func (me *ModelExtractor) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{me.dk: attrmodels.ModelDataCollection{}}
}
