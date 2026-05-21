package models

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

const (
	ModelsAttributeKey  = "/v1/models"
	ModelsExtractorType = "models-data-extractor"
)

// ModelDataCollection defines models' data returned from /v1/models API
type ModelDataCollection []ModelData

// ModelData defines model's data returned from /v1/models API
type ModelData struct {
	ID     string `json:"id"`
	Parent string `json:"parent,omitempty"`
}

// String returns a string representation of the model info
func (m *ModelData) String() string {
	return fmt.Sprintf("%+v", *m)
}

// Clone returns a full copy of the object
func (m ModelDataCollection) Clone() fwkdl.Cloneable {
	if m == nil {
		return nil
	}
	clone := make([]ModelData, len(m))
	copy(clone, m)
	return (*ModelDataCollection)(&clone)
}

func (m ModelDataCollection) String() string {
	if m == nil {
		return "[]"
	}
	parts := make([]string, len(m))
	for i, p := range m {
		parts[i] = p.String()
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

// ModelResponse is the response from /v1/models API
type ModelResponse struct {
	Object string      `json:"object"`
	Data   []ModelData `json:"data"`
}

// ModelsResponseType is the type of models response
var (
	ModelsResponseType = reflect.TypeOf(ModelResponse{})
)

// ModelExtractor implements the models extraction.
type ModelExtractor struct {
	typedName fwkplugin.TypedName
}

// NewModelExtractor returns a new model extractor.
func NewModelExtractor() *ModelExtractor {
	return &ModelExtractor{
		typedName: fwkplugin.TypedName{
			Type: ModelsExtractorType,
			Name: ModelsExtractorType,
		},
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

	ep.GetAttributes().Put(ModelsAttributeKey, ModelDataCollection(models.Data))
	return nil
}
