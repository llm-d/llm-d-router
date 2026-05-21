package models

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	extmodels "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/models"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/http"
)

const ModelsDataSourceType = "models-data-source"

// Default values for the models data source configuration.
const (
	defaultModelsScheme             = "http"
	defaultModelsPath               = "/v1/models"
	defaultModelsInsecureSkipVerify = true
)

// modelsDatasourceParams holds the configuration parameters for the models data source plugin.
// These values can be specified in the EndpointPickerConfig under the plugin's `parameters` field.
type modelsDatasourceParams struct {
	// Scheme defines the protocol scheme used in models retrieval (e.g., "http").
	Scheme string `json:"scheme"`
	// Path defines the URL path used in models retrieval (e.g., "/v1/models").
	Path string `json:"path"`
	// InsecureSkipVerify defines whether model server certificate should be verified or not.
	InsecureSkipVerify bool `json:"insecureSkipVerify"`
}

// NewHTTPModelsDataSource constructs a ModelsDataSource with the given scheme and path.
// InsecureSkipVerify defaults to true (matching the factory default).
// Use this function directly in tests to bypass JSON parameter marshaling.
func NewHTTPModelsDataSource(scheme, path, name string) (*http.HTTPDataSource, error) {
	return http.NewHTTPDataSource(scheme, path, defaultModelsInsecureSkipVerify,
		ModelsDataSourceType, name, parseModels, extmodels.ModelsResponseType)
}

// ModelDataSourceFactory is a factory function used to instantiate data layer's
// models data source plugins specified in a configuration.
func ModelDataSourceFactory(name string, parameters json.RawMessage, _ plugin.Handle) (plugin.Plugin, error) {
	cfg := defaultDataSourceConfigParams()
	if parameters != nil { // overlay the defaults with configured values
		if err := json.Unmarshal(parameters, cfg); err != nil {
			return nil, err
		}
	}
	if cfg.Scheme != "http" && cfg.Scheme != "https" {
		return nil, fmt.Errorf("unsupported scheme: %s", cfg.Scheme)
	}

	ds, err := http.NewHTTPDataSource(cfg.Scheme, cfg.Path, cfg.InsecureSkipVerify, ModelsDataSourceType,
		name, parseModels, extmodels.ModelsResponseType)
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP data source: %w", err)
	}
	return ds, nil
}

func defaultDataSourceConfigParams() *modelsDatasourceParams {
	return &modelsDatasourceParams{
		Scheme:             defaultModelsScheme,
		Path:               defaultModelsPath,
		InsecureSkipVerify: defaultModelsInsecureSkipVerify,
	}
}

func parseModels(data io.Reader) (any, error) {
	body, err := io.ReadAll(data)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %v", err)
	}
	var modelsResponse extmodels.ModelResponse
	err = json.Unmarshal(body, &modelsResponse)
	return &modelsResponse, err
}
