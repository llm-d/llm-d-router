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

package dcgm

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/http"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/metrics"
)

const DCGMDataSourceType = "dcgm-data-source"

const (
	defaultDCGMScheme             = "http"
	defaultDCGMPath               = "/metrics"
	defaultDCGMPort               = 9400
	defaultDCGMInsecureSkipVerify = true
)

type dcgmDatasourceParams struct {
	Scheme             string `json:"scheme"`
	Path               string `json:"path"`
	Port               int    `json:"port"`
	InsecureSkipVerify bool   `json:"insecureSkipVerify"`
	// UseNodeAddress scrapes the node IP (DaemonSet) instead of the pod IP (sidecar).
	UseNodeAddress bool `json:"useNodeAddress"`
}

// NewHTTPDCGMDataSource constructs a DCGMDataSource with the given parameters.
// Use this function directly in tests to bypass JSON parameter marshaling.
func NewHTTPDCGMDataSource(scheme, path string, port int, name string, opts ...http.Option) (*http.HTTPDataSource[metrics.PrometheusMetricMap], error) {
	opts = append([]http.Option{http.WithPortOverride(port)}, opts...)
	return http.NewHTTPDataSource(scheme, path, http.TLSOptions{SkipVerify: defaultDCGMInsecureSkipVerify},
		DCGMDataSourceType, name, parsePrometheus, opts...)
}

// DCGMDataSourceFactory instantiates a dcgm-data-source plugin from configuration.
func DCGMDataSourceFactory(name string, parameters *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	cfg := defaultDCGMConfigParams()
	if parameters != nil {
		if err := parameters.Decode(cfg); err != nil {
			return nil, err
		}
	}
	if cfg.Scheme != "http" && cfg.Scheme != "https" {
		return nil, fmt.Errorf("unsupported scheme: %s", cfg.Scheme)
	}

	opts := []http.Option{http.WithPortOverride(cfg.Port)}
	if cfg.UseNodeAddress {
		opts = append(opts, http.WithUseNodeAddress())
	}

	ds, err := http.NewHTTPDataSource(cfg.Scheme, cfg.Path, http.TLSOptions{SkipVerify: cfg.InsecureSkipVerify},
		DCGMDataSourceType, name, parsePrometheus, opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create DCGM data source: %w", err)
	}
	return ds, nil
}

func defaultDCGMConfigParams() *dcgmDatasourceParams {
	return &dcgmDatasourceParams{
		Scheme:             defaultDCGMScheme,
		Path:               defaultDCGMPath,
		Port:               defaultDCGMPort,
		InsecureSkipVerify: defaultDCGMInsecureSkipVerify,
	}
}

func parsePrometheus(data io.Reader) (metrics.PrometheusMetricMap, error) {
	parser := expfmt.NewTextParser(model.LegacyValidation)
	return parser.TextToMetricFamilies(data)
}
