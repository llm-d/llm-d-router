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

package metrics

import (
	"encoding/json"
	"io"

	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/http"
)

const MetricsDataSourceType = "metrics-data-source"

// Default values for the metrics data source configuration.
const (
	defaultMetricsScheme             = "http"
	defaultMetricsPath               = "/metrics"
	defaultMetricsInsecureSkipVerify = true
)

// metricsDatasourceParams holds the configuration parameters for the metrics data source plugin.
// These values can be specified in the EndpointPickerConfig under the plugin's `parameters` field.
type metricsDatasourceParams struct {
	// Scheme defines the protocol scheme used in metrics retrieval (e.g., "http").
	Scheme string `json:"scheme"`
	// Path defines the URL path used in metrics retrieval (e.g., "/metrics").
	Path string `json:"path"`
	// InsecureSkipVerify defines whether model server certificate should be verified or not.
	InsecureSkipVerify bool `json:"insecureSkipVerify"`
	// CACertPath is an optional PEM CA bundle to verify the scrape target cert.
	CACertPath string `json:"caCertPath"`
	// ClientCertPath and ClientKeyPath present a client certificate for mTLS, so the scrape
	// target authenticates this scraper by certificate instead of a bearer token. Both set together.
	ClientCertPath string `json:"clientCertPath"`
	ClientKeyPath  string `json:"clientKeyPath"`
}

// NewHTTPMetricsDataSource constructs a MetricsDataSource with the given scheme and path.
// InsecureSkipVerify defaults to true (matching the factory default).
// Use this function directly in tests to bypass JSON parameter marshaling.
func NewHTTPMetricsDataSource(scheme, path, name string) (*http.HTTPDataSource[PrometheusMetricMap], error) {
	return http.NewHTTPDataSource(scheme, path, http.TLSOptions{SkipVerify: defaultMetricsInsecureSkipVerify},
		MetricsDataSourceType, name, parseMetrics)
}

// MetricsDataSourceFactory is a factory function used to instantiate data layer's
// metrics data source plugins specified in a configuration.
func MetricsDataSourceFactory(name string, parameters *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	cfg := defaultDataSourceConfigParams()

	if parameters != nil { // overlay the defaults with configured values
		if err := parameters.Decode(cfg); err != nil {
			return nil, err
		}
	}

	return http.NewHTTPDataSource(cfg.Scheme, cfg.Path,
		http.TLSOptions{
			SkipVerify:     cfg.InsecureSkipVerify,
			CACertPath:     cfg.CACertPath,
			ClientCertPath: cfg.ClientCertPath,
			ClientKeyPath:  cfg.ClientKeyPath,
		},
		MetricsDataSourceType, name, parseMetrics)
}

func defaultDataSourceConfigParams() *metricsDatasourceParams {
	return &metricsDatasourceParams{
		Scheme:             defaultMetricsScheme,
		Path:               defaultMetricsPath,
		InsecureSkipVerify: defaultMetricsInsecureSkipVerify,
	}
}

func parseMetrics(data io.Reader) (PrometheusMetricMap, error) {
	parser := expfmt.NewTextParser(model.LegacyValidation)
	return parser.TextToMetricFamilies(data)
}
