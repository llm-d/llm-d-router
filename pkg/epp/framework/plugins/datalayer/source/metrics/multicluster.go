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

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/http"
)

const (
	// MultiClusterMetricsDataSourceType scrapes like the pod metrics source. A distinct type
	// lets a config pair it with the multicluster extractor, separate from the pod pipeline.
	MultiClusterMetricsDataSourceType = "multicluster-metrics-data-source"

	// maxResponseBytes caps the metrics payload read from a peer cluster, so a hostile or
	// broken peer across the cross-cluster boundary cannot exhaust memory in the parser.
	maxResponseBytes = 8 << 20 // 8 MiB

	// defaultMultiClusterScheme is https, not the pod source's http: a cross-cluster scrape
	// crosses a trust boundary, so it is encrypted and verified by default.
	defaultMultiClusterScheme = "https"
)

// parseBoundedMetrics parses the metrics payload under maxResponseBytes.
func parseBoundedMetrics(r io.Reader) (PrometheusMetricMap, error) {
	return parseMetrics(io.LimitReader(r, maxResponseBytes))
}

// NewHTTPMultiClusterMetricsDataSource constructs the source with the given scheme and path.
// Use directly in tests to bypass JSON parameter marshaling.
func NewHTTPMultiClusterMetricsDataSource(scheme, path, name string) (*http.HTTPDataSource[PrometheusMetricMap], error) {
	return http.NewHTTPDataSource(scheme, path, http.TLSOptions{},
		MultiClusterMetricsDataSourceType, name, parseBoundedMetrics)
}

// MultiClusterMetricsDataSourceFactory instantiates the source. Unlike the pod source it
// verifies the peer's certificate by default, given the cross-cluster trust boundary. Set
// a caCertPath (or insecureSkipVerify) for a private cluster CA. Endpoint targeting is
// unchanged from the pod source.
func MultiClusterMetricsDataSourceFactory(name string, parameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	cfg := &metricsDatasourceParams{Scheme: defaultMultiClusterScheme, Path: defaultMetricsPath}

	if parameters != nil { // overlay the defaults with configured values
		if err := parameters.Decode(cfg); err != nil {
			return nil, err
		}
	}

	intervalOpt, err := http.ParseIntervalOption(cfg.Interval)
	if err != nil {
		return nil, err
	}

	return http.NewHTTPDataSource(cfg.Scheme, cfg.Path,
		http.TLSOptions{
			SkipVerify:     cfg.InsecureSkipVerify,
			CACertPath:     cfg.CACertPath,
			ClientCertPath: cfg.ClientCertPath,
			ClientKeyPath:  cfg.ClientKeyPath,
		},
		MultiClusterMetricsDataSourceType, name, parseBoundedMetrics, intervalOpt)
}
