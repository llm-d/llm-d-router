/*
Copyright 2026 The Kubernetes Authors.
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

package metrics

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sync"

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
	// Port, when set, overrides the endpoint's inference port for metrics retrieval.
	// Use when the model server exposes metrics on a port other than the InferencePool target port.
	// The override applies to every endpoint of the source. Do not set it on pools with
	// multiple target ports (data-parallel ranks): each rank must be scraped on its own
	// port, and a single override would point all ranks at the same one.
	Port *int `json:"port,omitempty"`
	// InsecureSkipVerify defines whether model server certificate should be verified or not.
	InsecureSkipVerify bool `json:"insecureSkipVerify"`
	// CACertPath is an optional PEM CA bundle to verify the scrape target cert.
	CACertPath string `json:"caCertPath"`
	// ClientCertPath and ClientKeyPath present a client certificate for mTLS, so the scrape
	// target authenticates this scraper by certificate instead of a bearer token. Both set together.
	ClientCertPath string `json:"clientCertPath"`
	ClientKeyPath  string `json:"clientKeyPath"`
	// Interval is the scrape period (e.g. "1s"). Rounded to the nearest multiple
	// of --refresh-metrics-interval. Empty or omitted means every base tick.
	Interval string `json:"interval"`
	// Families, when set, keeps only these metric families from each scrape and drops every
	// other line before parsing. Model servers expose far more families than the extractors
	// read, so parsing only the listed ones cuts the scrape's CPU and allocations. The list must
	// cover every family the source's extractors read; the rest are invisible to them.
	Families []string `json:"families,omitempty"`
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

	if cfg.Port != nil && (*cfg.Port < 1 || *cfg.Port > 65535) {
		return nil, fmt.Errorf("invalid port %d: must be between 1 and 65535", *cfg.Port)
	}

	intervalOpt, err := http.ParseIntervalOption(cfg.Interval)
	if err != nil {
		return nil, err
	}

	opts := []http.Option{intervalOpt}
	if cfg.Port != nil {
		opts = append(opts, http.WithPortOverride(*cfg.Port))
	}

	parser := parseMetrics
	if len(cfg.Families) > 0 {
		parser = newFamilyFilter(cfg.Families).parse
	}

	return http.NewHTTPDataSource(cfg.Scheme, cfg.Path,
		http.TLSOptions{
			SkipVerify:     cfg.InsecureSkipVerify,
			CACertPath:     cfg.CACertPath,
			ClientCertPath: cfg.ClientCertPath,
			ClientKeyPath:  cfg.ClientKeyPath,
		},
		MetricsDataSourceType, name, parser, opts...)
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

// sampleSuffixes are the sample-name suffixes a family's samples may carry in the text format.
var sampleSuffixes = [][]byte{[]byte("_bucket"), []byte("_sum"), []byte("_count"), []byte("_total"), []byte("_created")}

// familyFilter parses only the lines of the listed metric families.
type familyFilter struct {
	families map[string]struct{}
	readers  sync.Pool
}

func newFamilyFilter(families []string) *familyFilter {
	f := &familyFilter{families: make(map[string]struct{}, len(families))}
	for _, name := range families {
		f.families[name] = struct{}{}
	}
	f.readers.New = func() any { return bufio.NewReaderSize(nil, 64*1024) }
	return f
}

func (f *familyFilter) parse(data io.Reader) (PrometheusMetricMap, error) {
	r := f.readers.Get().(*bufio.Reader)
	r.Reset(data)
	defer func() {
		r.Reset(nil)
		f.readers.Put(r)
	}()

	var kept bytes.Buffer
	var long []byte // a line longer than the reader's buffer, assembled across reads
	var excluded map[string]struct{}
	for {
		line, err := r.ReadSlice('\n')
		if errors.Is(err, bufio.ErrBufferFull) {
			long = append(long, line...)
			continue
		}
		if long != nil {
			line = append(long, line...)
			long = nil
		}
		if f.keep(line, &excluded) {
			kept.Write(line)
			if len(line) > 0 && line[len(line)-1] != '\n' {
				kept.WriteByte('\n')
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
	}
	return parseMetrics(&kept)
}

// keep reports whether a text-format line belongs to a listed family. A declared, unlisted
// family takes precedence over suffix matching for its samples.
func (f *familyFilter) keep(line []byte, excluded *map[string]struct{}) bool {
	line = bytes.TrimLeft(line, " \t")
	metadata := bytes.HasPrefix(line, []byte("#"))
	if metadata {
		comment := bytes.TrimLeft(line[1:], " \t")
		end := bytes.IndexAny(comment, " \t")
		if end < 0 || (!bytes.Equal(comment[:end], []byte("HELP")) && !bytes.Equal(comment[:end], []byte("TYPE"))) {
			return false
		}
		line = bytes.TrimLeft(comment[end:], " \t")
	}
	end := bytes.IndexAny(line, "{ \t\r\n")
	if end < 0 {
		end = len(line)
	}
	name := line[:end]
	if len(name) == 0 {
		return false
	}
	if _, ok := f.families[string(name)]; ok {
		return true
	}
	if metadata {
		if f.matchesSuffix(name) {
			if *excluded == nil {
				*excluded = make(map[string]struct{})
			}
			(*excluded)[string(name)] = struct{}{}
		}
		return false
	}
	if _, ok := (*excluded)[string(name)]; ok {
		return false
	}
	return f.matchesSuffix(name)
}

func (f *familyFilter) matchesSuffix(name []byte) bool {
	for _, suffix := range sampleSuffixes {
		if bytes.HasSuffix(name, suffix) {
			if _, ok := f.families[string(name[:len(name)-len(suffix)])]; ok {
				return true
			}
		}
	}
	return false
}
