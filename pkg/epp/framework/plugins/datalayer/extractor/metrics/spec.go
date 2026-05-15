/*
Copyright 2025 The Kubernetes Authors.

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
	"errors"
	"fmt"
	"regexp"
	"strings"

	dto "github.com/prometheus/client_model/go"

	sourcemetrics "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/datalayer/source/metrics"
)

// metricSpecRe matches: optional-space metric-name optional-space optional-{labels} optional-space end.
// Prometheus metric names may contain letters, digits, underscores, and colons.
var metricSpecRe = regexp.MustCompile(`^\s*([a-zA-Z_:][a-zA-Z0-9_:]*)\s*(?:\{([^}]*)\})?\s*$`)

// labelPairRe matches a single key="value" label pair (after addQuotesToLabelValues normalisation).
var labelPairRe = regexp.MustCompile(`^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"([^"]*)"\s*$`)

// addQuotesRe is the compiled form of the pattern used by addQuotesToLabelValues.
var addQuotesRe = regexp.MustCompile(`(\w+)\s*=\s*([^",}\s]+)`)

// Spec represents a single metric's specification.
type Spec struct {
	Name   string            // the metric's name
	Labels map[string]string // maps metric's label name to value
}

// parseStringToSpec converts a string to a metrics.Spec.
// Inputs are expected in PromQL Instant Vector Selector syntax:
// metric_name{label1=value1,label2=value2}, where labels are optional.
func parseStringToSpec(spec string) (*Spec, error) {
	if spec == "" {
		// allow empty string to represent the nil Spec
		return nil, nil //nolint:nilnil
	}

	// Normalise unquoted label values so {label=value} and {label="value"} both work.
	quoted := addQuotesToLabelValues(spec)
	m := metricSpecRe.FindStringSubmatch(quoted)
	if m == nil {
		return nil, fmt.Errorf("not a valid metric specification: %q", spec)
	}

	metricLabels := make(map[string]string)
	if labelBlock := strings.TrimSpace(m[2]); labelBlock != "" {
		for _, pair := range strings.Split(labelBlock, ",") {
			lm := labelPairRe.FindStringSubmatch(pair)
			if lm == nil {
				return nil, fmt.Errorf("invalid label pair %q in specification: %q", pair, spec)
			}
			metricLabels[lm[1]] = lm[2]
		}
	}

	return &Spec{Name: m[1], Labels: metricLabels}, nil
}

// addQuotesToLabelValues wraps label values with quotes, if missing,
// allowing both {label=value} and {label="value"} inputs.
func addQuotesToLabelValues(input string) string {
	return addQuotesRe.ReplaceAllString(input, `$1="$2"`)
}

// extract the metric family is common to standard and LoRA spec's.
func extractFamily(spec *Spec, families sourcemetrics.PrometheusMetricMap) (*dto.MetricFamily, error) {
	if spec == nil {
		return nil, errors.New("metric specification is nil")
	}

	family, exists := families[spec.Name]
	if !exists {
		return nil, fmt.Errorf("metric family %q not found", spec.Name)
	}

	if len(family.GetMetric()) == 0 {
		return nil, fmt.Errorf("no metrics found for %q", spec.Name)
	}
	return family, nil
}

// getLatestMetric retrieves the latest metric based on Spec.
func (spec *Spec) getLatestMetric(families sourcemetrics.PrometheusMetricMap) (*dto.Metric, error) {
	family, err := extractFamily(spec, families)
	if err != nil {
		return nil, err
	}

	var latest *dto.Metric
	var recent int64 = -1

	for _, metric := range family.GetMetric() {
		if spec.labelsMatch(metric.GetLabel()) {
			ts := metric.GetTimestampMs()
			if ts > recent {
				recent = ts
				latest = metric
			}
		}
	}

	if latest == nil {
		return nil, fmt.Errorf("no matching metric found for %q with labels %v", spec.Name, spec.Labels)
	}

	return latest, nil
}

// labelsMatch checks if metric labels match the specification labels.
func (spec *Spec) labelsMatch(metricLabels []*dto.LabelPair) bool {
	if len(spec.Labels) == 0 {
		return true // no label requirements
	}

	metricLabelMap := make(map[string]string)
	for _, label := range metricLabels {
		metricLabelMap[label.GetName()] = label.GetValue()
	}

	// check if all spec labels match
	for name, value := range spec.Labels {
		if metricValue, exists := metricLabelMap[name]; !exists || metricValue != value {
			return false
		}
	}

	return true
}

// extractValue gets the numeric value from different metric types.
// Currently only Gauge and Counter are supported.
func extractValue(metric *dto.Metric) float64 {
	if metric == nil {
		return 0
	}
	if gauge := metric.GetGauge(); gauge != nil {
		return gauge.GetValue()
	}
	if counter := metric.GetCounter(); counter != nil {
		return counter.GetValue()
	}
	return 0
}
