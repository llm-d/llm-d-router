/*
Copyright 2025 The Kubernetes Authors.
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
	"errors"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"

	dto "github.com/prometheus/client_model/go"
	"github.com/prometheus/common/model"

	sourcemetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/metrics"
)

// metricSpecRE handles structural parsing: extracts the metric name token and optional label block.
// Field validation (metric name syntax) is delegated to model.LegacyValidation.IsValidMetricName.
var metricSpecRE = regexp.MustCompile(`^\s*(\S+?)\s*(?:\{([^}]*)\})?\s*$`)

// labelPairRE handles structural parsing: extracts the key, operator, and
// value from a key="value" or key!="value" label matcher (after
// addQuotesToLabelValues normalisation). Field validation (label name syntax)
// is delegated to model.LegacyValidation.IsValidLabelName.
var labelPairRE = regexp.MustCompile(`^\s*(\S+?)\s*(!=|=)\s*"([^"]*)"\s*$`)

// addQuotesRE is the compiled form of the pattern used by addQuotesToLabelValues.
var addQuotesRE = regexp.MustCompile(`(\w+)\s*(!=|=)\s*([^",}\s]+)`)

// Spec represents a single metric's specification.
type Spec struct {
	Name      string            // the metric's name
	Labels    map[string]string // maps metric's label name to required value
	NotLabels map[string]string // maps metric's label name to excluded value
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
	m := metricSpecRE.FindStringSubmatch(quoted)
	if m == nil || !model.LegacyValidation.IsValidMetricName(m[1]) {
		return nil, fmt.Errorf("not a valid metric specification: %q", spec)
	}

	metricLabels := make(map[string]string)
	var metricNotLabels map[string]string
	if labelBlock := strings.TrimSpace(m[2]); labelBlock != "" {
		for _, pair := range strings.Split(labelBlock, ",") {
			lm := labelPairRE.FindStringSubmatch(pair)
			if lm == nil || !model.LegacyValidation.IsValidLabelName(lm[1]) {
				return nil, fmt.Errorf("invalid label pair %q in specification: %q", pair, spec)
			}
			if lm[2] == "!=" {
				if metricNotLabels == nil {
					metricNotLabels = make(map[string]string)
				}
				metricNotLabels[lm[1]] = lm[3]
			} else {
				metricLabels[lm[1]] = lm[3]
			}
		}
	}

	return &Spec{Name: m[1], Labels: metricLabels, NotLabels: metricNotLabels}, nil
}

// addQuotesToLabelValues wraps label values with quotes, if missing,
// allowing both {label=value} and {label="value"} inputs.
func addQuotesToLabelValues(input string) string {
	return addQuotesRE.ReplaceAllString(input, `$1$2"$3"`)
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
	switch family.GetType() {
	case dto.MetricType_GAUGE, dto.MetricType_COUNTER, dto.MetricType_UNTYPED:
	default:
		return nil, fmt.Errorf("unsupported metric type %v for %q", family.GetType(), spec.Name)
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
		return nil, fmt.Errorf(
			"no matching metric found for %q with labels %v and excluded label values %v",
			spec.Name,
			spec.Labels,
			spec.NotLabels,
		)
	}

	if v := extractValue(latest); math.IsNaN(v) || math.IsInf(v, 0) {
		return nil, fmt.Errorf("non-finite metric value %v for %q", v, spec.Name)
	}

	return latest, nil
}

// aggregation selects how the series of one metric family fold into a single value.
// The README section "Families with several series" gives the choice per metric.
type aggregation int

const (
	aggregateSum aggregation = iota
	aggregateMax
)

// aggregateMetric folds every series matching Spec into one value, so a pod
// exposing one series per engine reports all of its engines. A non-finite
// value on any matching series fails the whole family.
func (spec *Spec) aggregateMetric(families sourcemetrics.PrometheusMetricMap, agg aggregation) (float64, error) {
	family, err := extractFamily(spec, families)
	if err != nil {
		return 0, err
	}

	var result float64
	matched := 0

	for _, metric := range family.GetMetric() {
		if !spec.labelsMatch(metric.GetLabel()) {
			continue
		}
		value := extractValue(metric)
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return 0, fmt.Errorf("non-finite metric value %v for %q", value, spec.Name)
		}
		switch {
		case matched == 0:
			result = value
		case agg == aggregateSum:
			result += value
		case value > result:
			result = value
		}
		matched++
	}

	if matched == 0 {
		return 0, fmt.Errorf(
			"no matching metric found for %q with labels %v and excluded label values %v",
			spec.Name,
			spec.Labels,
			spec.NotLabels,
		)
	}
	return result, nil
}

// maxValue returns the largest positive integer value across the series
// matching Spec, read from the gauge or counter value or, when labelName is
// set, from that label. A pod running several engines reports one series per
// engine, and each series is one engine's value. Series whose value is
// missing, non-numeric (vLLM writes "None" for a capacity it cannot compute),
// or not positive are skipped. ok is false when no series qualifies, and when
// any matching series is NaN, infinite, or beyond the int range: the largest
// of the remaining series can be lower than the true maximum.
func (spec *Spec) maxValue(families sourcemetrics.PrometheusMetricMap, labelName string) (value int, ok bool) {
	family, err := extractFamily(spec, families)
	if err != nil {
		return 0, false
	}
	for _, metric := range family.GetMetric() {
		if !spec.labelsMatch(metric.GetLabel()) {
			continue
		}
		var v float64
		if labelName == "" {
			v = extractValue(metric)
		} else {
			raw, found := labelValue(metric, labelName)
			if !found {
				continue
			}
			parsed, err := strconv.ParseFloat(raw, 64)
			if err != nil {
				continue
			}
			v = parsed
		}
		// float64(math.MaxInt) rounds up to 2^63, which does not fit an int.
		if math.IsNaN(v) || math.IsInf(v, 0) || v >= float64(math.MaxInt) {
			return 0, false
		}
		if v <= 0 {
			continue
		}
		if int(v) > value {
			value, ok = int(v), true
		}
	}
	return value, ok
}

// labelValue returns the value of the named label on metric.
func labelValue(metric *dto.Metric, name string) (string, bool) {
	for _, label := range metric.GetLabel() {
		if label.GetName() == name {
			return label.GetValue(), true
		}
	}
	return "", false
}

// labelsMatch checks if metric labels match the specification labels.
// Scans the label pairs directly rather than building a map: this runs per
// series per scrape tick, and specs carry at most a couple of label matchers.
func (spec *Spec) labelsMatch(metricLabels []*dto.LabelPair) bool {
	for name, value := range spec.Labels {
		found := false
		for _, label := range metricLabels {
			if label.GetName() == name {
				found = label.GetValue() == value
				break
			}
		}
		if !found {
			return false
		}
	}
	for name, value := range spec.NotLabels {
		actual := ""
		for _, label := range metricLabels {
			if label.GetName() == name {
				actual = label.GetValue()
				break
			}
		}
		if actual == value {
			return false
		}
	}
	return true
}

// extractValue gets the numeric value from different metric types.
// Gauge, Counter and Untyped samples carry one; the text parser files a
// sample whose family has no TYPE line as Untyped.
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
	if untyped := metric.GetUntyped(); untyped != nil {
		return untyped.GetValue()
	}
	return 0
}
