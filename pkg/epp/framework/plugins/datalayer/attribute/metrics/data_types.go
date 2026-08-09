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

package metrics

import (
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

const (
	// MetricsExtractorType is the plugin type for the core metrics extractor,
	// which publishes the custom scalar metric attributes.
	MetricsExtractorType = "core-metrics-extractor"
)

// ScalarMetricValue is a numeric endpoint attribute extracted from a configured scalar metric.
type ScalarMetricValue float64

func (v ScalarMetricValue) Clone() fwkdl.Cloneable {
	return v
}

// ScalarMetricDataKey returns the key under which the core metrics extractor
// publishes the custom scalar metric configured as attributeKey. The data type
// of a custom metric is chosen in configuration rather than in code, so the key
// is derived at construction time instead of being declared as a package var.
func ScalarMetricDataKey(attributeKey string) plugin.DataKey {
	return plugin.NewDataKey(attributeKey, MetricsExtractorType)
}

// ResolveScalarMetricAttribute resolves an attribute named in configuration to
// the key it is published under. A name that carries a producer -- the
// serialized form, e.g. "GPUUtilization/dcgm-extractor" -- addresses that
// producer's scalar attribute; a bare name addresses a custom scalar metric of
// the core metrics extractor, which is the only producer that takes its
// attribute names from configuration.
func ResolveScalarMetricAttribute(configured string) plugin.DataKey {
	return plugin.ParseDataKey(configured, MetricsExtractorType)
}

func ReadScalarMetricValue(attrs fwkdl.AttributeMap, key plugin.DataKey) (ScalarMetricValue, bool) {
	return fwkdl.ReadAttribute[ScalarMetricValue](attrs, key)
}
