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
	"fmt"
	"strings"

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

func ReadScalarMetricValue(attrs fwkdl.AttributeMap, key plugin.DataKey) (ScalarMetricValue, bool) {
	return fwkdl.ReadAttribute[ScalarMetricValue](attrs, key)
}

// ResolveConfiguredKey builds the key a plugin reads from an attribute name and
// an optional producer, as the two are spelled in configuration. A nil producer
// selects the core metrics extractor; a non-nil one is honoured verbatim, so the
// empty string addresses a producer-agnostic attribute.
//
// An attribute containing "/" with no producer is rejected rather than guessed
// at. The attribute and producer were once a single "Attribute/Producer" string,
// and a key built from that spelling matches nothing: the read resolves as
// absent, so a filter passes every endpoint and a scorer returns zero, with no
// error anywhere. A name that genuinely contains "/" is still reachable by
// setting the producer explicitly.
func ResolveConfiguredKey(pluginDesc, attribute string, producer *string) (plugin.DataKey, error) {
	if producer != nil {
		return plugin.NewDataKey(attribute, *producer), nil
	}
	if idx := strings.LastIndex(attribute, "/"); idx != -1 {
		return plugin.DataKey{}, fmt.Errorf(
			"%s: attribute %q contains %q but no producer is set. The attribute and its producer are "+
				"separate parameters; if this is the combined \"Attribute/Producer\" spelling, split it into "+
				"attribute: %q and producer: %q. If the attribute name itself contains %q, set producer explicitly",
			pluginDesc, attribute, "/", attribute[:idx], attribute[idx+1:], "/")
	}
	return ScalarMetricDataKey(attribute), nil
}
