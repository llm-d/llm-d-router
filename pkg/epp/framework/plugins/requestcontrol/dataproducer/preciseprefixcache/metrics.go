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

package preciseprefixcache

import (
	approximateprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/approximateprefix"
)

func recordPrefixCacheMatch(pluginName, pluginType string, matchedLength, totalLength float64) {
	if totalLength > 0 {
		hitRatio := matchedLength / totalLength
		approximateprefix.LlmdPrefixCacheHitRatio.WithLabelValues(pluginName, pluginType).Observe(hitRatio)
	}
}

func recordPrefixCacheHitRatioStats(pluginName, pluginType string, maxHitRatio float64, avgHitRatio float64, stdDevHitRatio float64) {
	approximateprefix.LlmdPrefixCacheMaxHitRatio.WithLabelValues(pluginName, pluginType).Observe(maxHitRatio)
	approximateprefix.LlmdPrefixCacheAvgHitRatio.WithLabelValues(pluginName, pluginType).Observe(avgHitRatio)
	approximateprefix.LlmdPrefixCacheStdDevHitRatio.WithLabelValues(pluginName, pluginType).Observe(stdDevHitRatio)
}
