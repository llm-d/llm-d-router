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

package approximateprefix

import (
	"math"
)

func CalculateHitRatioStats(hitRatios []float64) (maxRatio float64, avgRatio float64, stdDevRatio float64) {
	if len(hitRatios) == 0 {
		return 0, 0, 0
	}

	sum := 0.0
	maxRatio = 0.0
	for _, hitRatio := range hitRatios {
		sum += hitRatio
		if hitRatio > maxRatio {
			maxRatio = hitRatio
		}
	}
	avgRatio = sum / float64(len(hitRatios))

	varianceSum := 0.0
	for _, hitRatio := range hitRatios {
		diff := hitRatio - avgRatio
		varianceSum += diff * diff
	}
	stdDevRatio = 0.0
	if len(hitRatios) > 1 {
		stdDevRatio = varianceSum / float64(len(hitRatios)-1)
		stdDevRatio = math.Sqrt(stdDevRatio)
	}

	// Round to two decimal places for consistency in metrics reporting
	avgRatio = math.Round(avgRatio*100) / 100
	stdDevRatio = math.Round(stdDevRatio*100) / 100

	return maxRatio, avgRatio, stdDevRatio
}
