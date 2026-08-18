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

import "time"

// IncRequestTotal increments request_total for the given model.
func IncRequestTotal(modelName string) {
	requestTotal.WithLabelValues(boundModel(modelName)).Inc()
}

// IncRequestErrorTotal increments request_error_total for the given model
// and error code.
func IncRequestErrorTotal(modelName, errorCode string) {
	requestErrorTotal.WithLabelValues(boundModel(modelName), errorCode).Inc()
}

// RecordRequestDuration observes an end-to-end request latency.
func RecordRequestDuration(modelName string, d time.Duration) {
	requestDuration.WithLabelValues(boundModel(modelName)).Observe(d.Seconds())
}

// RecordRequestSize observes a request body size in bytes.
func RecordRequestSize(modelName string, bytes int) {
	requestSize.WithLabelValues(boundModel(modelName)).Observe(float64(bytes))
}

// IncRequestRunning increments the in-flight gauge for the given model.
func IncRequestRunning(modelName string) {
	requestRunning.WithLabelValues(boundModel(modelName)).Inc()
}

// DecRequestRunning decrements the in-flight gauge for the given model. It
// must be called exactly once per IncRequestRunning to stay balanced.
func DecRequestRunning(modelName string) {
	requestRunning.WithLabelValues(boundModel(modelName)).Dec()
}
