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

package semconv

import (
	"go.opentelemetry.io/otel/attribute"
	otelsemconv "go.opentelemetry.io/otel/semconv/v1.41.0"
)

// Standard OpenTelemetry GenAI Semantic Convention attribute keys.
// Official semconv constants are re-exported for consistent centralized imports across the codebase.
const (
	// GenAIRequestModelKey is the name of the GenAI model a request is targeting.
	GenAIRequestModelKey = otelsemconv.GenAIRequestModelKey

	// GenAIRequestIDKey is the unique identifier for the GenAI inference request.
	GenAIRequestIDKey = attribute.Key("gen_ai.request.id")
)

// Typed helper functions for building GenAI KeyValues safely without hardcoded strings.

// GenAIRequestModel returns an attribute for the target GenAI request model.
func GenAIRequestModel(model string) attribute.KeyValue {
	return otelsemconv.GenAIRequestModel(model)
}

// GenAIRequestID returns an attribute for the GenAI request ID.
func GenAIRequestID(requestID string) attribute.KeyValue {
	return GenAIRequestIDKey.String(requestID)
}
