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

package tracing

import (
	"context"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/propagation"
	grpcmetadata "google.golang.org/grpc/metadata"
)

type grpcMetadataCarrier struct {
	metadata grpcmetadata.MD
}

var _ propagation.TextMapCarrier = (*grpcMetadataCarrier)(nil)

func (c *grpcMetadataCarrier) Get(key string) string {
	values := c.metadata.Get(key)
	if len(values) == 0 {
		return ""
	}
	return values[0]
}

func (c *grpcMetadataCarrier) Set(key, value string) {
	c.metadata.Set(key, value)
}

func (c *grpcMetadataCarrier) Keys() []string {
	keys := make([]string, 0, len(c.metadata))
	for key := range c.metadata {
		keys = append(keys, key)
	}
	return keys
}

// ExtractGRPCMetadata extracts trace context from incoming gRPC metadata using
// the globally configured OpenTelemetry text map propagator.
func ExtractGRPCMetadata(ctx context.Context) context.Context {
	md, ok := grpcmetadata.FromIncomingContext(ctx)
	if !ok {
		return ctx
	}
	return otel.GetTextMapPropagator().Extract(ctx, &grpcMetadataCarrier{metadata: md})
}
