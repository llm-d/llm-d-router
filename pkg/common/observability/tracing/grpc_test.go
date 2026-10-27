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
	"testing"

	"github.com/stretchr/testify/assert"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/trace"
	grpcmetadata "google.golang.org/grpc/metadata"
)

type testContextKey struct{}

func TestExtractGRPCMetadata(t *testing.T) {
	prevPropagator := otel.GetTextMapPropagator()
	otel.SetTextMapPropagator(propagation.TraceContext{})
	t.Cleanup(func() { otel.SetTextMapPropagator(prevPropagator) })

	const traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0c9902b7-01"
	ctx := grpcmetadata.NewIncomingContext(context.Background(), grpcmetadata.Pairs("traceparent", traceparent))

	sc := trace.SpanContextFromContext(ExtractGRPCMetadata(ctx))

	assert.True(t, sc.IsValid())
	assert.Equal(t, "4bf92f3577b34da6a3ce929d0e0e4736", sc.TraceID().String())
	assert.True(t, sc.IsRemote())
}

func TestExtractGRPCMetadataWithoutIncomingMetadata(t *testing.T) {
	ctx := context.WithValue(context.Background(), testContextKey{}, "value")

	assert.Same(t, ctx, ExtractGRPCMetadata(ctx))
}
