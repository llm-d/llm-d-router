package models

import (
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

func TestExtractorExtract(t *testing.T) {
	ctx := context.Background()

	extPlugin, err := ModelServerExtractorFactory("test-extractor", nil, nil)
	if err != nil {
		t.Fatalf("failed to create extractor: %v", err)
	}
	extractor := extPlugin.(fwkdl.Extractor)

	if exType := extPlugin.TypedName().Type; exType == "" {
		t.Error("empty extractor type")
	}

	if exName := extPlugin.TypedName().Name; exName == "" {
		t.Error("empty extractor name")
	}

	if inputType := extractor.ExpectedInputType(); inputType != ModelsResponseType {
		t.Errorf("incorrect expected input type: %v", inputType)
	}

	ep := fwkdl.NewEndpoint(nil, nil)
	if ep == nil {
		t.Fatal("expected non-nil endpoint")
	}

	model := "food-review"

	tests := []struct {
		name    string
		data    any
		wantErr bool
		updated bool // whether metrics are expected to change
	}{
		{
			name:    "nil data",
			data:    nil,
			wantErr: true,
			updated: false,
		},
		{
			name:    "empty ModelsResponse",
			data:    &ModelResponse{},
			wantErr: false,
			updated: false,
		},
		{
			name: "valid models response",
			data: &ModelResponse{
				Object: "list",
				Data: []ModelData{
					{
						ID: model,
					},
					{
						ID:     "lora1",
						Parent: model,
					},
				},
			},
			wantErr: false,
			updated: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("Extract panicked: %v", r)
				}
			}()

			attr := ep.GetAttributes()
			before, ok := attr.Get(ModelsAttributeKey)
			if ok && before != nil {
				t.Error("expected empty attributes")
			}
			err := extractor.Extract(ctx, tt.data, ep)
			after, ok := attr.Get(ModelsAttributeKey)
			if !ok && tt.updated {
				t.Error("expected updated attributes")
			}

			if tt.wantErr && err == nil {
				t.Errorf("expected error but got nil")
			}
			if !tt.wantErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			if tt.updated {
				if diff := cmp.Diff(before, after); diff == "" {
					t.Errorf("expected models to be updated, but no change detected")
				}
			} else {
				if diff := cmp.Diff(before, after); diff != "" {
					t.Errorf("expected no models update, but got changes:\n%s", diff)
				}
			}
		})
	}
}
