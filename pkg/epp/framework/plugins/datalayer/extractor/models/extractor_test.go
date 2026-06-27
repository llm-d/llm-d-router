package models

import (
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	attrmodels "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/models"
)

func TestExtractorExtract(t *testing.T) {
	ctx := context.Background()

	extPlugin, err := ModelServerExtractorFactory("test-extractor", nil, nil)
	if err != nil {
		t.Fatalf("failed to create extractor: %v", err)
	}
	extractor := extPlugin.(*ModelExtractor)

	if exType := extPlugin.TypedName().Type; exType == "" {
		t.Error("empty extractor type")
	}

	if exName := extPlugin.TypedName().Name; exName == "" {
		t.Error("empty extractor name")
	}

	ep := fwkdl.NewEndpoint(nil, nil)
	if ep == nil {
		t.Fatal("expected non-nil endpoint")
	}

	model := "food-review"

	// Note: nil Payload is the dispatcher's responsibility per the
	// PollingDispatcher contract; Extract does not guard against it.
	tests := []struct {
		name    string
		data    *ModelResponse
		wantErr bool
		updated bool // whether metrics are expected to change
	}{
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
				Data: []attrmodels.ModelData{
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

	key := attrmodels.ModelsAttributeKey.WithNonEmptyProducerName(attrmodels.ModelsExtractorType).String()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("Extract panicked: %v", r)
				}
			}()

			attr := ep.GetAttributes()
			before, ok := attr.Get(key)
			if ok && before != nil {
				t.Error("expected empty attributes")
			}
			err := extractor.Extract(ctx, fwkdl.PollInput[*ModelResponse]{Payload: tt.data, Endpoint: ep})
			after, ok := attr.Get(key)
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

// TestExtractorCapturesMaxModelLen verifies the extractor surfaces max_model_len
// from the /v1/models payload onto the endpoint attribute.
func TestExtractorCapturesMaxModelLen(t *testing.T) {
	ctx := context.Background()
	extPlugin, err := ModelServerExtractorFactory("test-extractor", nil, nil)
	if err != nil {
		t.Fatalf("failed to create extractor: %v", err)
	}
	extractor := extPlugin.(*ModelExtractor)
	ep := fwkdl.NewEndpoint(nil, nil)

	resp := &ModelResponse{Data: []attrmodels.ModelData{{ID: "m", MaxModelLen: 131072}}}
	if err := extractor.Extract(ctx, fwkdl.PollInput[*ModelResponse]{Payload: resp, Endpoint: ep}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	val, ok := ep.GetAttributes().Get(attrmodels.ModelsAttributeKey.String())
	if !ok {
		t.Fatal("expected models attribute to be set")
	}
	models, ok := val.(attrmodels.ModelDataCollection)
	if !ok {
		t.Fatalf("unexpected attribute type %T", val)
	}
	if len(models) != 1 || models[0].MaxModelLen != 131072 {
		t.Errorf("expected max_model_len 131072, got %+v", models)
	}
}
