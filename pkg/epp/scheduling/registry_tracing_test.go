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

package scheduling

import (
	"context"
	"os"
	"strings"
	"testing"

	"go.opentelemetry.io/otel/attribute"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/yaml"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// Exercise the actual scheduling path against the handwritten registry.
func TestSchedulingSpansMatchRegistry(t *testing.T) {
	var registry struct {
		Groups []struct {
			SpanKind    string            `json:"span_kind"`
			Annotations map[string]string `json:"annotations"`
			Attributes  []struct {
				ID   string `json:"id"`
				Ref  string `json:"ref"`
				Type string `json:"type"`
			} `json:"attributes"`
		} `json:"groups"`
	}
	load := func(file string) {
		t.Helper()
		data, err := os.ReadFile("../../common/observability/semconv/registry/" + file)
		if err != nil {
			t.Fatal(err)
		}
		if err := yaml.Unmarshal(data, &registry); err != nil {
			t.Fatal(err)
		}
	}
	types := map[string]attribute.Type{
		"string": attribute.STRING, "int": attribute.INT64,
		"double": attribute.FLOAT64, "boolean": attribute.BOOL,
		"string[]": attribute.STRINGSLICE, "double[]": attribute.FLOAT64SLICE,
	}
	load("attributes.yaml")
	attrTypes := map[string]attribute.Type{}
	for _, group := range registry.Groups {
		for _, attr := range group.Attributes {
			attrType, ok := types[attr.Type]
			if !ok {
				t.Fatalf("unsupported attribute type %q", attr.Type)
			}
			attrTypes[attr.ID] = attrType
		}
	}
	load("spans.yaml")
	if len(registry.Groups) != 4 {
		t.Fatalf("expected four cataloged scheduling spans, got %d", len(registry.Groups))
	}

	recorder := setupSpanRecorder(t)
	plugin := &testPlugin{
		typedName: fwkplugin.TypedName{Type: "test-plugin", Name: "registry"},
		FilterRes: []k8stypes.NamespacedName{{Name: "pod1"}},
		PickRes:   k8stypes.NamespacedName{Name: "pod1"},
	}
	profile := NewSchedulerProfile().WithFilters(plugin).WithPicker(plugin)
	_, err := runSchedulerProfile(context.Background(), "decode", profile,
		&fwksched.InferenceRequest{TargetModel: "m1", RequestID: "r1"}, newTestEndpoints("pod1", "pod2"))
	if err != nil {
		t.Fatal(err)
	}
	for _, group := range registry.Groups {
		name := group.Annotations["span.name"]
		t.Run(name, func(t *testing.T) {
			spans := findSpans(recorder.Ended(), name)
			if len(spans) != 1 {
				t.Fatalf("got %d spans for %q, want one", len(spans), name)
			}
			span := spans[0]
			if strings.ToLower(span.SpanKind().String()) != group.SpanKind {
				t.Fatalf("span kind %v does not match registry %q", span.SpanKind(), group.SpanKind)
			}
			attrs := spanAttributes(span)
			for _, attr := range group.Attributes {
				wantType, ok := attrTypes[attr.Ref]
				if !ok {
					t.Fatalf("unresolved attribute %q", attr.Ref)
				}
				value, ok := attrs[attribute.Key(attr.Ref)]
				if !ok || value.Type() != wantType {
					t.Errorf("attribute %q: got %v (present=%v), want %v", attr.Ref, value.Type(), ok, wantType)
				}
				delete(attrs, attribute.Key(attr.Ref))
			}
			for key := range attrs {
				if strings.HasPrefix(string(key), "llm_d.") {
					t.Errorf("uncataloged custom attribute %q", key)
				}
			}
		})
	}
}
