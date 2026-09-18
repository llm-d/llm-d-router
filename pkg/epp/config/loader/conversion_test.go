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

package loader

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"

	configapiv1 "github.com/llm-d/llm-d-router/apix/config/v1"
	configapiv1alpha1 "github.com/llm-d/llm-d-router/apix/config/v1alpha1"
	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
)

// equivalentV1alpha1Text and equivalentV1Text express the same configuration in
// both accepted apiVersions, covering every field whose placement differs.
const equivalentV1alpha1Text = `
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- name: maxScore
  type: max-score-picker
- name: syncer
  type: local-syncer
- name: my-disc
  type: file-discovery
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: maxScore
dataLayer:
  injectDefaults: false
  discovery:
    pluginRef: my-disc
  crossReplicaSyncerPluginRef: syncer
  crossReplicaSyncInterval: 500ms
  crossReplicaPublishTimeout: 2s
flowControl:
  maxRequests: "1k"
  defaultRequestTTL: 30s
requestHandler:
  propagatePriority: true
`

const equivalentV1Text = `
apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- name: maxScore
  type: max-score-picker
- name: syncer
  type: local-syncer
- name: my-disc
  type: file-discovery
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: maxScore
dataLayer:
  injectDefaults: false
  discovery:
    endpoints:
      pluginRef: my-disc
  crossReplica:
    syncerPluginRef: syncer
    syncInterval: 500ms
    publishTimeout: 2s
flowControl:
  maxRequests: "1k"
  defaultRequestTTL: 30s
requestHandler:
  propagatePriority: true
`

// TestV1alpha1LoadsAsV1 pins the migration contract: a v1alpha1 configuration and
// its v1 equivalent produce the same effective configuration, and only the former
// reports a deprecation.
func TestV1alpha1LoadsAsV1(t *testing.T) {
	t.Parallel()

	oldWriter := &strings.Builder{}
	oldCfg, oldGates, err := LoadRawConfig([]byte(equivalentV1alpha1Text), logging.NewTestLoggerWithWriter(oldWriter))
	require.NoError(t, err, "v1alpha1 configuration must still load")

	newWriter := &strings.Builder{}
	newCfg, newGates, err := LoadRawConfig([]byte(equivalentV1Text), logging.NewTestLoggerWithWriter(newWriter))
	require.NoError(t, err, "v1 configuration must load")

	require.Empty(t, cmp.Diff(newCfg, oldCfg), "v1alpha1 must convert to the v1 equivalent")
	require.Equal(t, newGates, oldGates)

	require.Contains(t, oldWriter.String(), "deprecated", "loading v1alpha1 must report a deprecation")
	require.NotContains(t, newWriter.String(), "deprecated", "loading v1 must not report a deprecation")
}

// TestConversionDropsNoField verifies that converting a fully populated
// v1alpha1 configuration preserves every field. It also verifies the
// intentional moves to the v1 field layout.
func TestConversionDropsNoField(t *testing.T) {
	t.Parallel()

	in := fullyPopulatedV1alpha1()
	requireNoZeroField(t, reflect.ValueOf(in), "EndpointPickerConfig")

	got := asMap(t, convertV1alpha1ToV1(logging.NewTestLogger(), in))

	want := asMap(t, in)
	want["apiVersion"] = configapiv1.GroupVersion.String()
	dataLayer := want["dataLayer"].(map[string]any)
	dataLayer["crossReplica"] = map[string]any{
		"syncerPluginRef": dataLayer["crossReplicaSyncerPluginRef"],
		"syncInterval":    dataLayer["crossReplicaSyncInterval"],
		"publishTimeout":  dataLayer["crossReplicaPublishTimeout"],
	}
	delete(dataLayer, "crossReplicaSyncerPluginRef")
	delete(dataLayer, "crossReplicaSyncInterval")
	delete(dataLayer, "crossReplicaPublishTimeout")
	// endpoints is already set, so the deprecated bare pluginRef is discarded.
	delete(dataLayer["discovery"].(map[string]any), "pluginRef")

	require.Empty(t, cmp.Diff(want, got), "conversion lost or altered a field (-want +got)")
}

func asMap(t *testing.T, obj any) map[string]any {
	t.Helper()
	raw, err := json.Marshal(obj)
	require.NoError(t, err)
	var out map[string]any
	require.NoError(t, json.Unmarshal(raw, &out))
	return out
}

func requireNoZeroField(t *testing.T, v reflect.Value, path string) {
	t.Helper()

	switch v.Kind() {
	case reflect.Pointer:
		require.Falsef(t, v.IsNil(), "%s is nil", path)
		if v.Elem().Kind() == reflect.Struct {
			requireNoZeroField(t, v.Elem(), path)
		}
	case reflect.Struct:
		// Quantity and Duration carry unexported state; treat them as leaves.
		switch v.Type().String() {
		case "resource.Quantity", "v1.Duration":
			require.Falsef(t, v.IsZero(), "%s is zero", path)
			return
		}
		for i := range v.NumField() {
			field := v.Type().Field(i)
			if !field.IsExported() {
				continue
			}
			requireNoZeroField(t, v.Field(i), path+"."+field.Name)
		}
	case reflect.Slice:
		require.NotZerof(t, v.Len(), "%s is empty", path)
		if v.Type().Elem().Kind() != reflect.Uint8 { // json.RawMessage
			requireNoZeroField(t, v.Index(0), path+"[0]")
		}
	default:
		require.Falsef(t, v.IsZero(), "%s is zero", path)
	}
}

func fullyPopulatedV1alpha1() *configapiv1alpha1.EndpointPickerConfig {
	band := configapiv1alpha1.PriorityBandConfig{
		Priority:          10,
		MaxBytes:          resource.NewQuantity(1024, resource.DecimalSI),
		MaxRequests:       resource.NewQuantity(64, resource.DecimalSI),
		DefaultRequestTTL: &metav1.Duration{Duration: 5 * time.Second},
		FairnessPolicyRef: "fairness",
		OrderingPolicyRef: "ordering",
	}

	return &configapiv1alpha1.EndpointPickerConfig{
		TypeMeta: metav1.TypeMeta{
			Kind:       "EndpointPickerConfig",
			APIVersion: configapiv1alpha1.GroupVersion.String(),
		},
		FeatureGates: configapiv1alpha1.FeatureGates{"gate=true"},
		Plugins: []configapiv1alpha1.PluginSpec{
			{Name: "plugin", Type: "plugin-type", Parameters: json.RawMessage(`{"k":"v"}`)},
		},
		SchedulingProfiles: []configapiv1alpha1.SchedulingProfile{
			{Name: "default", Plugins: []configapiv1alpha1.SchedulingPlugin{{PluginRef: "plugin", Weight: ptr.To(1.5)}}},
		},
		DataLayer: &configapiv1alpha1.DataLayerConfig{
			InjectDefaults: ptr.To(true),
			Sources: []configapiv1alpha1.DataLayerSource{
				{PluginRef: "source", Extractors: []configapiv1alpha1.DataLayerExtractor{{PluginRef: "extractor"}}},
			},
			Discovery: &configapiv1alpha1.DiscoveryConfig{
				Endpoints: &configapiv1alpha1.EndpointDiscoveryConfig{PluginRef: "endpoints"},
				Peers:     &configapiv1alpha1.PeerDiscoveryConfig{PluginRef: "peers"},
				PluginRef: "bare-endpoints",
			},
			CrossReplicaSyncerPluginRef: "syncer",
			CrossReplicaSyncInterval:    &metav1.Duration{Duration: 500 * time.Millisecond},
			CrossReplicaPublishTimeout:  &metav1.Duration{Duration: 2 * time.Second},
		},
		FlowControl: &configapiv1alpha1.FlowControlConfig{
			MaxBytes:                    resource.NewQuantity(2048, resource.DecimalSI),
			MaxRequests:                 resource.NewQuantity(128, resource.DecimalSI),
			DefaultRequestTTL:           &metav1.Duration{Duration: 30 * time.Second},
			NoEndpointRequestTTL:        &metav1.Duration{Duration: 90 * time.Second},
			DefaultPriorityBand:         &band,
			DefaultNegativePriorityBand: &band,
			PriorityBands:               []configapiv1alpha1.PriorityBandConfig{band},
			UsageLimitPolicyPluginRef:   "usage-limit",
			SaturationDetector:          &configapiv1alpha1.SaturationDetectorConfig{PluginRef: "detector"},
			EnableEviction:              true,
		},
		RequestHandler: &configapiv1alpha1.RequestHandlerConfig{
			Parsers:           []configapiv1alpha1.ParserConfig{{PluginRef: "parser"}},
			PropagatePriority: true,
		},
	}
}
