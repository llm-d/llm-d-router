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

package metrics

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestDataSourceConfigParams_UsesBuiltInDefaults(t *testing.T) {
	cfg, err := defaultDataSourceConfigParams()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := &metricsDatasourceParams{
		Scheme:             defaultMetricsScheme,
		Path:               defaultMetricsPath,
		InsecureSkipVerify: defaultMetricsInsecureSkipVerify,
	}

	if !reflect.DeepEqual(cfg, expected) {
		t.Fatalf("expected %+v, got %+v", expected, cfg)
	}
}

func TestDataSourceConfigParams_UsesConfig(t *testing.T) {
	cfg, err := defaultDataSourceConfigParams()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Config JSON overrides
	parameters := metricsDatasourceParams{
		Scheme:             "https",
		Path:               "/custom-metrics",
		InsecureSkipVerify: false,
	}

	raw, err := json.Marshal(parameters)
	if err != nil {
		t.Fatalf("failed to marshal parameters: %v", err)
	}

	if err := json.Unmarshal(raw, cfg); err != nil {
		t.Fatalf("unexpected error unmarshalling: %v", err)
	}

	expected := &metricsDatasourceParams{
		Scheme:             "https",
		Path:               "/custom-metrics",
		InsecureSkipVerify: false,
	}

	if !reflect.DeepEqual(cfg, expected) {
		t.Fatalf("expected %+v, got %+v", expected, cfg)
	}
}
