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

package epp

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	eppharness "github.com/llm-d/llm-d-router/test/framework/epp/harness"
)

// TestMockMetricsSourceConfigsOptOut guards against reintroducing the flake fixed by opting
// mock-metrics-source testdata configs out of default metrics-source injection: any config
// declaring mock-metrics-source must also set dataLayer.injectDefaults: false.
func TestMockMetricsSourceConfigsOptOut(t *testing.T) {
	paths, err := filepath.Glob("testdata/*.yaml")
	if err != nil {
		t.Fatalf("failed to glob testdata configs: %v", err)
	}
	if len(paths) == 0 {
		t.Fatal("no testdata configs found matching testdata/*.yaml")
	}
	for _, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("failed to read %s: %v", path, err)
		}
		if strings.Contains(string(data), eppharness.MockDataSourceType) && !strings.Contains(string(data), "injectDefaults: false") {
			t.Errorf("%s configures %s without dataLayer.injectDefaults: false", path, eppharness.MockDataSourceType)
		}
	}
}
