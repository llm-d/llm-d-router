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
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker/weightedrandom"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

func TestExplorationBudgetLoading(t *testing.T) {
	registerTestPlugins(t)
	fwkplugin.Register(weightedrandom.WeightedRandomPickerType, fwkplugin.StabilityStable, weightedrandom.WeightedRandomPickerFactory)
	for _, version := range []string{"v1", "v1alpha1"} {
		for _, temperature := range []string{"", "0.5", "1", "2", "0", "-1", ".nan", ".inf"} {
			for _, budget := range []string{"", "0", "0.05", "1", "-0.01", "1.01", ".nan", ".inf"} {
				t.Run(version+"/temperature="+temperature+"/budget="+budget, func(t *testing.T) {
					parameters := "  parameters:\n"
					if temperature != "" {
						parameters += "    temperature: " + temperature + "\n"
					}
					if budget != "" {
						parameters += "    explorationBudget: " + budget + "\n"
					}
					text := fmt.Sprintf("apiVersion: llm-d.ai/%s\nkind: EndpointPickerConfig\nplugins:\n- type: weighted-random-picker\n%s", version, parameters)
					logger := logging.NewTestLogger()
					raw, _, err := LoadRawConfig([]byte(text), logger)
					if err == nil {
						_, err = InstantiateAndConfigure(raw, testutils.NewTestHandle(context.Background()), logger)
					}
					invalid := temperature == "0" || temperature == "-1" || temperature == ".nan" || temperature == ".inf" || strings.HasPrefix(budget, "-") || budget == "1.01" || budget == ".nan" || budget == ".inf"
					if (err != nil) != invalid {
						t.Fatalf("budget %q: error %v, want invalid %v", budget, err, invalid)
					}
				})
			}
		}
	}
}
