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

package groupsched

// DOCS CAN'T DRIFT: every plugin block a README shows has to decode through the plugin's OWN
// factory, which is strict — a retired or misspelled parameter fails here rather than on a deploy.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"sigs.k8s.io/yaml"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/threadidentity"
)

// readmes are the documents that promise a configuration.
var readmes = map[string]string{
	"group scheduler": filepath.Join("..", "groupsched", "README.md"),
	"thread identity": filepath.Join("..", "threadidentity", "README.md"),
}

// yamlBlocks is every ```yaml block of a document, in order.
func yamlBlocks(t *testing.T, path string) []string {
	t.Helper()
	raw, err := os.ReadFile(path)
	require.NoError(t, err)
	re := regexp.MustCompile("(?s)```yaml\n(.*?)```")
	var out []string
	for _, m := range re.FindAllStringSubmatch(string(raw), -1) {
		out = append(out, m[1])
	}
	require.NotEmpty(t, out, "%s carries no yaml example", path)
	return out
}

// paramsOf is the `parameters` block of the plugin of the given type, as the factory would receive
// it. ok is false when the block names no such plugin.
func paramsOf(t *testing.T, block, pluginType string) (string, bool) {
	t.Helper()
	var plugins []map[string]any
	if err := yaml.Unmarshal([]byte(block), &plugins); err != nil {
		// a partial configuration (`plugins:` and nothing else) rather than a bare plugin list
		var cfg struct {
			Plugins []map[string]any `json:"plugins"`
		}
		require.NoError(t, yaml.Unmarshal([]byte(block), &cfg), "a README yaml block is neither a plugin list nor a configuration")
		plugins = cfg.Plugins
	}
	for _, p := range plugins {
		if p["type"] != pluginType {
			continue
		}
		params, ok := p["parameters"]
		if !ok {
			return "", false
		}
		raw, err := json.Marshal(params)
		require.NoError(t, err)
		return string(raw), true
	}
	return "", false
}

// TestReadmeExamplesDecodeWithStrictFactories: every plugin block in both READMEs decodes through
// its own Factory — an unknown or retired parameter fails here.
func TestReadmeExamplesDecodeWithStrictFactories(t *testing.T) {
	checked := 0
	for name, path := range readmes {
		for i, block := range yamlBlocks(t, path) {
			if strings.Contains(block, "EndpointPickerConfig") {
				continue // a complete configuration, not a plugin block
			}
			if p, ok := paramsOf(t, block, PluginType); ok {
				_, err := Factory("group", json.NewDecoder(strings.NewReader(p)), nil)
				require.NoError(t, err, "%s README: yaml block %d, %s parameters do not decode", name, i, PluginType)
				checked++
			}
			if p, ok := paramsOf(t, block, threadidentity.PluginType); ok {
				_, err := threadidentity.Factory("bi", json.NewDecoder(strings.NewReader(p)), nil)
				require.NoError(t, err, "%s README: yaml block %d, %s parameters do not decode", name, i, threadidentity.PluginType)
				checked++
			}
		}
	}
	require.GreaterOrEqual(t, checked, 4, "both READMEs must carry a decodable example for each plugin")
}
