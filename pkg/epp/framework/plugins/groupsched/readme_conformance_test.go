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

package groupsched_test

// DOCS CAN'T DRIFT.
//
// A README is a promise about the code, and nothing else in the tree checks it. Two promises are
// machine-checkable, so they are checked here, on both plugins' READMEs:
//
//  1. EVERY fenced ```yaml block that is a plugin configuration is loaded through the REAL config
//     loader and instantiated (which runs each plugin's STRICT parameter decoder). A renamed or
//     retired parameter in an example therefore fails the test, not the reader's deployment.
//  2. EVERY metric name the README mentions in backticks exists in the surface the plugins
//     register, and every series the plugins register is mentioned. A series added without a
//     README line, or a README line without a series, is a failure either way.
//
// The plugin blocks of an example are decoded by their own Factory (the same decoder the loader
// uses); the full example additionally goes through loader.LoadRawConfig so its shape — plugin
// refs, flow-control wiring, feature gates — is checked too.

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"

	"github.com/go-logr/logr"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/require"

	"sigs.k8s.io/yaml"

	"github.com/llm-d/llm-d-router/pkg/epp/config/loader"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/groupsched"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/threadidentity"
)

var readmes = map[string]string{
	"group scheduler": filepath.Join("..", "groupsched", "README.md"),
	"thread identity": filepath.Join("..", "threadidentity", "README.md"),
}

// yamlBlocks returns every ```yaml fenced block of a README.
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

// paramsOf extracts one plugin's `parameters:` mapping out of a YAML plugin list, as JSON — the
// input its Factory decodes. The list is decoded as data, so nothing here has to understand YAML.
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
		out, err := json.Marshal(params)
		require.NoError(t, err)
		return string(out), true
	}
	return "", false
}

// TestReadmeExamplesDecodeWithStrictFactories: every plugin block in both READMEs decodes through
// its own Factory, which is STRICT — an unknown or retired parameter fails here.
func TestReadmeExamplesDecodeWithStrictFactories(t *testing.T) {
	checked := 0
	for name, path := range readmes {
		for i, block := range yamlBlocks(t, path) {
			if strings.Contains(block, "EndpointPickerConfig") {
				continue // a complete configuration: the loader test covers it end to end
			}
			if p, ok := paramsOf(t, block, groupsched.PluginType); ok {
				_, err := groupsched.Factory("group", json.NewDecoder(strings.NewReader(p)), nil)
				require.NoError(t, err, "%s README: yaml block %d, %s parameters do not decode", name, i, groupsched.PluginType)
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

// TestReadmeFullExampleLoads: the complete EndpointPickerConfig in the group scheduler's README is
// a real configuration — the loader accepts its shape, feature gates and plugin refs.
func TestReadmeFullExampleLoads(t *testing.T) {
	blocks := yamlBlocks(t, readmes["group scheduler"])
	var full string
	for _, b := range blocks {
		if strings.Contains(b, "kind: EndpointPickerConfig") {
			full = b
			break
		}
	}
	require.NotEmpty(t, full, "the group scheduler README must carry a complete example")

	_, _, err := loader.LoadRawConfig([]byte(full), logr.Discard())
	require.NoError(t, err, "the README's complete example must load through the real loader")
}

// metricShaped reports whether a backticked identifier is a metric name: the plugin's series all
// end in one of these suffixes, so a table row that says otherwise names nothing.
func metricShaped(name string) bool {
	for _, suffix := range []string{"_total", "_seconds", "_ns"} {
		if strings.HasSuffix(name, suffix) {
			return true
		}
	}
	return false
}

// registeredSeries registers one plugin on a fresh registry and returns its series names with the
// subsystem prefix removed, so a README table can use the bare names.
func registeredSeries(t *testing.T, register func(prometheus.Registerer) error, subsystem string) map[string]bool {
	t.Helper()
	reg := prometheus.NewRegistry()
	require.NoError(t, register(reg))
	// DESCRIBE, not Gather: a counter vec with no samples yet is registered and scarpable while
	// Gather omits it, and the README documents the series, not today's samples.
	descs := make(chan *prometheus.Desc, 256)
	go func() {
		reg.Describe(descs)
		close(descs)
	}()
	names := map[string]bool{}
	fq := regexp.MustCompile(`fqName: "([^"]+)"`)
	for d := range descs {
		m := fq.FindStringSubmatch(d.String())
		if m == nil {
			continue
		}
		names[strings.TrimPrefix(m[1], subsystem+"_")] = true
	}
	require.NotEmpty(t, names, "%s registers no series", subsystem)
	return names
}

// TestReadmeMetricsMatchTheSurface: for EACH plugin, the metric names its README names in backticks
// are exactly the series that plugin registers — no line without a series, no series without a line.
func TestReadmeMetricsMatchTheSurface(t *testing.T) {
	cases := []struct {
		name      string
		readme    string
		subsystem string
		register  func(prometheus.Registerer) error
	}{
		{
			name:      "group scheduler",
			readme:    readmes["group scheduler"],
			subsystem: "llm_d_epp_group",
			register: func(reg prometheus.Registerer) error {
				handle := fwkplugin.NewEppHandle(context.Background(), nil, fwkplugin.WithMetricsRecorder(reg))
				_, err := groupsched.Factory("group", nil, handle)
				return err
			},
		},
		{
			name:      "thread identity",
			readme:    readmes["thread identity"],
			subsystem: "llm_d_epp_thread",
			register: func(reg prometheus.Registerer) error {
				handle := fwkplugin.NewEppHandle(context.Background(), nil, fwkplugin.WithMetricsRecorder(reg))
				_, err := threadidentity.Factory("bi", json.NewDecoder(strings.NewReader(
					`{"tokenProducer":{"modelName":"m","vllm":{"url":"http://127.0.0.1:1"}}}`)), handle)
				return err
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			surface := registeredSeries(t, tc.register, tc.subsystem)

			raw, err := os.ReadFile(tc.readme)
			require.NoError(t, err)
			mentioned := map[string]bool{}
			var undocumented []string
			for _, m := range regexp.MustCompile("`([a-z0-9_]+)`").FindAllStringSubmatch(string(raw), -1) {
				name := strings.TrimPrefix(m[1], tc.subsystem+"_")
				name = strings.TrimPrefix(name, "llm_d_epp_group_")
				name = strings.TrimPrefix(name, "llm_d_epp_thread_")
				if surface[name] {
					mentioned[name] = true
					continue
				}
				// A metric-SHAPED name that the plugin does not register is drift too: the table row
				// would document a series nobody can scrape.
				if metricShaped(name) && !strings.Contains(name, "llm_d") {
					undocumented = append(undocumented, name)
				}
			}

			var missingDoc, missingSeries []string
			for series := range surface {
				if !mentioned[series] {
					missingDoc = append(missingDoc, series)
				}
			}
			for series := range mentioned {
				if !surface[series] {
					missingSeries = append(missingSeries, series)
				}
			}
			sort.Strings(missingDoc)
			sort.Strings(missingSeries)
			require.Zero(t, len(undocumented), "the README names metric-shaped identifiers this plugin does not register: %s",
				strings.Join(undocumented, ", "))
			require.Zero(t, len(missingSeries), "the README documents series that do not exist: %s", strings.Join(missingSeries, ", "))
			require.Zero(t, len(missingDoc), "the plugin registers series the README does not document: %s", strings.Join(missingDoc, ", "))
		})
	}
}
