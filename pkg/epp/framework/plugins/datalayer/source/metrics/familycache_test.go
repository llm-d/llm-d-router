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

package metrics

import (
	"fmt"
	"io"
	"strings"
	"sync"
	"testing"

	"google.golang.org/protobuf/proto"
)

const cacheConfigFamily = "vllm:cache_config_info"

// cacheConfigLines returns a vllm:cache_config_info family with one sample of 40 labels, as vLLM exports.
func cacheConfigLines(blocks string) string {
	labels := make([]string, 0, 40)
	labels = append(labels, `num_gpu_blocks="`+blocks+`"`)
	for i := 1; i < 40; i++ {
		labels = append(labels, fmt.Sprintf(`label_%02d="value_%02d"`, i, i))
	}
	return "# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig\n" +
		"# TYPE vllm:cache_config_info gauge\n" +
		"vllm:cache_config_info{" + strings.Join(labels, ",") + "} 1.0\n"
}

func highLabelPage(blocks string) string {
	return "# HELP vllm:num_requests_running Number of requests in model execution batches.\n" +
		"# TYPE vllm:num_requests_running gauge\n" +
		`vllm:num_requests_running{engine="0",model_name="m"} 3.0` + "\n" +
		cacheConfigLines(blocks) +
		"# HELP vllm:num_requests_waiting Number of requests waiting to be processed.\n" +
		"# TYPE vllm:num_requests_waiting gauge\n" +
		`vllm:num_requests_waiting{engine="0",model_name="m"} 1.0` + "\n"
}

// parsers returns the whole-response and the filtered parser of a source, each with its own cache.
func parsers(t *testing.T) map[string]func(io.Reader) (PrometheusMetricMap, error) {
	t.Helper()
	whole, _ := newMetricsParser(nil)
	filtered, _ := newMetricsParser([]string{
		cacheConfigFamily, "vllm:num_requests_running", "vllm:num_requests_waiting",
		"vllm:e2e_request_latency_seconds", "vllm:prompt_tokens_total", "latency", "a",
	})
	return map[string]func(io.Reader) (PrometheusMetricMap, error){"whole": whole, "filtered": filtered}
}

func requireSameAsPlainParse(t *testing.T, page string, got PrometheusMetricMap) {
	t.Helper()
	want, err := parseMetrics(strings.NewReader(page))
	if err != nil {
		t.Fatalf("plain parse: %v", err)
	}
	for name, family := range got {
		if !proto.Equal(family, want[name]) {
			t.Errorf("family %s differs from the plain parse:\n got  %v\n want %v", name, family, want[name])
		}
	}
}

func TestMetricsParserReusesHighLabelFamily(t *testing.T) {
	page := highLabelPage("5330")
	for name, parse := range parsers(t) {
		t.Run(name, func(t *testing.T) {
			first, err := parse(strings.NewReader(page))
			if err != nil {
				t.Fatalf("first parse: %v", err)
			}
			second, err := parse(strings.NewReader(page))
			if err != nil {
				t.Fatalf("second parse: %v", err)
			}
			requireSameAsPlainParse(t, page, second)
			if first[cacheConfigFamily] == nil || first[cacheConfigFamily] != second[cacheConfigFamily] {
				t.Fatalf("%s was parsed again for unchanged lines", cacheConfigFamily)
			}
			if first["vllm:num_requests_running"] == second["vllm:num_requests_running"] {
				t.Fatal("a family with few labels was reused, want a fresh parse")
			}
			if len(second) != 3 {
				t.Fatalf("families = %v, want 3", sortedKeys(second))
			}
		})
	}
}

func TestMetricsParserReparsesChangedHighLabelFamily(t *testing.T) {
	for name, parse := range parsers(t) {
		t.Run(name, func(t *testing.T) {
			first, err := parse(strings.NewReader(highLabelPage("5330")))
			if err != nil {
				t.Fatalf("first parse: %v", err)
			}
			changed := highLabelPage("6000")
			second, err := parse(strings.NewReader(changed))
			if err != nil {
				t.Fatalf("second parse: %v", err)
			}
			requireSameAsPlainParse(t, changed, second)
			if first[cacheConfigFamily] == second[cacheConfigFamily] {
				t.Fatalf("%s was reused after its lines changed", cacheConfigFamily)
			}
		})
	}
}

func TestMetricsParserMatchesPlainParse(t *testing.T) {
	var highLabelBuckets strings.Builder
	highLabelBuckets.WriteString("# TYPE latency histogram\n")
	labels := make([]string, 0, 20)
	for i := range 20 {
		labels = append(labels, fmt.Sprintf(`l%02d="v"`, i))
	}
	for _, le := range []string{"1.0", "+Inf"} {
		highLabelBuckets.WriteString(`latency_bucket{` + strings.Join(labels, ",") + `,le="` + le + `"} 2` + "\n")
	}
	highLabelBuckets.WriteString(`latency_sum{` + strings.Join(labels, ",") + `} 3` + "\n")
	highLabelBuckets.WriteString(`latency_count{` + strings.Join(labels, ",") + `} 2` + "\n")

	pages := map[string]string{
		"vllm page":                        familyFilterPage + "\n",
		"high-label page":                  highLabelPage("5330"),
		"high-label histogram":             highLabelBuckets.String(),
		"untyped high-label family":        strings.Replace(cacheConfigLines("1"), "# TYPE vllm:cache_config_info gauge\n", "", 1),
		"samples of a family in two parts": cacheConfigLines("1") + "# TYPE a gauge\na 1\n" + `vllm:cache_config_info{engine="1"} 1.0` + "\n",
		"comments inside a family":         strings.Replace(cacheConfigLines("1"), "gauge\n", "gauge\n# a comment\n\n", 1) + "# TYPE a gauge\na 1\n",
	}
	for name, page := range pages {
		for parserName, parse := range parsers(t) {
			t.Run(name+"/"+parserName, func(t *testing.T) {
				for range 2 {
					got, err := parse(strings.NewReader(page))
					if err != nil {
						t.Fatalf("parse: %v", err)
					}
					requireSameAsPlainParse(t, page, got)
				}
			})
		}
	}
}

func TestMetricsParserKeepsPlainParseErrors(t *testing.T) {
	pages := map[string]string{
		"family declared twice":     cacheConfigLines("1") + "# TYPE a gauge\na 1\n" + cacheConfigLines("2"),
		"malformed high-label line": strings.Replace(cacheConfigLines("1"), "} 1.0", "} one", 1),
		"error after the family":    cacheConfigLines("1") + "# TYPE a gauge\na one\n",
	}
	whole, _ := newMetricsParser(nil)
	for name, page := range pages {
		t.Run(name, func(t *testing.T) {
			partial, want := parseMetrics(strings.NewReader(page))
			if want == nil {
				t.Fatal("plain parse succeeded, want an error")
			}
			for range 2 {
				got, err := whole(strings.NewReader(page))
				if err == nil || err.Error() != want.Error() {
					t.Fatalf("error = %v, want %v", err, want)
				}
				if keys, wantKeys := strings.Join(sortedKeys(got), ","), strings.Join(sortedKeys(partial), ","); keys != wantKeys {
					t.Fatalf("families = %s, want the plain parse's %s", keys, wantKeys)
				}
			}
		})
	}
}

func TestMetricsParserDoesNotCacheOversizedFamily(t *testing.T) {
	page := strings.Replace(cacheConfigLines("1"), `label_01="value_01"`,
		`label_01="`+strings.Repeat("x", maxCachedRunBytes)+`"`, 1)
	whole, _ := newMetricsParser(nil)
	first, err := whole(strings.NewReader(page))
	if err != nil {
		t.Fatalf("first parse: %v", err)
	}
	second, err := whole(strings.NewReader(page))
	if err != nil {
		t.Fatalf("second parse: %v", err)
	}
	requireSameAsPlainParse(t, page, second)
	if first[cacheConfigFamily] == second[cacheConfigFamily] {
		t.Fatalf("a family over %d bytes was cached", maxCachedRunBytes)
	}
}

func TestMetricsParserConcurrentParses(t *testing.T) {
	pages := []string{highLabelPage("1"), highLabelPage("2"), familyFilterPage + "\n"}
	for name, parse := range parsers(t) {
		t.Run(name, func(t *testing.T) {
			var wg sync.WaitGroup
			for i := range 16 {
				wg.Go(func() {
					page := pages[i%len(pages)]
					for range 20 {
						got, err := parse(strings.NewReader(page))
						if err != nil {
							t.Errorf("parse: %v", err)
							return
						}
						requireSameAsPlainParse(t, page, got)
					}
				})
			}
			wg.Wait()
		})
	}
}

func BenchmarkMetricsParser(b *testing.B) {
	// The vLLM page with its cache_config_info sample widened to the 40 labels vLLM exports.
	page := strings.Replace(familyFilterPage,
		`vllm:cache_config_info{block_size="64",engine="0",num_gpu_blocks="5330"} 1.0`,
		strings.Split(cacheConfigLines("5330"), "\n")[2], 1) + "\n"
	whole, _ := newMetricsParser(nil)
	b.Run("plain", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if _, err := parseMetrics(strings.NewReader(page)); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("source", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if _, err := whole(strings.NewReader(page)); err != nil {
				b.Fatal(err)
			}
		}
	})
}
