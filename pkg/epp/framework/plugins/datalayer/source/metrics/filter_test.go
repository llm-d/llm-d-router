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
	"bytes"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// namerStub is an extractor-shaped plugin that declares metric families.
type namerStub struct {
	names []string
}

func (n namerStub) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "stub", Name: "stub"}
}
func (n namerStub) MetricNames() []string { return n.names }

// plainStub is a plugin that declares nothing, as extractors predating
// FamilyNamer do.
type plainStub struct{}

func (plainStub) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "plain", Name: "plain"}
}

func nameSet(names ...string) map[string]struct{} {
	set := make(map[string]struct{}, len(names))
	for _, n := range names {
		set[n] = struct{}{}
	}
	return set
}

func filterToString(t *testing.T, payload string, want map[string]struct{}) string {
	t.Helper()
	var buf bytes.Buffer
	if err := filterFamilies(&buf, strings.NewReader(payload), want); err != nil {
		t.Fatalf("filterFamilies: %v", err)
	}
	return buf.String()
}

func TestFilterFamiliesKeepsOnlyWantedFamilies(t *testing.T) {
	tests := []struct {
		name    string
		payload string
		want    map[string]struct{}
		expect  string
	}{
		{
			name: "drops unwanted family with its headers",
			payload: "# HELP a_metric help.\n# TYPE a_metric gauge\na_metric 1\n" +
				"# HELP b_metric help.\n# TYPE b_metric gauge\nb_metric 2\n",
			want:   nameSet("a_metric"),
			expect: "# HELP a_metric help.\n# TYPE a_metric gauge\na_metric 1\n",
		},
		{
			name: "keeps histogram series of a wanted family",
			payload: "# HELP h help.\n# TYPE h histogram\n" +
				"h_bucket{le=\"1\"} 1\nh_sum 2\nh_count 3\n" +
				"# HELP other help.\n# TYPE other counter\nother 9\n",
			want: nameSet("h"),
			expect: "# HELP h help.\n# TYPE h histogram\n" +
				"h_bucket{le=\"1\"} 1\nh_sum 2\nh_count 3\n",
		},
		{
			name:    "matches samples by name when headers are absent",
			payload: "a_metric 1\nb_metric 2\na_metric{x=\"y\"} 3\n",
			want:    nameSet("a_metric"),
			expect:  "a_metric 1\na_metric{x=\"y\"} 3\n",
		},
		{
			name:    "matches histogram series by name when headers are absent",
			payload: "h_bucket{le=\"1\"} 1\nh_sum 2\nother_sum 5\n",
			want:    nameSet("h"),
			expect:  "h_bucket{le=\"1\"} 1\nh_sum 2\n",
		},
		{
			name: "a free-form comment does not end the current family",
			payload: "# HELP a help.\n# TYPE a gauge\n# a note\na 1\n" +
				"# HELP b help.\n# TYPE b gauge\nb 2\n",
			want:   nameSet("a"),
			expect: "# HELP a help.\n# TYPE a gauge\na 1\n",
		},
		{
			name:    "a blank line ends the current family",
			payload: "# HELP a help.\n# TYPE a gauge\na 1\n\nb 2\n",
			want:    nameSet("a"),
			expect:  "# HELP a help.\n# TYPE a gauge\na 1\n",
		},
		{
			name:    "a label value containing a brace does not confuse name scanning",
			payload: "a{note=\"{b_metric}\"} 1\nb_metric 2\n",
			want:    nameSet("a"),
			expect:  "a{note=\"{b_metric}\"} 1\n",
		},
		{
			name:    "a family whose name prefixes another is not confused with it",
			payload: "a_metric 1\na_metric_extra 2\n",
			want:    nameSet("a_metric"),
			expect:  "a_metric 1\n",
		},
		{
			name: "a bare sample after a wanted family is judged on its own name",
			payload: "# HELP a help.\n# TYPE a gauge\na 1\n" +
				"b 2\na_bucket 3\n",
			want:   nameSet("a"),
			expect: "# HELP a help.\n# TYPE a gauge\na 1\na_bucket 3\n",
		},
		{
			name: "a bare sample after an unwanted family is judged on its own name",
			payload: "# HELP z help.\n# TYPE z gauge\nz 1\n" +
				"a 2\n",
			want:   nameSet("a"),
			expect: "a 2\n",
		},
		{
			name: "a family named like a series of the previous one is judged on its own name",
			payload: "# HELP a help.\n# TYPE a gauge\na 1\n" +
				"# HELP a_count help.\n# TYPE a_count gauge\na_count 2\n",
			want:   nameSet("a_count"),
			expect: "# HELP a_count help.\n# TYPE a_count gauge\na_count 2\n",
		},
		{
			name:    "final line without a trailing newline is kept",
			payload: "a_metric 1",
			want:    nameSet("a_metric"),
			expect:  "a_metric 1",
		},
		{
			name:    "empty want set drops everything",
			payload: "a_metric 1\nb_metric 2\n",
			want:    nameSet(),
			expect:  "",
		},
		{
			name:    "empty payload yields empty output",
			payload: "",
			want:    nameSet("a_metric"),
			expect:  "",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := filterToString(t, tc.payload, tc.want); got != tc.expect {
				t.Errorf("filtered output mismatch\n got: %q\nwant: %q", got, tc.expect)
			}
		})
	}
}

// TestFilterFamiliesHandlesLinesLongerThanReadBuffer covers the reassembly path:
// a LoRA info metric can carry hundreds of adapter names on a single line.
func TestFilterFamiliesHandlesLinesLongerThanReadBuffer(t *testing.T) {
	adapters := strings.Repeat("adapter-with-a-long-name,", (defaultReadBufferSize/25)+64)
	long := fmt.Sprintf("lora{running_lora_adapters=%q} 1\n", adapters)
	payload := "dropped 0\n" + long + "also_dropped 0\n"

	got := filterToString(t, payload, nameSet("lora"))
	if got != long {
		t.Errorf("long line not reassembled intact: got %d bytes, want %d", len(got), len(long))
	}
}

// TestFilteredParseMatchesFullParse is the contract the optimization rests on:
// for every family an extractor reads, parsing the filtered scrape must yield
// exactly what parsing the whole scrape yields.
func TestFilteredParseMatchesFullParse(t *testing.T) {
	payload := vllmScrapeFixture()
	want := nameSet(vllmConsumedFamilies...)

	fullParser := expfmt.NewTextParser(model.LegacyValidation)
	full, err := fullParser.TextToMetricFamilies(strings.NewReader(payload))
	if err != nil {
		t.Fatalf("full parse: %v", err)
	}

	var buf bytes.Buffer
	if err := filterFamilies(&buf, strings.NewReader(payload), want); err != nil {
		t.Fatalf("filterFamilies: %v", err)
	}
	filteredParser := expfmt.NewTextParser(model.LegacyValidation)
	filtered, err := filteredParser.TextToMetricFamilies(&buf)
	if err != nil {
		t.Fatalf("filtered parse: %v", err)
	}

	if len(filtered) != len(vllmConsumedFamilies) {
		t.Fatalf("filtered parse produced %d families, want %d", len(filtered), len(vllmConsumedFamilies))
	}
	for _, name := range vllmConsumedFamilies {
		expected, ok := full[name]
		if !ok {
			t.Fatalf("fixture does not expose %s", name)
		}
		got, ok := filtered[name]
		if !ok {
			t.Fatalf("filtered parse dropped %s", name)
		}
		if expected.String() != got.String() {
			t.Errorf("%s differs after filtering\n full: %s\nfiltered: %s", name, expected, got)
		}
	}
}

func TestFamilySelectorUnionsDeclarations(t *testing.T) {
	var s familySelector
	if s.wanted() != nil {
		t.Fatal("a selector with no declarations must keep everything")
	}

	s.observe(plainStub{})
	if s.wanted() != nil {
		t.Fatal("a plugin that declares nothing must keep everything")
	}

	s.observe(namerStub{names: []string{"a", "b"}})
	s.observe(namerStub{names: []string{"b", "c"}})

	got := s.wanted()
	if len(got) != 3 {
		t.Fatalf("union has %d names, want 3: %v", len(got), got)
	}
	for _, name := range []string{"a", "b", "c"} {
		if _, ok := got[name]; !ok {
			t.Errorf("union is missing %q", name)
		}
	}
}

func TestFamilySelectorIgnoresEmptyNames(t *testing.T) {
	var s familySelector
	s.observe(namerStub{names: []string{"", "a"}})

	got := s.wanted()
	if len(got) != 1 {
		t.Fatalf("selector kept %d names, want 1: %v", len(got), got)
	}
	if _, ok := got["a"]; !ok {
		t.Error("selector dropped the non-empty name")
	}
}

// TestFamilySelectorPublishesSnapshots guards the read path: scrapes read the
// selector concurrently while extractors may still be binding.
func TestFamilySelectorPublishesSnapshots(t *testing.T) {
	var s familySelector
	s.observe(namerStub{names: []string{"a"}})
	first := s.wanted()

	s.observe(namerStub{names: []string{"b"}})

	if _, leaked := first["b"]; leaked {
		t.Error("a previously published set was mutated in place")
	}
}

func TestFilteringParserFallsBackWithoutDeclarations(t *testing.T) {
	parser := newFamilyFilteringParser()
	payload := vllmScrapeFixture()

	families, err := parser.parse(strings.NewReader(payload))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(families) <= len(vllmConsumedFamilies) {
		t.Fatalf("undeclared parser kept %d families, expected the whole scrape", len(families))
	}

	parser.observeExtractor(namerStub{names: vllmConsumedFamilies})

	families, err = parser.parse(strings.NewReader(payload))
	if err != nil {
		t.Fatalf("parse after declaration: %v", err)
	}
	if len(families) != len(vllmConsumedFamilies) {
		t.Fatalf("declared parser kept %d families, want %d", len(families), len(vllmConsumedFamilies))
	}
}

// TestFilteringParserIsConcurrencySafe exercises the pooled buffers: one data
// source serves every endpoint it is configured for, and their collectors poll
// on independent goroutines.
func TestFilteringParserIsConcurrencySafe(t *testing.T) {
	parser := newFamilyFilteringParser()
	parser.observeExtractor(namerStub{names: vllmConsumedFamilies})
	payload := vllmScrapeFixture()

	var wg sync.WaitGroup
	errs := make(chan error, 64)
	for range 64 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 20 {
				families, err := parser.parse(strings.NewReader(payload))
				if err != nil {
					errs <- err
					return
				}
				if len(families) != len(vllmConsumedFamilies) {
					errs <- fmt.Errorf("got %d families, want %d", len(families), len(vllmConsumedFamilies))
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
}
