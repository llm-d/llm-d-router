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
	"strings"
	"testing"
)

// BenchmarkParseScrape contrasts the cost of one scrape parsed whole against the
// same scrape filtered to the families an extractor reads. Every endpoint is
// scraped on its own ticker, so this cost scales with fleet size and is paid
// whether or not the endpoint picker is serving traffic.
func BenchmarkParseScrape(b *testing.B) {
	payload := vllmScrapeFixture()
	want := nameSet(vllmConsumedFamilies...)

	b.Run("unfiltered", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(len(payload)))
		for b.Loop() {
			families, err := parseMetrics(strings.NewReader(payload))
			if err != nil {
				b.Fatal(err)
			}
			if len(families) == 0 {
				b.Fatal("no families parsed")
			}
		}
	})

	b.Run("filtered", func(b *testing.B) {
		var buf bytes.Buffer
		b.ReportAllocs()
		b.SetBytes(int64(len(payload)))
		for b.Loop() {
			buf.Reset()
			if err := filterFamilies(&buf, strings.NewReader(payload), want); err != nil {
				b.Fatal(err)
			}
			families, err := parseMetrics(&buf)
			if err != nil {
				b.Fatal(err)
			}
			if len(families) != len(vllmConsumedFamilies) {
				b.Fatalf("got %d families, want %d", len(families), len(vllmConsumedFamilies))
			}
		}
	})
}

// BenchmarkFilterFamilies isolates the filter from the parse that follows it.
func BenchmarkFilterFamilies(b *testing.B) {
	payload := vllmScrapeFixture()
	want := nameSet(vllmConsumedFamilies...)
	var buf bytes.Buffer

	b.ReportAllocs()
	b.SetBytes(int64(len(payload)))
	for b.Loop() {
		buf.Reset()
		if err := filterFamilies(&buf, strings.NewReader(payload), want); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkFilteringParser measures the parser as the data source uses it,
// including the pooled buffers.
func BenchmarkFilteringParser(b *testing.B) {
	payload := vllmScrapeFixture()

	b.Run("undeclared", func(b *testing.B) {
		parser := newFamilyFilteringParser()
		b.ReportAllocs()
		b.SetBytes(int64(len(payload)))
		for b.Loop() {
			if _, err := parser.parse(strings.NewReader(payload)); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("declared", func(b *testing.B) {
		parser := newFamilyFilteringParser()
		parser.observeExtractor(namerStub{names: vllmConsumedFamilies})
		b.ReportAllocs()
		b.SetBytes(int64(len(payload)))
		for b.Loop() {
			if _, err := parser.parse(strings.NewReader(payload)); err != nil {
				b.Fatal(err)
			}
		}
	})
}
