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
	"io"

	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// familyFilteringParser discards the families no bound extractor declared
// before handing the scrape to expfmt. A model server exposes far more families
// than the endpoint picker reads, and parsing the remainder costs CPU and
// garbage on every scrape of every endpoint.
type familyFilteringParser struct {
	selector familySelector
}

func newFamilyFilteringParser() *familyFilteringParser {
	return &familyFilteringParser{}
}

// observeExtractor feeds the parser's selector from the data source's extractor
// hook. Until an extractor declares a family the parser keeps the whole scrape,
// so a source bound to extractors that do not implement FamilyNamer behaves as
// before.
func (p *familyFilteringParser) observeExtractor(ext fwkplugin.Plugin) {
	p.selector.observe(ext)
}

func (p *familyFilteringParser) parse(data io.Reader) (PrometheusMetricMap, error) {
	want := p.selector.wanted()
	if want == nil {
		return parseMetrics(data)
	}

	buf, _ := bufferPool.Get().(*bytes.Buffer)
	buf.Reset()
	defer bufferPool.Put(buf)

	if err := filterFamilies(buf, data, want); err != nil {
		return nil, err
	}
	return parseMetrics(buf)
}

func parseMetrics(data io.Reader) (PrometheusMetricMap, error) {
	parser := expfmt.NewTextParser(model.LegacyValidation)
	return parser.TextToMetricFamilies(data)
}
