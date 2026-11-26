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
	"bytes"
	"io"

	"github.com/cespare/xxhash/v2"
	lru "github.com/hashicorp/golang-lru/v2"
	dto "github.com/prometheus/client_model/go"
)

// heavyLabels is the label count from which a sample makes its family worth caching. The text
// parser rebuilds the set of a sample's labels read so far at each label, so a sample costs the
// square of its label count; vllm:cache_config_info carries about 40.
const heavyLabels = 16

// The cache bounds keep one source's cached families to about 30 MiB for any server output.
// vllm:cache_config_info is about 1 KiB and 30 to 40 labels, and is cached once per distinct
// content, usually once for all endpoints of a pool.
const (
	familyCacheSize   = 1024
	maxCachedRunBytes = 8 << 10
	maxCachedLabels   = 128
)

var labelAssign = []byte(`="`)

// familyCache holds parsed families keyed by the exact text of their lines. Callers share the
// cached families and must not modify them.
type familyCache struct {
	entries *lru.Cache[uint64, cachedFamily]
}

type cachedFamily struct {
	raw    []byte
	family *dto.MetricFamily
}

func newFamilyCache() *familyCache {
	entries, err := lru.New[uint64, cachedFamily](familyCacheSize)
	if err != nil {
		panic(err) // only for a non-positive size
	}
	return &familyCache{entries: entries}
}

// parse parses kept, a text-format response, taking the families of the cacheable runs from the
// cache. When a run does not parse alone, or the rest of kept fails or holds a family of a run,
// it parses kept whole so the result and error match a plain parse.
func (c *familyCache) parse(kept []byte, runs []familyRun) (PrometheusMetricMap, error) {
	if len(runs) == 0 {
		return parseMetrics(bytes.NewReader(kept))
	}
	cached := make(map[string]*dto.MetricFamily, len(runs))
	rest := make([]io.Reader, 0, len(runs)+1)
	prev := 0
	for _, run := range runs {
		family := c.get(kept[run.start:run.end], run.name)
		if family == nil || cached[run.name] != nil {
			return parseMetrics(bytes.NewReader(kept))
		}
		cached[run.name] = family
		rest = append(rest, bytes.NewReader(kept[prev:run.start]))
		prev = run.end
	}
	rest = append(rest, bytes.NewReader(kept[prev:]))
	families, err := parseMetrics(io.MultiReader(rest...))
	if err != nil {
		return parseMetrics(bytes.NewReader(kept))
	}
	for name, family := range cached {
		if _, ok := families[name]; ok {
			return parseMetrics(bytes.NewReader(kept))
		}
		families[name] = family
	}
	return families, nil
}

// get returns the family parsed from raw, the lines of one family, or nil when raw does not parse
// to that single family.
func (c *familyCache) get(raw []byte, name string) *dto.MetricFamily {
	key := xxhash.Sum64(raw)
	if entry, ok := c.entries.Get(key); ok && bytes.Equal(entry.raw, raw) {
		return entry.family
	}
	families, err := parseMetrics(bytes.NewReader(raw))
	family := families[name]
	if err != nil || len(families) != 1 || family == nil {
		return nil
	}
	c.entries.Add(key, cachedFamily{raw: bytes.Clone(raw), family: family})
	return family
}

// familyRun is the byte range of one family's lines in the kept response.
type familyRun struct {
	name       string
	start, end int
}

// familyRuns splits kept lines into runs of one family name and records the cacheable runs: a
// gauge, counter or untyped family with one TYPE line, at most one HELP line, a sample of
// heavyLabels or more labels, no sampleSuffixes suffix, and a name that starts no other run.
// Comments and blank lines stay in the current run.
type familyRuns struct {
	name      []byte
	start     int
	helps     int
	types     int
	typed     bool
	labels    int
	heavy     bool
	names     []uint64 // the name hash of each run
	cacheable []familyRun
}

func (r *familyRuns) observe(line []byte, offset int) {
	kind, name, rest := lineHead(line)
	if kind == lineOther {
		return
	}
	if !bytes.Equal(name, r.name) {
		r.close(offset)
		*r = familyRuns{
			name:      append(r.name[:0], name...),
			start:     offset,
			names:     append(r.names, xxhash.Sum64(name)),
			cacheable: r.cacheable,
		}
	}
	switch kind {
	case lineHelp:
		r.helps++
	case lineType:
		r.types++
		switch string(bytes.TrimSpace(rest)) {
		case "gauge", "counter", "untyped":
			r.typed = true
		}
	case lineSample:
		n := bytes.Count(rest, labelAssign)
		r.labels += n
		r.heavy = r.heavy || n >= heavyLabels
	}
}

// close ends the current run at offset end of the kept response.
func (r *familyRuns) close(end int) {
	if r.types == 1 && r.helps <= 1 && r.typed && r.heavy &&
		r.labels <= maxCachedLabels && end-r.start <= maxCachedRunBytes && !hasSampleSuffix(r.name) {
		r.cacheable = append(r.cacheable, familyRun{name: string(r.name), start: r.start, end: end})
	}
}

// finish ends the last run at offset end and returns the cacheable runs. A name that starts a
// second run, such as a later TYPE line, must reach the parser together with its other lines.
func (r *familyRuns) finish(end int) []familyRun {
	r.close(end)
	runs := r.cacheable[:0]
	for _, run := range r.cacheable {
		hash := xxhash.Sum64String(run.name)
		count := 0
		for _, h := range r.names {
			if h == hash {
				count++
			}
		}
		if count == 1 {
			runs = append(runs, run)
		}
	}
	return runs
}

// hasSampleSuffix reports whether name could be a histogram or summary sample, which the text
// parser files under the base family.
func hasSampleSuffix(name []byte) bool {
	for _, suffix := range sampleSuffixes {
		if bytes.HasSuffix(name, suffix) {
			return true
		}
	}
	return false
}

type lineKind int

const (
	lineOther lineKind = iota
	lineHelp
	lineType
	lineSample
)

// lineHead returns the kind of a text-format line, the metric name it carries, and the text after
// the name. Free-form comments, blank lines and lines without a name are lineOther.
func lineHead(line []byte) (lineKind, []byte, []byte) {
	line = bytes.TrimLeft(line, " \t")
	kind := lineSample
	if bytes.HasPrefix(line, []byte("#")) {
		comment := bytes.TrimLeft(line[1:], " \t")
		end := bytes.IndexAny(comment, " \t")
		if end < 0 {
			return lineOther, nil, nil
		}
		switch {
		case bytes.Equal(comment[:end], []byte("HELP")):
			kind = lineHelp
		case bytes.Equal(comment[:end], []byte("TYPE")):
			kind = lineType
		default:
			return lineOther, nil, nil
		}
		line = bytes.TrimLeft(comment[end:], " \t")
	}
	end := bytes.IndexAny(line, "{ \t\r\n")
	if end < 0 {
		end = len(line)
	}
	if end == 0 {
		return lineOther, nil, nil
	}
	return kind, line[:end], line[end:]
}
