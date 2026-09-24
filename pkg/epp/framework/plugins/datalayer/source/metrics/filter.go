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
	"bufio"
	"bytes"
	"io"
	"sync"
	"sync/atomic"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// FamilyNamer is the optional contract an extractor implements to declare the
// Prometheus metric families it reads. The metrics data source unions the
// declarations of its bound extractors and discards every other family before
// parsing. An extractor that does not implement it forces a full parse, so the
// interface is safe to adopt incrementally.
type FamilyNamer interface {
	MetricNames() []string
}

// suffixes a scrape may append to a family name for histogram and summary series.
var seriesSuffixes = [...]string{"_bucket", "_sum", "_count"}

// familySelector holds the union of family names the bound extractors declared.
// Extractors bind during configuration and the set is read on every scrape, so
// the resolved set is published through an atomic pointer rather than a mutex:
// scrapes never contend with each other.
type familySelector struct {
	mu    sync.Mutex
	names map[string]struct{}
	// resolved is nil until at least one extractor declares a family. A nil
	// value means "keep everything", which is the behaviour of a data source
	// whose extractors do not implement FamilyNamer.
	resolved atomic.Pointer[map[string]struct{}]
}

// observe records the families ext declares, if it declares any.
func (s *familySelector) observe(ext fwkplugin.Plugin) {
	namer, ok := ext.(FamilyNamer)
	if !ok {
		return
	}
	names := namer.MetricNames()
	if len(names) == 0 {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.names == nil {
		s.names = make(map[string]struct{}, len(names))
	}
	for _, name := range names {
		if name != "" {
			s.names[name] = struct{}{}
		}
	}
	// Publish a copy so readers never observe a map being written.
	published := make(map[string]struct{}, len(s.names))
	for name := range s.names {
		published[name] = struct{}{}
	}
	s.resolved.Store(&published)
}

// wanted returns the published set, or nil when no extractor declared anything.
func (s *familySelector) wanted() map[string]struct{} {
	if p := s.resolved.Load(); p != nil {
		return *p
	}
	return nil
}

// bufferPool recycles the scratch buffers filterFamilies writes into. A scrape
// keeps only the families an extractor reads, so the buffer stays small and
// steady-state scrapes allocate nothing.
var bufferPool = sync.Pool{
	New: func() any { return new(bytes.Buffer) },
}

// lineReaderPool recycles the readers used to split a scrape into lines.
var lineReaderPool = sync.Pool{
	New: func() any { return bufio.NewReaderSize(nil, defaultReadBufferSize) },
}

// defaultReadBufferSize is the initial line-splitting buffer. Lines longer than
// this are handled by accumulating fragments, so the value bounds memory rather
// than the accepted line length.
const defaultReadBufferSize = 16 << 10

// maxFamilyNameSize preallocates the buffer holding the family name in force.
// Longer names still work; the buffer grows.
const maxFamilyNameSize = 128

// filterFamilies copies the text-exposition lines belonging to want from src
// into dst, dropping everything else.
//
// A scrape groups a family's "# HELP"/"# TYPE" headers immediately before its
// samples, so the decision made for a header usually carries to the samples
// that follow it. Carrying it blindly would be wrong, because headers are
// optional and a scrape may start a new family with a bare sample line, so a
// sample is only granted the standing decision when its own name belongs to
// the family that header named. Any other sample is matched on its own name.
func filterFamilies(dst *bytes.Buffer, src io.Reader, want map[string]struct{}) error {
	reader, _ := lineReaderPool.Get().(*bufio.Reader)
	reader.Reset(src)
	defer func() {
		reader.Reset(nil)
		lineReaderPool.Put(reader)
	}()

	var state filterState
	state.family = make([]byte, 0, maxFamilyNameSize)
	for {
		line, err := readLine(reader)
		if len(line) > 0 && state.keepLine(line, want) {
			dst.Write(line)
		}
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
	}
}

// filterState carries the decision made for the family a header named, so the
// samples that follow it are admitted without repeating the lookup.
type filterState struct {
	// family is the name from the most recent header, empty when no header is
	// in force. It is copied because the line it came from is reused.
	family []byte
	// keep is the decision made for family.
	keep bool
}

// keepLine reports whether line survives filtering, updating the state.
func (s *filterState) keepLine(line []byte, want map[string]struct{}) bool {
	trimmed := trimLeadingSpace(line)
	if isBlank(trimmed) {
		s.family = s.family[:0]
		return false
	}

	if trimmed[0] == '#' {
		name, ok := headerFamilyName(trimmed[1:])
		if !ok {
			// A free-form comment belongs to no family and ends none.
			return false
		}
		s.family = append(s.family[:0], name...)
		s.keep = matches(name, want)
		return s.keep
	}

	name := sampleFamilyName(trimmed)
	if len(s.family) > 0 && inFamily(name, s.family) {
		return s.keep
	}
	// A sample that does not belong to the named family starts a new one.
	s.family = s.family[:0]
	return matches(name, want)
}

// inFamily reports whether a sample named name is a series of family.
func inFamily(name, family []byte) bool {
	if !bytes.HasPrefix(name, family) {
		return false
	}
	rest := name[len(family):]
	if len(rest) == 0 {
		return true
	}
	for _, suffix := range seriesSuffixes {
		if string(rest) == suffix {
			return true
		}
	}
	return false
}

// isBlank reports whether a line carries no content.
func isBlank(line []byte) bool {
	for _, c := range line {
		if c != ' ' && c != '\t' && c != '\n' && c != '\r' {
			return false
		}
	}
	return true
}

// headerFamilyName extracts the family name from a "# HELP name ..." or
// "# TYPE name ..." line, with the leading '#' already removed. It reports
// false for any other comment.
func headerFamilyName(comment []byte) ([]byte, bool) {
	rest := trimLeadingSpace(comment)
	keyword := rest[:fieldEnd(rest)]
	if !bytes.Equal(keyword, []byte("HELP")) && !bytes.Equal(keyword, []byte("TYPE")) {
		return nil, false
	}
	rest = trimLeadingSpace(rest[len(keyword):])
	name := rest[:fieldEnd(rest)]
	if len(name) == 0 {
		return nil, false
	}
	return name, true
}

// sampleFamilyName returns the metric name of a sample line, which ends at the
// label list or at the whitespace before the value.
func sampleFamilyName(line []byte) []byte {
	for i := range line {
		switch line[i] {
		case '{', ' ', '\t', '\n', '\r':
			return line[:i]
		}
	}
	return line
}

// matches reports whether name, or the family it is a series of, is wanted.
func matches(name []byte, want map[string]struct{}) bool {
	if len(name) == 0 {
		return false
	}
	// A []byte key in a map lookup does not allocate.
	if _, ok := want[string(name)]; ok {
		return true
	}
	for _, suffix := range seriesSuffixes {
		base, found := bytes.CutSuffix(name, []byte(suffix))
		if !found {
			continue
		}
		if _, ok := want[string(base)]; ok {
			return true
		}
	}
	return false
}

func trimLeadingSpace(b []byte) []byte {
	for len(b) > 0 && (b[0] == ' ' || b[0] == '\t') {
		b = b[1:]
	}
	return b
}

// fieldEnd returns the index that ends the leading whitespace-delimited field.
func fieldEnd(b []byte) int {
	for i := range b {
		switch b[i] {
		case ' ', '\t', '\n', '\r':
			return i
		}
	}
	return len(b)
}

// readLine returns one line including its terminator. Lines longer than the
// reader's buffer are reassembled, so no scrape is rejected for line length.
// The returned slice is only valid until the next call.
func readLine(r *bufio.Reader) ([]byte, error) {
	line, err := r.ReadSlice('\n')
	if err != bufio.ErrBufferFull {
		return line, err
	}
	// Rare path: the line exceeds the buffer, so copy it out and keep reading.
	joined := make([]byte, len(line))
	copy(joined, line)
	for {
		line, err = r.ReadSlice('\n')
		joined = append(joined, line...)
		if err != bufio.ErrBufferFull {
			return joined, err
		}
	}
}
