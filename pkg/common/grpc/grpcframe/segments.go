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

package grpcframe

import (
	"io"
	"net/http"
)

// SetSegmentedBody installs the concatenation of segments as the body of req
// and returns its length. It does not join the segments, so a large segment
// can be shared between requests. It sets ContentLength, and GetBody so the
// HTTP/2 transport can replay the body after a GOAWAY. http.NewRequest does
// that only for a bytes.Reader body. Empty segments add nothing.
func SetSegmentedBody(req *http.Request, segments [][]byte) int64 {
	total := SegmentsLen(segments)
	if total == 0 {
		req.Body = http.NoBody
		req.GetBody = func() (io.ReadCloser, error) { return http.NoBody, nil }
		req.ContentLength = 0
		return 0
	}
	req.Body = NewSegmentsReader(segments)
	req.GetBody = func() (io.ReadCloser, error) { return NewSegmentsReader(segments), nil }
	req.ContentLength = total
	return total
}

// SegmentsLen returns the total length of segments.
func SegmentsLen(segments [][]byte) int64 {
	var total int64
	for _, s := range segments {
		total += int64(len(s))
	}
	return total
}

// NewSegmentsReader returns a reader over the concatenation of segments. It
// does not copy the segments. Close has no effect.
func NewSegmentsReader(segments [][]byte) io.ReadCloser {
	return &segmentsReader{segs: segments}
}

type segmentsReader struct {
	segs [][]byte
	idx  int
	off  int
}

func (r *segmentsReader) Read(p []byte) (int, error) {
	total := 0
	for len(p) > 0 {
		for r.idx < len(r.segs) && r.off >= len(r.segs[r.idx]) {
			r.idx++
			r.off = 0
		}
		if r.idx >= len(r.segs) {
			if total > 0 {
				return total, nil
			}
			return 0, io.EOF
		}
		n := copy(p, r.segs[r.idx][r.off:])
		r.off += n
		total += n
		p = p[n:]
	}
	return total, nil
}

func (r *segmentsReader) Close() error { return nil }
