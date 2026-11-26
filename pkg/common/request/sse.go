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

package request

import (
	"bytes"
	"strings"
)

// sseDataField is the SSE data field name. SSEDataPrefix is the wire form
// servers emit, with the one optional space the spec allows after the colon.
var sseDataField = []byte(strings.TrimSuffix(SSEDataPrefix, " "))

// SSEDataPayload cuts the data-field prefix off one SSE line, reporting the
// payload. The spec names the field "data:" followed by one optional space, so
// both the wire form and a spaceless "data:" read; any other line, including
// another field whose name starts with data, reports false. The payload may be
// empty: a data line with nothing after the prefix is still a data line. The
// line carries no line ending; a caller that splits a stream strips it.
func SSEDataPayload(line []byte) ([]byte, bool) {
	payload, ok := bytes.CutPrefix(line, sseDataField)
	if !ok {
		return nil, false
	}
	return bytes.TrimPrefix(payload, []byte{' '}), true
}

// SSEIsDone reports whether a data payload is the [DONE] terminator that ends
// an OpenAI-compatible stream, tolerating the padding a server may emit.
func SSEIsDone(payload []byte) bool {
	return bytes.Equal(bytes.TrimSpace(payload), []byte(SSEDoneMarker))
}
