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

// Package httplog provides helpers for logging HTTP headers safely and
// consistently across services.
package httplog

import (
	"net/http"
	"strings"
)

const redactedValue = "[REDACTED]"

// maxValueLen caps a logged header value. Longer values are truncated so a
// verbose header (e.g. X-Envoy-Peer-Metadata, a base64 proto blob) does not
// dominate the log line.
const maxValueLen = 256

var sensitiveHeaders = map[string]struct{}{
	"Authorization":       {},
	"Proxy-Authorization": {},
	"Cookie":              {},
	"Set-Cookie":          {},
	"X-Api-Key":           {},
	"X-Auth-Token":        {},
}

func isSensitiveHeader(name string) bool {
	_, ok := sensitiveHeaders[http.CanonicalHeaderKey(name)]
	return ok
}

func truncate(v string) string {
	if len(v) > maxValueLen {
		return v[:maxValueLen] + "...[truncated]"
	}
	return v
}

// RedactedHeaders returns a flattened copy of h with values of sensitive
// headers replaced by a redaction sentinel. It accepts either http.Header
// (map[string][]string) or map[string]string and always returns the flat
// form. Keys are normalized to lowercase so log output is consistent
// regardless of how the input was canonicalized. A multi-valued header keeps
// only its first value. A header with no value (an empty slice or an empty
// string) retains its key with an empty-string value, so both input forms
// behave the same. Values longer than maxValueLen are truncated.
func RedactedHeaders[V string | []string](h map[string]V) map[string]string {
	out := make(map[string]string, len(h))
	for k, v := range h {
		lower := strings.ToLower(k)
		if isSensitiveHeader(k) {
			out[lower] = redactedValue
			continue
		}
		switch val := any(v).(type) {
		case string:
			out[lower] = truncate(val)
		case []string:
			first := ""
			if len(val) > 0 {
				first = val[0]
			}
			out[lower] = truncate(first)
		}
	}
	return out
}
