/*
Copyright 2025 The Kubernetes Authors.

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

import "strings"

// GetHeader returns the value for key from headers, with case-insensitive lookup.
func GetHeader(headers map[string]string, key string) string {
	if v, ok := headers[key]; ok {
		return v
	}
	lower := strings.ToLower(key)
	for k, v := range headers {
		if strings.ToLower(k) == lower {
			return v
		}
	}
	return ""
}

// GetRequestPath extracts the request path from headers with fallback priority.
// Query parameters are stripped because the path is used only for parser routing.
func GetRequestPath(headers map[string]string) string {
	if path := headers[":path"]; path != "" {
		return stripQuery(path)
	}
	if path := headers["x-original-path"]; path != "" {
		return stripQuery(path)
	}
	if path := headers["x-forwarded-path"]; path != "" {
		return stripQuery(path)
	}
	// Default to completions API for backward compatibility with existing clients and integration tests
	return "/v1/completions"
}

func stripQuery(path string) string {
	clean, _, _ := strings.Cut(path, "?")
	return clean
}

// MatchPathSuffix checks if the path matches the suffix.
func MatchPathSuffix(path, suffix string) bool {
	path = strings.TrimSuffix(strings.TrimSpace(path), "/")
	suffix = strings.Trim(strings.TrimSpace(suffix), "/")
	return strings.HasSuffix(path, suffix)
}
