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
	"errors"
	"net/http"
)

// ClassifyOptions injects the caller's error sentinels so ClassifyErrorCode
// can map pipeline-domain errors to error_code labels without importing the
// pipeline package (which would invert the coordinator dependency direction).
type ClassifyOptions struct {
	// BadRequest is the sentinel error that means "invalid client input". Any
	// err with errors.Is(err, BadRequest) == true classifies to
	// ErrorCodeBadRequest.
	BadRequest error
	// IsUpstream returns (status, true) when err carries an upstream HTTP
	// status; the returned status routes into the ErrorCodeUpstream4xx /
	// ErrorCodeUpstream5xx buckets. Returning ok=false means "not an
	// upstream error"; classification falls through.
	IsUpstream func(err error) (status int, ok bool)
}

// ClassifyErrorCode maps a pipeline error to the error_code label emitted on
// request_error_total and step_errors_total. Both metric families use the
// same label vocabulary, so the two call sites share this function to stay
// consistent — a new bucket or a shifted status band is one edit, both
// metrics pick it up.
func ClassifyErrorCode(err error, opts ClassifyOptions) string {
	if opts.BadRequest != nil && errors.Is(err, opts.BadRequest) {
		return ErrorCodeBadRequest
	}
	if opts.IsUpstream != nil {
		if status, ok := opts.IsUpstream(err); ok {
			switch {
			case status == 0:
				return ErrorCodeUpstreamTransport
			case status >= http.StatusBadRequest && status < http.StatusInternalServerError:
				return ErrorCodeUpstream4xx
			case status >= http.StatusInternalServerError:
				return ErrorCodeUpstream5xx
			}
		}
	}
	return ErrorCodeInternal
}
