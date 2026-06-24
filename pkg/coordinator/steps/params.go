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

package steps

import (
	"encoding/json"
	"fmt"
	"math"
	"time"
)

// Parameter key constants for step configuration maps.
const (
	ParamKVConnector = "kv_connector"
	ParamECConnector = "ec_connector"
)

const ModalityImage = "image"

// paramInt reads an integer step parameter. The config decoder may hand a number
// back as int, int64, float64, or json.Number depending on its source and YAML
// representation (a float-formatted literal such as 8192.0 decodes as float64),
// so all are accepted; a non-integral float, an out-of-range value, or a
// non-numeric value is an error rather than a silent truncation or wrap. ok is
// false when the key is absent, leaving the caller's default in place.
func paramInt(params map[string]any, key string) (value int, ok bool, err error) {
	var i64 int64
	switch v := params[key].(type) {
	case nil:
		return 0, false, nil
	case int:
		// An int is by definition in range; return it directly.
		return v, true, nil
	case int64:
		i64 = v
	case float64:
		if v != math.Trunc(v) {
			return 0, false, fmt.Errorf("%s: expected integer, got %v", key, v)
		}
		// float64 cannot represent MaxInt64 exactly (it rounds to 2^63), so
		// reject at that bound to keep the int64(v) conversion well-defined.
		if v < math.MinInt64 || v >= math.MaxInt64 {
			return 0, false, fmt.Errorf("%s: out of range, got %v", key, v)
		}
		i64 = int64(v)
	case json.Number:
		n, convErr := v.Int64()
		if convErr != nil {
			return 0, false, fmt.Errorf("%s: %w", key, convErr)
		}
		i64 = n
	default:
		return 0, false, fmt.Errorf("%s: expected number, got %T", key, v)
	}
	if i64 < math.MinInt || i64 > math.MaxInt {
		return 0, false, fmt.Errorf("%s: out of range for int, got %d", key, i64)
	}
	return int(i64), true, nil
}

// paramDuration reads a duration step parameter from a Go duration string (e.g.
// "30s"). An unparsable string is an error rather than a silent fallback, so a
// malformed value such as "30" (no unit) fails config load instead of running
// the default. ok is false when the key is absent.
func paramDuration(params map[string]any, key string) (value time.Duration, ok bool, err error) {
	switch v := params[key].(type) {
	case nil:
		return 0, false, nil
	case string:
		d, parseErr := time.ParseDuration(v)
		if parseErr != nil {
			return 0, false, fmt.Errorf("%s: %w", key, parseErr)
		}
		return d, true, nil
	default:
		return 0, false, fmt.Errorf("%s: expected duration string, got %T", key, v)
	}
}
