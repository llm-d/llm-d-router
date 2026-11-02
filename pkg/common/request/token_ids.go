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
	"fmt"
	"math"
)

// ParseTokenIDs converts a JSON-decoded array of numbers into token IDs.
// Each element is read by NonNegativeInt and must also be at most MaxUint32;
// the error names the offending index.
func ParseTokenIDs(values []any) ([]uint32, error) {
	ids := make([]uint32, len(values))
	for i, v := range values {
		n, err := NonNegativeInt(v)
		if err != nil {
			return nil, fmt.Errorf("token id at index %d: %w", i, err)
		}
		// int64 keeps the comparison compilable where int is 32 bits.
		if int64(n) > math.MaxUint32 {
			return nil, fmt.Errorf("token id at index %d is above the uint32 range: %d", i, n)
		}
		ids[i] = uint32(n) //#nosec G115 -- guarded by [0, MaxUint32] above
	}
	return ids, nil
}
