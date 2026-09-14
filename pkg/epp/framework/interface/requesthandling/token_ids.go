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

package requesthandling

import "math"

func parseCanonicalTokenIDArrays(data []byte) ([][]uint32, bool) {
	i := skipJSONSpace(data, 0)
	if i >= len(data) || data[i] != '[' {
		return nil, false
	}
	i = skipJSONSpace(data, i+1)
	if i >= len(data) || data[i] == ']' {
		return nil, false
	}

	if data[i] != '[' {
		tokenIDs, ok := ParseCanonicalTokenIDs(data)
		if !ok {
			return nil, false
		}
		return [][]uint32{tokenIDs}, true
	}

	tokenCapacity, arrayCapacity := 1, 0
	for _, c := range data {
		switch c {
		case ',':
			tokenCapacity++
		case '[':
			arrayCapacity++
		}
	}

	tokenIDs := make([]uint32, 0, tokenCapacity)
	arrays := make([][]uint32, 0, arrayCapacity-1)
	for {
		i = skipJSONSpace(data, i)
		if i >= len(data) || data[i] != '[' {
			return nil, false
		}

		start := len(tokenIDs)
		var ok bool
		tokenIDs, i, ok = parseCanonicalTokenIDArray(data, i+1, tokenIDs)
		if !ok {
			return nil, false
		}
		end := len(tokenIDs)
		arrays = append(arrays, tokenIDs[start:end:end])

		i = skipJSONSpace(data, i)
		switch {
		case i >= len(data):
			return nil, false
		case data[i] == ']':
			i = skipJSONSpace(data, i+1)
			if i != len(data) {
				return nil, false
			}
			return arrays, true
		case data[i] == ',':
			i++
		default:
			return nil, false
		}
	}
}

// ParseCanonicalTokenIDs parses a non-empty JSON array of canonical uint32 values.
func ParseCanonicalTokenIDs(data []byte) ([]uint32, bool) {
	i := skipJSONSpace(data, 0)
	if i >= len(data) || data[i] != '[' {
		return nil, false
	}
	i = skipJSONSpace(data, i+1)
	if i >= len(data) || data[i] == ']' || data[i] == '[' {
		return nil, false
	}

	capacity := 1
	for _, c := range data {
		if c == ',' {
			capacity++
		}
	}
	tokenIDs, i, ok := parseCanonicalTokenIDArray(data, i, make([]uint32, 0, capacity))
	if !ok || skipJSONSpace(data, i) != len(data) {
		return nil, false
	}
	return tokenIDs, true
}

func parseCanonicalTokenIDArray(data []byte, i int, tokenIDs []uint32) ([]uint32, int, bool) {
	i = skipJSONSpace(data, i)
	if i >= len(data) || data[i] < '0' || data[i] > '9' {
		return nil, 0, false
	}

	for {
		start := i
		value := uint64(0)
		for i < len(data) && data[i] >= '0' && data[i] <= '9' {
			value = value*10 + uint64(data[i]-'0')
			if value > math.MaxUint32 {
				return nil, 0, false
			}
			i++
		}
		if data[start] == '0' && i-start > 1 {
			return nil, 0, false
		}
		tokenIDs = append(tokenIDs, uint32(value))

		i = skipJSONSpace(data, i)
		switch {
		case i >= len(data):
			return nil, 0, false
		case data[i] == ']':
			return tokenIDs, i + 1, true
		case data[i] == ',':
			i = skipJSONSpace(data, i+1)
			if i >= len(data) || data[i] < '0' || data[i] > '9' {
				return nil, 0, false
			}
		default:
			return nil, 0, false
		}
	}
}

func skipJSONSpace(data []byte, i int) int {
	for i < len(data) {
		switch data[i] {
		case ' ', '\t', '\n', '\r':
			i++
		default:
			return i
		}
	}
	return i
}
