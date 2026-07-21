/*
Copyright 2025 The llm-d Authors.

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

package utils

// SliceMap applies a function to each element of a slice and returns a new
// slice with the results.
func SliceMap[Domain, Range any](slice []Domain, fn func(Domain) Range) []Range {
	if slice == nil {
		return nil
	}

	ans := make([]Range, len(slice))
	for idx, elt := range slice {
		ans[idx] = fn(elt)
	}

	return ans
}
