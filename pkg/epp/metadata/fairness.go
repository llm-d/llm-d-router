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

package metadata

import "strings"

// EscapeTenantID percent-encodes the characters that would otherwise make the
// tenant/remainder split of a FairnessIDTenantSeparator-joined key ambiguous: "%" and the
// separator itself. A tenant with neither is returned unchanged without allocating.
func EscapeTenantID(id string) string {
	if !strings.ContainsAny(id, "%"+FairnessIDTenantSeparator) {
		return id
	}
	id = strings.ReplaceAll(id, "%", "%25")
	return strings.ReplaceAll(id, FairnessIDTenantSeparator, "%2F")
}
