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

// Package stringattribute declares string-valued endpoint attributes.
package stringattribute

import fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"

// Value is a string-valued endpoint attribute.
type Value string

// Clone returns the value unchanged.
func (v Value) Clone() fwkdl.Cloneable {
	return v
}

// ReadValue retrieves a string value from an endpoint attribute map.
func ReadValue(attrs fwkdl.AttributeMap, key string) (Value, bool) {
	return fwkdl.ReadAttribute[Value](attrs, key)
}
