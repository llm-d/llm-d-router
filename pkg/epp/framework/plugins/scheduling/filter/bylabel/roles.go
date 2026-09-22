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

package bylabel

const (
	// RoleLabel name
	RoleLabel = "llm-d.ai/role"

	// RoleDecode set for designated decode workers
	RoleDecode = "decode"
	// RolePrefill set for designated prefill workers
	RolePrefill = "prefill"
	// RoleEncode set for designated encode workers (first stage in pipeline)
	RoleEncode = "encode"

	// RolePrefillDecode set for workers that can handle prefill and decode stages
	RolePrefillDecode = "prefill-decode"
	// RoleEncodePrefill set for workers that can handle encode+prefill (EP/D or P/D disaggregation)
	RoleEncodePrefill = "encode-prefill"
	// RoleEncodePrefillDecode set for workers that can handle encode+prefill+decode
	RoleEncodePrefillDecode = "encode-prefill-decode"
)
