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

package constants

const (
	// KVConnectorNIXLV2 enables the P/D KV NIXL v2 protocol
	KVConnectorNIXLV2 = "nixlv2"

	// KVConnectorSharedStorage enables the P/D KV Shared Storage protocol
	KVConnectorSharedStorage = "shared-storage"

	// KVConnectorSGLang enables SGLang the P/D KV disaggregation protocol
	KVConnectorSGLang = "sglang"

	// ECExampleConnector enables the Encoder disaggregation protocol (E/PD, E/P/D)
	ECExampleConnector = "ec-example"
)
