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

package ec

import (
	"context"

	"github.com/llm-d/coordinator/pkg/pipeline"
)

// sharedStorageEC is the EC connector for the ec-shared-storage topology. Encoder
// pods write embeddings to shared storage keyed by mm_hash; the consumer
// reads them back, so no ec_transfer_params is emitted on the wire.
type sharedStorageEC struct{}

func (sharedStorageEC) Name() string { return SharedStorage }

func (sharedStorageEC) MergeEncodeResponse(_ context.Context, _ *pipeline.RequestContext, _ map[string]any) {
}

func (sharedStorageEC) PreparePrefillECParams(_ context.Context, _ *pipeline.RequestContext) (map[string]any, error) {
	return make(map[string]any), nil
}
