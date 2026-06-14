package ec

import (
	"github.com/llm-d/coordinator/pkg/pipeline"
)

// sharedStorageEC is the EC connector for the ec-shared-storage topology. Encoder
// pods write embeddings to shared storage keyed by mm_hash; the consumer
// reads them back, so no ec_transfer_params is emitted on the wire.
type sharedStorageEC struct{}

func (sharedStorageEC) Name() string { return SharedStorage }

func (sharedStorageEC) MergeEncodeResponse(_ *pipeline.RequestContext, _ map[string]any) {}

func (sharedStorageEC) PreparePrefillECParams(_ *pipeline.RequestContext) (map[string]any, error) {
	return make(map[string]any), nil
}
