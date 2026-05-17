package ec

import (
	"github.com/llm-d/coordinator/pkg/connector"
	"github.com/llm-d/coordinator/pkg/pipeline"
)

// sharedStorage is the EC connector for the shared_storage topology. Encoder
// pods write embeddings to shared storage keyed by mm_hash; the consumer
// reads them back, so no ec_transfer_params is emitted on the wire.
type sharedStorage struct{}

func (sharedStorage) Name() string { return connector.ECSharedStorage }

func (sharedStorage) MergeEncodeResponse(_ *pipeline.RequestContext, _ map[string]any) {}

func (sharedStorage) PreparePrefillECParams(_ *pipeline.RequestContext) map[string]any {
	return nil
}
