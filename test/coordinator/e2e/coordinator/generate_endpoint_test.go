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

package coordinate2e

import (
	"encoding/json"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
)

// genImage describes one image entry in a native /inference/v1/generate request.
// Cached marks a cache-hit entry: its kwargs_data is emitted as null so the
// encoder resolves it from its own cache by hash. Offset/Length locate the
// image's placeholder span within token_ids.
type genImage struct {
	Hash   string
	Offset int
	Length int
	Cached bool
}

// generateTestKwargs is a base64 blob standing in for a real (non-cache-hit)
// encoder kwargs payload.
const generateTestKwargs = "dGVuc29y"

// generateSteps lists the pipeline steps a token-only generate request drives:
// render parses token_ids locally, then prefill and decode. replace-media-urls
// and encode run but no-op (no messages, no images), so they are not asserted.
var generateSteps = []string{"render", "prefill", "decode"}

var _ = ginkgo.Describe("Coordinator pipeline - generate endpoint", func() {
	ginkgo.It("routes a text-only generate end-to-end", func() {
		runCoordinatorPipeline(gateway.DefaultGeneratePath,
			generateBody(modelName, nil), generateSteps, 0)
	})

	ginkgo.It("routes a single-image generate end-to-end", func() {
		images := []genImage{
			{Hash: "e2e-gen-hash-0", Offset: 1, Length: 3},
		}
		runCoordinatorPipeline(gateway.DefaultGeneratePath,
			generateBody(modelName, images), allSteps, 1)
	})

	// A cached (null kwargs_data) entry is still parsed and fanned out to the
	// encoder: the fan-out is one sub-request per multimodal entry, so the
	// encode count equals the number of entries (2), not the number of non-null
	// entries (1). verifyCoordinatorSteps asserts count=2 via the
	// "all sub-requests complete" / "merged encode response" markers.
	ginkgo.It("routes a two-image generate with one cached (null kwargs) image end-to-end", func() {
		images := []genImage{
			{Hash: "e2e-gen-hash-0", Offset: 1, Length: 2},
			{Hash: "e2e-gen-hash-1", Offset: 4, Length: 2, Cached: true},
		}
		runCoordinatorPipeline(gateway.DefaultGeneratePath,
			generateBody(modelName, images), allSteps, 2)
	})
})

// generateBody builds a native /inference/v1/generate request body. With no
// images it is text-only (token_ids, no features). With images it adds the
// parallel features arrays (mm_hashes, mm_placeholders, kwargs_data) keyed by
// the image modality; a nil Kwargs marshals to JSON null (cache hit).
func generateBody(model string, images []genImage) []byte {
	body := map[string]any{
		"model":           model,
		"token_ids":       generateTokenIDs(images),
		"sampling_params": map[string]any{"max_tokens": 1},
	}

	if len(images) > 0 {
		hashes := make([]any, len(images))
		placeholders := make([]any, len(images))
		kwargs := make([]any, len(images))
		for i, img := range images {
			hashes[i] = img.Hash
			placeholders[i] = map[string]any{"offset": img.Offset, "length": img.Length}
			// A cached entry leaves kwargs[i] nil (JSON null); the encoder
			// resolves it from its own cache by hash.
			if !img.Cached {
				kwargs[i] = generateTestKwargs
			}
		}
		body["features"] = map[string]any{
			"mm_hashes":       map[string]any{"image": hashes},
			"mm_placeholders": map[string]any{"image": placeholders},
			"kwargs_data":     map[string]any{"image": kwargs},
		}
	}

	raw, err := json.Marshal(body)
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred(), "marshal generate body")
	return raw
}

// generateTokenIDs builds a token_ids slice that covers every image's
// placeholder span (positions [offset, offset+length) hold a placeholder
// token). Text-only requests get a fixed 20-token prompt.
func generateTokenIDs(images []genImage) []int {
	if len(images) == 0 {
		ids := make([]int, 20)
		for i := range ids {
			ids[i] = 1000 + i
		}
		return ids
	}

	const placeholderTokenID = 32000
	maxEnd := 1
	for _, img := range images {
		if end := img.Offset + img.Length; end > maxEnd {
			maxEnd = end
		}
	}
	ids := make([]int, maxEnd+1)
	for i := range ids {
		ids[i] = 1000 + i
	}
	for _, img := range images {
		for j := img.Offset; j < img.Offset+img.Length; j++ {
			ids[j] = placeholderTokenID
		}
	}
	return ids
}
