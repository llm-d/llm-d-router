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

package steps

import (
	"fmt"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/kv"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

func validateMooncakeRequest(reqCtx *pipeline.RequestContext) error {
	apiType, matched := reqcommon.MatchAPIType(reqCtx.OriginalPath)
	if reqCtx.GRPC != nil || !matched || apiType != reqcommon.APITypeChatCompletions {
		return fmt.Errorf("%w: %s requires HTTP chat completions", pipeline.ErrBadRequest, kv.Mooncake)
	}
	if len(reqCtx.MultimodalEntries) > 0 || len(clientMediaParts(reqCtx)) > 0 {
		return fmt.Errorf("%w: %s requires text-only input", pipeline.ErrBadRequest, kv.Mooncake)
	}
	if reqcommon.OutputChoiceCount(reqCtx.Body, apiType) != 1 {
		return fmt.Errorf("%w: %s requires one completion", pipeline.ErrBadRequest, kv.Mooncake)
	}
	return nil
}
