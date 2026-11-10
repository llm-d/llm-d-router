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

package inflightload

import (
	"context"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
)

// TestRenamedPrefillProfile verifies that the prefill endpoint's in-flight request and
// tokens are released at the first response chunk whatever the prefill profile is named.
func TestRenamedPrefillProfile(t *testing.T) {
	t.Parallel()

	for _, profile := range []string{"prefill", "prefill-h200"} {
		for _, addOutput := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/addEstimatedOutputTokens=%t", profile, addOutput), func(t *testing.T) {
				t.Parallel()

				producer := newTestProducer(t)
				producer.addEstimatedOutputTokens = addOutput
				ctx := context.Background()
				prefillID := fullEndpointName("prefill-endpoint")

				req := makeTokenRequest("req-pd", 4)
				res := &fwksched.SchedulingResult{
					PrimaryProfileName: "decode",
					PrefillProfileName: profile,
					ProfileResults: map[string]*fwksched.ProfileRunResult{
						profile:  {TargetEndpoints: []fwksched.Endpoint{newStubSchedulingEndpointWithRole("prefill-endpoint", bylabel.RolePrefill)}},
						"decode": {TargetEndpoints: []fwksched.Endpoint{newStubSchedulingEndpointWithRole("decode-endpoint", bylabel.RoleDecode)}},
					},
				}
				require.NoError(t, producer.PreRequest(ctx, req, res))
				require.Equal(t, int64(1), producer.requestTracker.get(prefillID))
				require.Equal(t, int64(4), producer.tokenTracker.get(prefillID))

				req.SchedulingResult = res
				producer.ResponseBody(ctx, req, &fwkrh.Response{StartOfStream: true}, nil)
				assert.Equal(t, int64(0), producer.requestTracker.get(prefillID), "prefill request should be released at the first chunk")
				assert.Equal(t, int64(0), producer.tokenTracker.get(prefillID), "prefill tokens should be released at the first chunk")
			})
		}
	}
}
