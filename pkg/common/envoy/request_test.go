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

package envoy

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestGenerateUnmodifiedRequestBodyResponse(t *testing.T) {
	resps := GenerateUnmodifiedRequestBodyResponse()
	require.Len(t, resps, 1)
	body := resps[0].GetRequestBody()
	require.NotNil(t, body)
	require.NotNil(t, body.GetResponse())
	require.Nil(t, body.GetResponse().GetBodyMutation())
}

func TestGenerateRequestBodyResponsesIncludesBody(t *testing.T) {
	body := []byte("hello")
	resps := GenerateRequestBodyResponses(body, true)
	require.Len(t, resps, 1)
	streamed := resps[0].GetRequestBody().GetResponse().GetBodyMutation().GetStreamedResponse()
	require.Equal(t, body, streamed.GetBody())
	require.True(t, streamed.GetEndOfStream())
}
