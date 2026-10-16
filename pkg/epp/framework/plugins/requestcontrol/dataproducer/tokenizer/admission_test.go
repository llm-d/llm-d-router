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

package tokenizer

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

type admissionTokenBackend struct{ calls int }

func (b *admissionTokenBackend) produce(context.Context, *fwkrh.InferenceRequestBody) (*fwkrh.TokenizedRequest, error) {
	b.calls++
	return fwkrh.NewTokenizedRequest([][]uint32{{1, 2}}), nil
}

func TestAdmissionPreparationReusesTokensForProduce(t *testing.T) {
	backend := &admissionTokenBackend{}
	p := &Plugin{backend: backend}
	request := &scheduling.InferenceRequest{Body: &fwkrh.InferenceRequestBody{}}
	require.NoError(t, p.PrepareForAdmission(t.Context(), request, nil))
	require.NoError(t, p.PrepareForAdmission(t.Context(), request, nil))
	require.NoError(t, p.Produce(t.Context(), request, nil))
	require.Equal(t, 1, backend.calls)
	require.Equal(t, 2, request.Body.TokenizedRequest.TokenCount())
}
