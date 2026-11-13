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

package parsers

import (
	"math"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
)

func TestNewUsage(t *testing.T) {
	require.Equal(t, &fwkrequest.Usage{
		PromptTokens:       7,
		CompletionTokens:   3,
		TotalTokens:        10,
		PromptTokenDetails: &fwkrequest.PromptTokenDetails{CachedTokens: 0},
	}, NewUsage(7, 3, 0))

	cached := 4
	require.Equal(t, &fwkrequest.Usage{
		PromptTokens:       7,
		CompletionTokens:   3,
		TotalTokens:        10,
		PromptTokenDetails: &fwkrequest.PromptTokenDetails{CachedTokens: 4},
	}, NewUsageOptionalCached(7, 3, &cached))

	require.Equal(t, &fwkrequest.Usage{
		PromptTokens:     7,
		CompletionTokens: 3,
		TotalTokens:      10,
	}, NewUsageOptionalCached(7, 3, nil))
}

func TestParseUnaryFrame(t *testing.T) {
	frame, err := grpcframe.FramePayload([]byte("abc"))
	require.NoError(t, err)
	payload, err := ParseUnaryFrame(frame)
	require.NoError(t, err)
	require.Equal(t, "abc", string(payload))

	_, err = ParseUnaryFrame(append(frame, 0xde, 0xad))
	require.ErrorIs(t, err, grpcframe.ErrMalformed)
	require.ErrorContains(t, err, "2 bytes after the frame")

	_, err = ParseUnaryFrame([]byte{1, 0, 0, 0, 0})
	require.ErrorIs(t, err, grpcframe.ErrCompressed)
	_, err = ParseUnaryFrame([]byte{0, 0, 0, 0, 9, 'a'})
	require.Error(t, err)
}

func TestClampToInt(t *testing.T) {
	require.Equal(t, 0, ClampToInt(uint32(0)))
	require.Equal(t, math.MaxUint32, ClampToInt(uint32(math.MaxUint32)))
	require.Equal(t, 1<<40, ClampToInt(uint64(1<<40)))
	require.Equal(t, math.MaxInt, ClampToInt(uint64(math.MaxUint64)))
}
