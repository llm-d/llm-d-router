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

package metrics

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestBoundModel_EmptyIsUnknown(t *testing.T) {
	// Empty model name resolves to ModelUnknown before touching the cap, so
	// a flood of empty-model requests can never exhaust the cap.
	require.Equal(t, ModelUnknown, boundModel(""))
}

func TestBoundRoute_ClosedSet(t *testing.T) {
	require.Equal(t, RouteChatCompletions, boundRoute(RouteChatCompletions))
	require.Equal(t, RouteCompletions, boundRoute(RouteCompletions))
	require.Equal(t, RouteGenerate, boundRoute(RouteGenerate))
	require.Equal(t, RouteUnknown, boundRoute("/v1/chat/completions"))
	require.Equal(t, RouteUnknown, boundRoute(""))
}

func TestBoundMediaType_ClosedSet(t *testing.T) {
	require.Equal(t, MediaTypeImage, boundMediaType(MediaTypeImage))
	require.Equal(t, MediaTypeOther, boundMediaType("image/png"))
	require.Equal(t, MediaTypeOther, boundMediaType(""))
}

func TestBoundDownloadResult_ClosedSet(t *testing.T) {
	require.Equal(t, DownloadResultSuccess, boundDownloadResult(DownloadResultSuccess))
	require.Equal(t, DownloadResultError, boundDownloadResult(DownloadResultError))
	require.Equal(t, DownloadResultCancelled, boundDownloadResult(DownloadResultCancelled))
	require.Equal(t, DownloadResultError, boundDownloadResult("timeout"))
}
