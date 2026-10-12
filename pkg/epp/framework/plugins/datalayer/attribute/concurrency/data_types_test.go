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

package concurrency

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestInFlightLoadTokenLoad(t *testing.T) {
	require.Equal(t, TokenLoad{Text: 6, NonText: 4}, (&InFlightLoad{Tokens: 10, NonTextTokens: 4}).TokenLoad())
	// A value from a replica that predates NonTextTokens is all text.
	require.Equal(t, TokenLoad{Text: 10}, (&InFlightLoad{Tokens: 10}).TokenLoad())
	// A malformed value can't make the text share negative.
	require.Equal(t, TokenLoad{NonText: 10}, (&InFlightLoad{Tokens: 10, NonTextTokens: 15}).TokenLoad())
	require.Equal(t, TokenLoad{Text: 10}, (&InFlightLoad{Tokens: 10, NonTextTokens: -3}).TokenLoad())
}
