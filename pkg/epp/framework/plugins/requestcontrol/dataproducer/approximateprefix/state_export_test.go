/*
Copyright 2026 The Kubernetes Authors.

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

package approximateprefix

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestDataProducer_CommitAndGetPrefixMatch(t *testing.T) {
	config := config{
		BlockSizeTokens:        1,
		MaxPrefixBlocksToMatch: defaultMaxPrefixBlocks,
		LRUCapacityPerServer:   defaultLRUCapacityPerServer,
	}
	p, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, config, testHandle())
	require.NoError(t, err)

	p.CommitPrefix("default/pod-a", []uint64{1, 2, 3})

	require.ElementsMatch(t, []string{"default/pod-a"}, p.GetPrefixMatch(1))
	require.ElementsMatch(t, []string{"default/pod-a"}, p.GetPrefixMatch(2))
	require.Empty(t, p.GetPrefixMatch(999))
}

func TestDataProducer_RemovePrefixEndpoint(t *testing.T) {
	config := config{
		BlockSizeTokens:        1,
		MaxPrefixBlocksToMatch: defaultMaxPrefixBlocks,
		LRUCapacityPerServer:   defaultLRUCapacityPerServer,
	}
	p, err := newDataProducer(context.Background(), ApproxPrefixCachePluginType, config, testHandle())
	require.NoError(t, err)

	p.CommitPrefix("default/pod-a", []uint64{1})
	require.NotEmpty(t, p.GetPrefixMatch(1))

	p.RemovePrefixEndpoint("default/pod-a")

	require.Empty(t, p.GetPrefixMatch(1))
}
