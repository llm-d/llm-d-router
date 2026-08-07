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

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// The factory must delegate to the stock producer (the inner.(*dataProducer)
// type assertion) and report the multi-cluster type. A name-based data key
// (verified elsewhere) means the delegate binds to the prefix scorer unchanged.
func TestMultiClusterFactory(t *testing.T) {
	h := plugin.NewEppHandle(context.Background(), nil, plugin.WithMetricsRecorder(prometheus.NewRegistry()))
	p, err := MultiClusterFactory("mc", nil, h)
	require.NoError(t, err)
	require.IsType(t, &MultiClusterProducer{}, p)
	require.Equal(t, MultiClusterPluginType, p.TypedName().Type)
	require.Equal(t, "mc", p.TypedName().Name)
}
