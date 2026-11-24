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

package runner

import (
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/go-logr/logr/funcr"
	"github.com/spf13/pflag"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/labels"

	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"
)

func TestValidateKubernetesDiscoveryFlags(t *testing.T) {
	opts := runserver.NewOptions()
	require.ErrorContains(t, validateKubernetesDiscoveryFlags(opts), "either pool-name or endpoint-selector must be set")

	opts.PoolName = "pool"
	require.NoError(t, validateKubernetesDiscoveryFlags(opts))

	selector, err := labels.Parse("app=vllm")
	require.NoError(t, err)
	opts = runserver.NewOptions()
	opts.EndpointSelector = selector
	require.NoError(t, validateKubernetesDiscoveryFlags(opts))

	opts.PoolName = "pool"
	require.ErrorContains(t, validateKubernetesDiscoveryFlags(opts), "cannot both be set")

	opts.EndpointSelector = nil
	opts.ObjectivesFile = "/etc/epp/objectives.yaml"
	require.ErrorContains(t, validateKubernetesDiscoveryFlags(opts), "objectives-file requires file discovery")
}

func TestRun_LogsCompleteFlagsError(t *testing.T) {
	prevFlagSet := pflag.CommandLine
	prevArgs := os.Args
	pflag.CommandLine = pflag.NewFlagSet("epp-test", pflag.ContinueOnError)
	os.Args = []string{"epp", "--tls-min-version=1.3"}
	t.Cleanup(func() {
		pflag.CommandLine = prevFlagSet
		os.Args = prevArgs
	})

	var mu sync.Mutex
	var logged strings.Builder
	previous := setupLog
	setupLog = funcr.New(func(prefix, args string) {
		mu.Lock()
		defer mu.Unlock()
		logged.WriteString(prefix + " " + args + "\n")
	}, funcr.Options{})
	t.Cleanup(func() { setupLog = previous })

	err := NewRunner().Run(t.Context())
	require.ErrorContains(t, err, `invalid tls-min-version "1.3"`)

	mu.Lock()
	logs := logged.String()
	mu.Unlock()
	assert.Contains(t, logs, "Failed to complete flags")
	assert.Contains(t, logs, "invalid tls-min-version")
	assert.Contains(t, logs, "unknown TLS version")
}
