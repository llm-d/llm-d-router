/*
Copyright 2025 The Kubernetes Authors.

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

package file

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/datalayer"
)

// recordingNotifier captures Upsert and Delete calls for assertions.
type recordingNotifier struct {
	mu       sync.Mutex
	upserted []*fwkdl.EndpointMetadata
	deleted  []types.NamespacedName
}

func (r *recordingNotifier) Upsert(meta *fwkdl.EndpointMetadata) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.upserted = append(r.upserted, meta)
}

func (r *recordingNotifier) Delete(id types.NamespacedName) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.deleted = append(r.deleted, id)
}

func (r *recordingNotifier) upsertedNames() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	names := make([]string, len(r.upserted))
	for i, m := range r.upserted {
		names[i] = m.NamespacedName.String()
	}
	return names
}

func writeTemp(t *testing.T, content string) string {
	t.Helper()
	f, err := os.CreateTemp(t.TempDir(), "endpoints-*.yaml")
	require.NoError(t, err)
	_, err = f.WriteString(content)
	require.NoError(t, err)
	require.NoError(t, f.Close())
	return f.Name()
}

func newFD(path string, watch bool) *FileDiscovery {
	return &FileDiscovery{path: path, watchFile: watch, current: make(map[types.NamespacedName]struct{})}
}

const validYAML = `
endpoints:
  - name: ep1
    namespace: ns1
    address: "10.0.0.1"
    port: "8000"
  - name: ep2
    address: "10.0.0.2"
    port: "8001"
`

func TestFactory_MissingPath(t *testing.T) {
	_, err := Factory("", json.RawMessage(`{}`), nil)
	assert.ErrorContains(t, err, "'path' parameter is required")
}

func TestFactory_InvalidJSON(t *testing.T) {
	_, err := Factory("", json.RawMessage(`{bad json`), nil)
	assert.ErrorContains(t, err, "failed to parse parameters")
}

func TestFactory_ValidParams(t *testing.T) {
	path := writeTemp(t, validYAML)
	plugin, err := Factory("my-discovery", json.RawMessage(`{"path":"`+path+`"}`), nil)
	require.NoError(t, err)
	assert.Equal(t, PluginType, plugin.TypedName().Type)
	assert.Equal(t, "my-discovery", plugin.TypedName().Name)
}

func TestFactory_DefaultName(t *testing.T) {
	path := writeTemp(t, validYAML)
	plugin, err := Factory("", json.RawMessage(`{"path":"`+path+`"}`), nil)
	require.NoError(t, err)
	assert.Equal(t, PluginType, plugin.TypedName().Name)
}

func TestStart_LoadsEndpoints(t *testing.T) {
	path := writeTemp(t, validYAML)
	notifier := &recordingNotifier{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	require.NoError(t, newFD(path, false).Start(ctx, notifier))

	assert.ElementsMatch(t, []string{"ns1/ep1", "default/ep2"}, notifier.upsertedNames())
	assert.Empty(t, notifier.deleted)
}

func TestStart_DefaultNamespace(t *testing.T) {
	path := writeTemp(t, `
endpoints:
  - name: ep1
    address: "10.0.0.1"
    port: "8000"
`)
	notifier := &recordingNotifier{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	require.NoError(t, newFD(path, false).Start(ctx, notifier))
	assert.Equal(t, "default", notifier.upserted[0].NamespacedName.Namespace)
}

func TestStart_MetricsHostDefaultsToAddressPort(t *testing.T) {
	path := writeTemp(t, `
endpoints:
  - name: ep1
    address: "10.0.0.1"
    port: "8000"
`)
	notifier := &recordingNotifier{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	require.NoError(t, newFD(path, false).Start(ctx, notifier))
	assert.Equal(t, "10.0.0.1:8000", notifier.upserted[0].MetricsHost)
}

func TestStart_MetricsHostExplicit(t *testing.T) {
	path := writeTemp(t, `
endpoints:
  - name: ep1
    address: "10.0.0.1"
    port: "8000"
    metricsHost: "10.0.0.1:9090"
`)
	notifier := &recordingNotifier{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	require.NoError(t, newFD(path, false).Start(ctx, notifier))
	assert.Equal(t, "10.0.0.1:9090", notifier.upserted[0].MetricsHost)
}

func TestStart_MissingFile(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := newFD("/nonexistent/endpoints.yaml", false).Start(ctx, &recordingNotifier{})
	assert.ErrorContains(t, err, "initial load failed")
}

func TestStart_InvalidIP(t *testing.T) {
	path := writeTemp(t, `
endpoints:
  - name: ep1
    address: "not-an-ip"
    port: "8000"
`)
	err := newFD(path, false).Start(context.Background(), &recordingNotifier{})
	assert.ErrorContains(t, err, "invalid address")
}

func TestStart_InvalidPort(t *testing.T) {
	path := writeTemp(t, `
endpoints:
  - name: ep1
    address: "10.0.0.1"
    port: "99999"
`)
	err := newFD(path, false).Start(context.Background(), &recordingNotifier{})
	assert.ErrorContains(t, err, "invalid port")
}

func TestStart_FileTooLarge(t *testing.T) {
	content := strings.Repeat("x", maxEndpointsFileSize+1)
	path := writeTemp(t, content)
	err := newFD(path, false).Start(context.Background(), &recordingNotifier{})
	assert.ErrorContains(t, err, "exceeds 1 MiB limit")
}

func TestStart_DeletesRemovedEndpoints(t *testing.T) {
	path := writeTemp(t, validYAML)
	fd := newFD(path, false)
	notifier := &recordingNotifier{}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	require.NoError(t, fd.Start(ctx, notifier))
	assert.Len(t, notifier.upserted, 2)

	require.NoError(t, os.WriteFile(path, []byte(`
endpoints:
  - name: ep1
    namespace: ns1
    address: "10.0.0.1"
    port: "8000"
`), 0o600))
	notifier2 := &recordingNotifier{}
	ctx2, cancel2 := context.WithCancel(context.Background())
	cancel2()
	require.NoError(t, fd.Start(ctx2, notifier2))

	assert.Len(t, notifier2.upserted, 1)
	assert.Len(t, notifier2.deleted, 1)
	assert.Equal(t, types.NamespacedName{Name: "ep2", Namespace: "default"}, notifier2.deleted[0])
}

func TestStart_WatchFileReloadsOnWrite(t *testing.T) {
	path := writeTemp(t, validYAML)
	fd := newFD(path, true)
	notifier := &recordingNotifier{}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	done := make(chan error, 1)
	go func() { done <- fd.Start(ctx, notifier) }()

	time.Sleep(100 * time.Millisecond)
	require.NoError(t, os.WriteFile(path, []byte(`
endpoints:
  - name: ep3
    address: "10.0.0.3"
    port: "9000"
`), 0o600))

	assert.Eventually(t, func() bool {
		notifier.mu.Lock()
		defer notifier.mu.Unlock()
		for _, m := range notifier.upserted {
			if m.NamespacedName.Name == "ep3" {
				return true
			}
		}
		return false
	}, 2*time.Second, 50*time.Millisecond)

	cancel()
	assert.NoError(t, <-done)
}
