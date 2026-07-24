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
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
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
	return &FileDiscovery{
		path:      path,
		watchFile: watch,
		endpoints: make(map[types.NamespacedName]struct{}),
		ready:     make(chan struct{}),
	}
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
	_, err := Factory("", fwkplugin.StrictDecoder(json.RawMessage(`{}`)), nil)
	assert.ErrorContains(t, err, "'path' parameter is required")
}

func TestFactory_InvalidJSON(t *testing.T) {
	_, err := Factory("", fwkplugin.StrictDecoder(json.RawMessage(`{bad json`)), nil)
	assert.ErrorContains(t, err, "failed to parse parameters")
}

func TestFactory_ValidParams(t *testing.T) {
	path := writeTemp(t, validYAML)
	plugin, err := Factory("my-discovery", fwkplugin.StrictDecoder(json.RawMessage(`{"path":"`+path+`"}`)), nil)
	require.NoError(t, err)
	assert.Equal(t, PluginType, plugin.TypedName().Type)
	assert.Equal(t, "my-discovery", plugin.TypedName().Name)
}

func TestFactory_DefaultName(t *testing.T) {
	path := writeTemp(t, validYAML)
	plugin, err := Factory("", fwkplugin.StrictDecoder(json.RawMessage(`{"path":"`+path+`"}`)), nil)
	require.NoError(t, err)
	assert.Equal(t, PluginType, plugin.TypedName().Name)
}

func TestStart_LoadsEndpoints(t *testing.T) {
	path := writeTemp(t, validYAML)
	notifier := &recordingNotifier{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	fd := newFD(path, false)
	require.NoError(t, fd.Start(ctx, notifier))

	assert.ElementsMatch(t, []string{"ns1/ep1", "default/ep2"}, notifier.upsertedNames())
	assert.Empty(t, notifier.deleted)

	select {
	case <-fd.Ready():
	default:
		t.Fatal("Ready() channel should be closed after a successful initial load")
	}
}

func TestStart_EndpointType(t *testing.T) {
	tests := []struct {
		name    string
		typeVal string // value for `type:`; empty means the key is omitted
		want    fwkdl.EndpointType
		wantErr error
	}{
		{name: "omitted defaults to engine", want: fwkdl.EndpointTypeEngine},
		{name: "explicit engine", typeVal: "engine", want: fwkdl.EndpointTypeEngine},
		{name: "epp", typeVal: "epp", want: fwkdl.EndpointTypeEPP},
		{name: "unknown value", typeVal: "bogus", wantErr: fwkdl.ErrUnknownEndpointType},
		{name: "wrong case", typeVal: "EPP", wantErr: fwkdl.ErrUnknownEndpointType},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			content := `
endpoints:
  - name: ep1
    address: "10.0.0.1"
    port: "8000"
`
			if tt.typeVal != "" {
				content += "    type: " + tt.typeVal + "\n"
			}
			notifier := &recordingNotifier{}
			ctx, cancel := context.WithCancel(context.Background())
			cancel()

			err := newFD(writeTemp(t, content), false).Start(ctx, notifier)
			if tt.wantErr != nil {
				require.ErrorIs(t, err, tt.wantErr)
				assert.Empty(t, notifier.upserted, "an entry with an invalid type is not upserted")
				return
			}
			require.NoError(t, err)
			require.Len(t, notifier.upserted, 1)
			assert.Equal(t, tt.want, notifier.upserted[0].Type)
		})
	}
}

func TestStart_MixedEndpointTypes(t *testing.T) {
	path := writeTemp(t, `
endpoints:
  - name: spoke-epp
    address: "10.0.0.1"
    port: "8000"
    type: epp
  - name: vllm-pod
    address: "10.0.0.2"
    port: "8001"
`)
	notifier := &recordingNotifier{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	require.NoError(t, newFD(path, false).Start(ctx, notifier))

	got := map[string]fwkdl.EndpointType{}
	for _, m := range notifier.upserted {
		got[m.NamespacedName.Name] = m.Type
	}
	assert.Equal(t, fwkdl.EndpointTypeEPP, got["spoke-epp"])
	assert.Equal(t, fwkdl.EndpointTypeEngine, got["vllm-pod"])
}

func TestReady_StaysOpenWhenInitialLoadFails(t *testing.T) {
	fd := newFD("/nonexistent/endpoints.yaml", false)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := fd.Start(ctx, &recordingNotifier{})
	require.Error(t, err)

	select {
	case <-fd.Ready():
		t.Fatal("Ready() must not be closed when initial load fails")
	default:
	}
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

func TestStart_MetricsHostIsAddressPort(t *testing.T) {
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
	assert.ErrorContains(t, err, "invalid IPv4 address")
}

func TestStart_RejectsIPv6(t *testing.T) {
	path := writeTemp(t, `
endpoints:
  - name: ep1
    address: "::1"
    port: "8000"
`)
	err := newFD(path, false).Start(context.Background(), &recordingNotifier{})
	assert.ErrorContains(t, err, "invalid IPv4 address")
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

	newContent := []byte(`
endpoints:
  - name: ep3
    address: "10.0.0.3"
    port: "9000"
`)
	// Re-touch the file each poll so the write that lands after the watcher
	// is attached is the one that triggers the reload. Avoids racing on the
	// gap between Start()'s initial load and watcher.Add().
	require.Eventually(t, func() bool {
		if err := os.WriteFile(path, newContent, 0o600); err != nil {
			return false
		}
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

func TestDumpState(t *testing.T) {
	f := &FileDiscovery{
		endpoints: map[types.NamespacedName]struct{}{
			{Namespace: "default", Name: "pod-b"}: {},
			{Namespace: "default", Name: "pod-a"}: {},
		},
	}

	payload, err := f.DumpState()
	require.NoError(t, err)

	var state discoveryState
	require.NoError(t, json.Unmarshal(payload, &state))
	assert.Equal(t, []string{"default/pod-a", "default/pod-b"}, state.Endpoints)
	assert.Equal(t, 2, state.TotalEndpoints)
	assert.Equal(t, maxDebugDumpEndpoints, state.MaxEndpoints)
	// The full set fits, so the dump is complete (TotalEndpoints does not exceed MaxEndpoints).
	assert.LessOrEqual(t, state.TotalEndpoints, state.MaxEndpoints)
}

func TestDumpStateCaps(t *testing.T) {
	eps := make(map[types.NamespacedName]struct{}, maxDebugDumpEndpoints+5)
	for i := 0; i < maxDebugDumpEndpoints+5; i++ {
		eps[types.NamespacedName{Namespace: "default", Name: fmt.Sprintf("pod-%03d", i)}] = struct{}{}
	}
	f := &FileDiscovery{endpoints: eps}

	payload, err := f.DumpState()
	require.NoError(t, err)

	var state discoveryState
	require.NoError(t, json.Unmarshal(payload, &state))
	// The dump is partial: TotalEndpoints exceeds the returned count, capped at MaxEndpoints.
	assert.Equal(t, maxDebugDumpEndpoints+5, state.TotalEndpoints)
	assert.Greater(t, state.TotalEndpoints, state.MaxEndpoints)
	assert.Len(t, state.Endpoints, maxDebugDumpEndpoints)
	// Sorted ascending, then capped, so the first maxDebugDumpEndpoints names are kept.
	assert.Equal(t, "default/pod-000", state.Endpoints[0])
	assert.Equal(t, fmt.Sprintf("default/pod-%03d", maxDebugDumpEndpoints-1), state.Endpoints[maxDebugDumpEndpoints-1])
}

func TestDumpStateConcurrentWithLoad(t *testing.T) {
	path := writeTemp(t, "endpoints:\n- name: ep1\n  address: 10.0.0.1\n  port: \"8000\"\n")
	f := &FileDiscovery{path: path, endpoints: map[types.NamespacedName]struct{}{}}
	notifier := &recordingNotifier{}

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		for i := 0; i < 100; i++ {
			_ = f.load(notifier)
		}
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < 100; i++ {
			if _, err := f.DumpState(); err != nil {
				t.Errorf("DumpState returned error: %v", err)
			}
		}
	}()
	wg.Wait()
}

func TestDumpStateEmpty(t *testing.T) {
	f := &FileDiscovery{endpoints: map[types.NamespacedName]struct{}{}}

	payload, err := f.DumpState()
	require.NoError(t, err)
	assert.True(t, json.Valid(payload))

	var state discoveryState
	require.NoError(t, json.Unmarshal(payload, &state))
	assert.Empty(t, state.Endpoints)
	assert.Equal(t, 0, state.TotalEndpoints)
	assert.Equal(t, maxDebugDumpEndpoints, state.MaxEndpoints)
}
