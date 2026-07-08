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

func TestStart_EmptyAddress(t *testing.T) {
	path := writeTemp(t, `
endpoints:
  - name: ep1
    address: ""
    port: "8000"
`)
	err := newFD(path, false).Start(context.Background(), &recordingNotifier{})
	assert.ErrorContains(t, err, "is not a valid IPv4 or hostname")
}

func TestStart_InvalidAddressFormat(t *testing.T) {
	tests := []struct {
		name    string
		address string
	}{
		{"spaces", "not a valid host"},
		{"special chars", "!!!garbage!!!"},
		{"trailing dot label", "host-.example.com"},
		{"leading hyphen", "-host.example.com"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := writeTemp(t, fmt.Sprintf(`
endpoints:
  - name: ep1
    address: %q
    port: "8000"
`, tt.address))
			err := newFD(path, false).Start(context.Background(), &recordingNotifier{})
			assert.ErrorContains(t, err, "is not a valid IPv4 or hostname")
		})
	}
}

func TestStart_RejectsIPv6(t *testing.T) {
	path := writeTemp(t, `
endpoints:
  - name: ep1
    address: "::1"
    port: "8000"
`)
	err := newFD(path, false).Start(context.Background(), &recordingNotifier{})
	assert.ErrorContains(t, err, "is not a valid IPv4 or hostname")
}

func TestStart_HostnameAddress(t *testing.T) {
	path := writeTemp(t, `
endpoints:
  - name: cluster-us-east
    namespace: clusters
    address: "spoke-us-east.example.com"
    port: "443"
    labels:
      region: us-east
      model: llama-7b
  - name: cluster-eu-west
    namespace: clusters
    address: "spoke-eu-west.example.com"
    port: "443"
    labels:
      region: eu-west
      model: llama-7b
`)
	notifier := &recordingNotifier{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	require.NoError(t, newFD(path, false).Start(ctx, notifier))
	require.Len(t, notifier.upserted, 2)

	for _, m := range notifier.upserted {
		assert.Empty(t, m.PodName, "hostname address should produce empty PodName")
		assert.Equal(t, "cluster", m.Labels[EndpointTypeLabel], "hostname should be auto-labeled as cluster")
	}
	assert.Equal(t, "spoke-us-east.example.com:443", notifier.upserted[0].MetricsHost)
	assert.Equal(t, "spoke-eu-west.example.com:443", notifier.upserted[1].MetricsHost)
	assert.ElementsMatch(t, []string{"clusters/cluster-us-east", "clusters/cluster-eu-west"}, notifier.upsertedNames())
}

func TestStart_IPv4PodNameIsSet(t *testing.T) {
	path := writeTemp(t, `
endpoints:
  - name: vllm-pod-1
    address: "10.0.0.5"
    port: "8000"
`)
	notifier := &recordingNotifier{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	require.NoError(t, newFD(path, false).Start(ctx, notifier))
	require.Len(t, notifier.upserted, 1)
	assert.Equal(t, "vllm-pod-1", notifier.upserted[0].PodName, "IPv4 address should preserve PodName")
	assert.Equal(t, "10.0.0.5:8000", notifier.upserted[0].MetricsHost)
	assert.Equal(t, "pod", notifier.upserted[0].Labels[EndpointTypeLabel], "IPv4 should be auto-labeled as pod")
}

func TestStart_MixedIPv4AndHostname(t *testing.T) {
	path := writeTemp(t, `
endpoints:
  - name: vllm-pod-1
    address: "10.0.0.5"
    port: "8000"
  - name: cluster-us-east
    address: "spoke-us-east.example.com"
    port: "443"
`)
	notifier := &recordingNotifier{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	require.NoError(t, newFD(path, false).Start(ctx, notifier))
	require.Len(t, notifier.upserted, 2)

	var pod, cluster *fwkdl.EndpointMetadata
	for _, m := range notifier.upserted {
		if m.NamespacedName.Name == "vllm-pod-1" {
			pod = m
		} else {
			cluster = m
		}
	}
	require.NotNil(t, pod)
	require.NotNil(t, cluster)
	assert.Equal(t, "vllm-pod-1", pod.PodName, "IPv4 endpoint should have PodName set")
	assert.Empty(t, cluster.PodName, "hostname endpoint should have empty PodName")
	assert.Equal(t, "pod", pod.Labels[EndpointTypeLabel], "IPv4 should be labeled as pod")
	assert.Equal(t, "cluster", cluster.Labels[EndpointTypeLabel], "hostname should be labeled as cluster")
}

func TestStart_EndpointTypeLabelAutoDetected(t *testing.T) {
	path := writeTemp(t, `
endpoints:
  - name: ep-ipv4
    address: "10.0.0.1"
    port: "8000"
  - name: ep-hostname
    address: "spoke.example.com"
    port: "443"
`)
	notifier := &recordingNotifier{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	require.NoError(t, newFD(path, false).Start(ctx, notifier))
	require.Len(t, notifier.upserted, 2)

	var ipv4, hostname *fwkdl.EndpointMetadata
	for _, m := range notifier.upserted {
		if m.NamespacedName.Name == "ep-ipv4" {
			ipv4 = m
		} else {
			hostname = m
		}
	}
	assert.Equal(t, "pod", ipv4.Labels[EndpointTypeLabel], "IPv4 should auto-detect as pod")
	assert.Equal(t, "cluster", hostname.Labels[EndpointTypeLabel], "hostname should auto-detect as cluster")
}

func TestStart_EndpointTypeLabelUserProvidedValid(t *testing.T) {
	path := writeTemp(t, `
endpoints:
  - name: ep1
    address: "spoke.example.com"
    port: "443"
    labels:
      llm-d.ai/endpoint-type: cluster
`)
	notifier := &recordingNotifier{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	require.NoError(t, newFD(path, false).Start(ctx, notifier))
	require.Len(t, notifier.upserted, 1)
	assert.Equal(t, "cluster", notifier.upserted[0].Labels[EndpointTypeLabel],
		"user-provided valid label should be preserved")
}

func TestStart_EndpointTypeLabelInvalidValueAutoCorrected(t *testing.T) {
	path := writeTemp(t, `
endpoints:
  - name: ep1
    address: "spoke.example.com"
    port: "443"
    labels:
      llm-d.ai/endpoint-type: clustr
`)
	notifier := &recordingNotifier{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	require.NoError(t, newFD(path, false).Start(ctx, notifier))
	require.Len(t, notifier.upserted, 1)
	assert.Equal(t, "cluster", notifier.upserted[0].Labels[EndpointTypeLabel],
		"invalid label value should be auto-corrected based on address type")
}

func TestStart_MetricsPortOverride(t *testing.T) {
	path := writeTemp(t, `
endpoints:
  - name: cluster-us-east
    address: "spoke-gateway.example.com"
    port: "443"
    metricsPort: "9090"
`)
	notifier := &recordingNotifier{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	require.NoError(t, newFD(path, false).Start(ctx, notifier))
	require.Len(t, notifier.upserted, 1)

	m := notifier.upserted[0]
	assert.Equal(t, "spoke-gateway.example.com", m.Address, "Address should be unchanged")
	assert.Equal(t, "443", m.Port, "Port should be unchanged")
	assert.Equal(t, "spoke-gateway.example.com:9090", m.MetricsHost, "MetricsHost should use address with metricsPort")
}

func TestStart_MetricsFieldsNotSetFallsBack(t *testing.T) {
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
	require.Len(t, notifier.upserted, 1)
	assert.Equal(t, "10.0.0.1:8000", notifier.upserted[0].MetricsHost, "MetricsHost should fall back to Address:Port")
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
