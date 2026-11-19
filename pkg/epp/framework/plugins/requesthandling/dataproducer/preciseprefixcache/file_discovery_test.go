/*
Copyright 2025 The llm-d Authors.

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

package preciseprefixcache

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/discovery/file"
)

// extractNotifier forwards file-discovery notifications to the producer the
// way the datastore runtime does.
type extractNotifier struct {
	ctx  context.Context
	p    *Producer
	seen *sync.Map // endpoint ID string -> last upserted *fwkdl.EndpointMetadata
}

func (n extractNotifier) Upsert(meta *fwkdl.EndpointMetadata) {
	n.seen.Store(meta.ID.String(), meta)
	_ = n.p.Extract(n.ctx, fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: fwkdl.NewEndpoint(meta, nil)})
}

func (n extractNotifier) Delete(id k8stypes.NamespacedName) {
	_ = n.p.Extract(n.ctx, fwkdl.EndpointEvent{Type: fwkdl.EventDelete, Endpoint: fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: id}, nil)})
}

// startFileDiscovery runs the real file-discovery plugin against path, feeding p.
// It returns the endpoint metadata discovery upserted, keyed by endpoint ID.
func startFileDiscovery(t *testing.T, p *Producer, path string, watch bool) *sync.Map {
	t.Helper()
	params := fmt.Sprintf(`{"path": %q, "watchFile": %t}`, path, watch)
	plug, err := file.Factory("fd", json.NewDecoder(strings.NewReader(params)), nil)
	require.NoError(t, err)
	fd := plug.(*file.FileDiscovery)

	ctx, cancel := context.WithCancel(discardCtx(t))
	done := make(chan struct{})
	seen := &sync.Map{}
	go func() {
		defer close(done)
		_ = fd.Start(ctx, extractNotifier{ctx: ctx, p: p, seen: seen})
	}()
	t.Cleanup(func() {
		cancel()
		<-done
		p.subscribersManager.Shutdown(discardCtx(t))
	})

	select {
	case <-fd.Ready():
	case <-done:
		t.Fatal("file-discovery exited before ready")
	case <-time.After(5 * time.Second):
		t.Fatal("file-discovery not ready")
	}
	return seen
}

func writeEndpoints(t *testing.T, path, body string) {
	t.Helper()
	tmp := path + ".tmp"
	require.NoError(t, os.WriteFile(tmp, []byte(body), 0o600))
	require.NoError(t, os.Rename(tmp, path))
}

func TestProducer_FileDiscoveryEndpointsDriveSubscribers(t *testing.T) {
	tests := []struct {
		name          string
		yaml          string
		wantIDs       []string
		wantEndpoints []string
	}{
		{
			name: "workers on different hosts use default socket port",
			yaml: `endpoints:
- {name: vllm-0, address: 10.0.0.1, port: "8000"}
- {name: vllm-1, address: 10.0.0.2, port: "8000"}
`,
			wantIDs:       []string{"default/vllm-0", "default/vllm-1"},
			wantEndpoints: []string{"tcp://10.0.0.1:5557", "tcp://10.0.0.2:5557"},
		},
		{
			name: "co-located ranks offset socket port by rankIndex",
			yaml: `endpoints:
- {name: vllm-0, address: 10.0.0.1, port: "8000", rankIndex: 0}
- {name: vllm-1, address: 10.0.0.1, port: "8001", rankIndex: 1}
`,
			wantIDs:       []string{"default/vllm-0", "default/vllm-1"},
			wantEndpoints: []string{"tcp://10.0.0.1:5557", "tcp://10.0.0.1:5558"},
		},
		{
			name: "per-entry zmqPort is used as the exact socket port",
			yaml: `endpoints:
- {name: vllm-0, address: 10.0.0.1, port: "8000", zmqPort: 6001}
- {name: vllm-1, address: 10.0.0.1, port: "8001", zmqPort: 7002, rankIndex: 1}
`,
			wantIDs:       []string{"default/vllm-0", "default/vllm-1"},
			wantEndpoints: []string{"tcp://10.0.0.1:6001", "tcp://10.0.0.1:7002"},
		},
		{
			name: "zmqPort on one entry leaves the other on the default port",
			yaml: `endpoints:
- {name: vllm-0, address: 10.0.0.1, port: "8000", zmqPort: 6001}
- {name: vllm-1, address: 10.0.0.2, port: "8000"}
`,
			wantIDs:       []string{"default/vllm-0", "default/vllm-1"},
			wantEndpoints: []string{"tcp://10.0.0.1:6001", "tcp://10.0.0.2:5557"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := newExtractorProducer(t, true)
			path := filepath.Join(t.TempDir(), "endpoints.yaml")
			writeEndpoints(t, path, tc.yaml)
			startFileDiscovery(t, p, path, false)

			ids, endpoints := p.subscribersManager.GetActiveSubscribers()
			assert.ElementsMatch(t, tc.wantIDs, ids)
			assert.ElementsMatch(t, tc.wantEndpoints, endpoints)
		})
	}
}

func TestProducer_FileDiscoveryReplayAndSnapshotEndpoints(t *testing.T) {
	type endpoints struct{ replay, snapshot string }
	tests := []struct {
		name string
		yaml string
		want map[string]endpoints
	}{
		{
			name: "entries sharing an address with distinct zmqPort run live-only",
			yaml: `endpoints:
- {name: vllm-0, address: 10.0.0.1, port: "8000", zmqPort: 6001}
- {name: vllm-1, address: 10.0.0.1, port: "8001", zmqPort: 6002}
`,
			want: map[string]endpoints{
				"default/vllm-0": {},
				"default/vllm-1": {},
			},
		},
		{
			name: "entries without zmqPort offset replay and snapshot ports by rankIndex",
			yaml: `endpoints:
- {name: vllm-0, address: 10.0.0.1, port: "8000", rankIndex: 0}
- {name: vllm-1, address: 10.0.0.1, port: "8001", rankIndex: 1}
`,
			want: map[string]endpoints{
				"default/vllm-0": {replay: "tcp://10.0.0.1:5600", snapshot: "tcp://10.0.0.1:5700"},
				"default/vllm-1": {replay: "tcp://10.0.0.1:5601", snapshot: "tcp://10.0.0.1:5701"},
			},
		},
		{
			name: "zmqPort entry runs live-only while a rank entry keeps replay and snapshot",
			yaml: `endpoints:
- {name: vllm-0, address: 10.0.0.1, port: "8000", zmqPort: 6001, rankIndex: 1}
- {name: vllm-1, address: 10.0.0.1, port: "8001", rankIndex: 1}
`,
			want: map[string]endpoints{
				"default/vllm-0": {},
				"default/vllm-1": {replay: "tcp://10.0.0.1:5601", snapshot: "tcp://10.0.0.1:5701"},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := newExtractorProducer(t, true)
			p.kvEventsConfig.PodDiscoveryConfig.ReplaySocketPort = 5600
			p.kvEventsConfig.PodDiscoveryConfig.SnapshotSocketPort = 5700
			path := filepath.Join(t.TempDir(), "endpoints.yaml")
			writeEndpoints(t, path, tc.yaml)
			seen := startFileDiscovery(t, p, path, false)

			ids, _ := p.subscribersManager.GetActiveSubscribers()
			require.Len(t, ids, len(tc.want))
			for id, want := range tc.want {
				v, ok := seen.Load(id)
				require.True(t, ok, "endpoint %s not discovered", id)
				_, replay, snapshot := p.subscriberEndpoints(v.(*fwkdl.EndpointMetadata))
				assert.Equal(t, want, endpoints{replay: replay, snapshot: snapshot}, id)
			}
		})
	}
}

func TestProducer_FileDiscoveryRemovedEntryStopsSubscriber(t *testing.T) {
	p := newExtractorProducer(t, true)
	path := filepath.Join(t.TempDir(), "endpoints.yaml")
	writeEndpoints(t, path, `endpoints:
- {name: vllm-0, address: 10.0.0.1, port: "8000"}
- {name: vllm-1, address: 10.0.0.2, port: "8000"}
`)
	startFileDiscovery(t, p, path, true)

	ids, _ := p.subscribersManager.GetActiveSubscribers()
	require.Len(t, ids, 2)

	// Ready() closes before the watcher is registered, so a single write can
	// land before watcher.Add and be missed. Rewrite on each poll (idempotent).
	require.Eventually(t, func() bool {
		writeEndpoints(t, path, `endpoints:
- {name: vllm-0, address: 10.0.0.1, port: "8000"}
`)
		ids, endpoints := p.subscribersManager.GetActiveSubscribers()
		return len(ids) == 1 && ids[0] == "default/vllm-0" && endpoints[0] == "tcp://10.0.0.1:5557"
	}, 5*time.Second, 20*time.Millisecond)
}
