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

package file

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

const clusterYAML = `
endpoints:
  - name: cluster-a
    address: "gw.cluster-a.example.com"
    port: "443"
`

func TestMultiClusterFactory_Type(t *testing.T) {
	path := writeTemp(t, clusterYAML)
	p, err := MultiClusterFactory("mc", fwkplugin.StrictDecoder(json.RawMessage(`{"path":"`+path+`"}`)), nil)
	require.NoError(t, err)
	assert.Equal(t, MultiClusterPluginType, p.TypedName().Type)
	assert.Equal(t, "mc", p.TypedName().Name)
}

// A cluster gateway hostname is accepted and its MetricsHost is host:port.
func TestMultiClusterFactory_AcceptsHostname(t *testing.T) {
	path := writeTemp(t, clusterYAML)
	notifier := &recordingNotifier{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	p, err := MultiClusterFactory("mc", fwkplugin.StrictDecoder(json.RawMessage(`{"path":"`+path+`"}`)), nil)
	require.NoError(t, err)
	require.NoError(t, p.(fwkdl.EndpointDiscovery).Start(ctx, notifier))

	require.Len(t, notifier.upserted, 1)
	assert.Equal(t, "gw.cluster-a.example.com:443", notifier.upserted[0].MetricsHost)
}

// metricsAddress/metricsPort labels move the scrape target off the routable address, so a
// peer can serve pool metrics on a different host or port than its gateway.
func TestMetricsHostFromLabels(t *testing.T) {
	tests := []struct {
		name   string
		labels map[string]string
		want   string
	}{
		{name: "both labels set the metrics host", labels: map[string]string{"metricsAddress": "10.0.0.11", "metricsPort": "9090"}, want: "10.0.0.11:9090"},
		{name: "metricsAddress alone defaults to the endpoint port", labels: map[string]string{"metricsAddress": "10.0.0.11"}, want: "10.0.0.11:8080"},
		{name: "metricsPort alone is ignored", labels: map[string]string{"metricsPort": "9090"}, want: "10.0.0.12:8080"},
		{name: "no labels keep the endpoint address", labels: nil, want: "10.0.0.12:8080"},
		{name: "hostname metrics address", labels: map[string]string{"metricsAddress": "gw.example.com", "metricsPort": "443"}, want: "gw.example.com:443"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			meta := &fwkdl.EndpointMetadata{Address: "10.0.0.12", Port: "8080", MetricsHost: "10.0.0.12:8080", Labels: tc.labels}
			metricsHostFromLabels(meta)
			assert.Equal(t, tc.want, meta.MetricsHost)
		})
	}
}

// The wrapper applies the label resolution on the upsert path, so it reaches the datastore.
func TestMultiClusterFactory_MetricsHostFromLabels(t *testing.T) {
	const yaml = `
endpoints:
  - name: split
    address: "10.0.0.12"
    port: "8080"
    labels: { metricsAddress: "10.0.0.11", metricsPort: "9090" }
`
	path := writeTemp(t, yaml)
	notifier := &recordingNotifier{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	p, err := MultiClusterFactory("mc", fwkplugin.StrictDecoder(json.RawMessage(`{"path":"`+path+`"}`)), nil)
	require.NoError(t, err)
	require.NoError(t, p.(fwkdl.EndpointDiscovery).Start(ctx, notifier))

	require.Len(t, notifier.upserted, 1)
	assert.Equal(t, "10.0.0.11:9090", notifier.upserted[0].MetricsHost)
}

// The stock pod discovery still rejects a hostname.
func TestFileDiscovery_StockRejectsHostname(t *testing.T) {
	path := writeTemp(t, clusterYAML)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := newFD(path, false).Start(ctx, &recordingNotifier{})
	require.Error(t, err)
	assert.ErrorContains(t, err, "invalid IPv4 address")
}

func TestValidateIPv4Address(t *testing.T) {
	tests := []struct {
		name    string
		address string
		wantErr bool
	}{
		{name: "ipv4 ok", address: "10.0.0.1"},
		{name: "hostname rejected", address: "gw.example.com", wantErr: true},
		{name: "ipv6 rejected", address: "::1", wantErr: true},
		{name: "empty rejected", address: "", wantErr: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := validateIPv4Address(tc.address)
			if tc.wantErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}
		})
	}
}

func TestValidateHostOrIP(t *testing.T) {
	tests := []struct {
		name    string
		address string
		wantErr bool
	}{
		{name: "ipv4 ok", address: "10.0.0.1"},
		{name: "ipv6 ok", address: "::1"},
		{name: "dns hostname ok", address: "gw.cluster-a.example.com"},
		{name: "single label ok", address: "gateway"},
		{name: "trailing dot rejected", address: "gw.example.com.", wantErr: true},
		{name: "uppercase rejected", address: "GW.example.com", wantErr: true},
		{name: "digit-led label ok", address: "3com.example.com"},
		{name: "empty rejected", address: "", wantErr: true},
		{name: "underscore rejected", address: "gw_1.example.com", wantErr: true},
		{name: "numeric only accepted", address: "12345"},
		{name: "host with colon rejected", address: "gw:443", wantErr: true},
		{name: "control char rejected", address: "gw\x00.example.com", wantErr: true},
		{name: "at sign rejected", address: "user@host", wantErr: true},
		{name: "leading hyphen label rejected", address: "-bad.example.com", wantErr: true},
		{name: "empty label rejected", address: "gw..example.com", wantErr: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := validateHostOrIP(tc.address)
			if tc.wantErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}
		})
	}
}
