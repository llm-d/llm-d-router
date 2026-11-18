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

package preciseprefixcache

import (
	"context"
	"testing"

	"github.com/go-logr/logr"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
)

// newRegistrationProducer builds a producer with only what registration and Extract read, and a fake subscriber
// manager, so no KV-events pool or queue goroutines are started.
func newRegistrationProducer(discoverPods bool, zmqEndpoint string) (*Producer, *fakeSubscriberManager) {
	cfg := kvevents.DefaultConfig()
	cfg.DiscoverPods = discoverPods
	cfg.ZMQEndpoint = zmqEndpoint
	subscribers := &fakeSubscriberManager{}
	return &Producer{
		typedName:          plugin.TypedName{Type: PluginType, Name: PluginType},
		subscribersManager: subscribers,
		kvEventsConfig:     cfg,
		subscriberCtx:      context.Background(),
	}, subscribers
}

type recordingRegistrar struct {
	registrations []fwkdl.PendingRegistration
}

func (r *recordingRegistrar) Register(reg fwkdl.PendingRegistration) error {
	r.registrations = append(r.registrations, reg)
	return nil
}

func TestProducer_RegisterDependencies(t *testing.T) {
	t.Run("pod discovery registers the endpoint-notification source by default", func(t *testing.T) {
		p, _ := newRegistrationProducer(true, "")
		r := &recordingRegistrar{}
		require.NoError(t, p.RegisterDependencies(r))

		require.Len(t, r.registrations, 1)
		reg := r.registrations[0]
		assert.Equal(t, p.TypedName(), reg.Owner)
		assert.Equal(t, sourcenotifications.EndpointNotificationSourceType, reg.SourceType)
		assert.Same(t, p, reg.Extractor)
		require.NotNil(t, reg.DefaultSource)
		assert.Equal(t, sourcenotifications.EndpointNotificationSourceType, reg.DefaultSource.TypedName().Type)
	})

	t.Run("pod discovery disabled registers nothing", func(t *testing.T) {
		p, _ := newRegistrationProducer(false, "")
		r := &recordingRegistrar{}
		require.NoError(t, p.RegisterDependencies(r))
		assert.Empty(t, r.registrations)
	})

	t.Run("pod discovery without a discovery config registers nothing", func(t *testing.T) {
		p, _ := newRegistrationProducer(true, "")
		p.kvEventsConfig.PodDiscoveryConfig = nil
		r := &recordingRegistrar{}
		require.NoError(t, p.RegisterDependencies(r))
		assert.Empty(t, r.registrations)
	})

	t.Run("global socket with the default discoverPods registers nothing", func(t *testing.T) {
		p, _ := newRegistrationProducer(true, "tcp://0.0.0.0:5557")
		r := &recordingRegistrar{}
		require.NoError(t, p.RegisterDependencies(r))
		assert.Empty(t, r.registrations)
	})
}

func TestProducer_RuntimeSubscribesPodsThroughRegistration(t *testing.T) {
	ctx := discardCtx(t)
	for _, tc := range []struct {
		name         string
		zmqEndpoint  string
		otherSource  bool
		wireExplicit bool
		want         int
	}{
		{name: "pull mode without an endpoint-notification source", want: 1},
		{name: "pull mode with the source listed but not wired", otherSource: true, want: 1},
		{name: "pull mode with explicit wiring", wireExplicit: true, want: 1},
		{name: "global socket with the source configured for other plugins", zmqEndpoint: "tcp://0.0.0.0:5557", otherSource: true, want: 0},
		{name: "global socket with explicit wiring", zmqEndpoint: "tcp://0.0.0.0:5557", wireExplicit: true, want: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, subscribers := newRegistrationProducer(true, tc.zmqEndpoint)

			rt := datalayer.NewRuntime(0)
			require.NoError(t, p.RegisterDependencies(rt))
			var cfg *datalayer.Config
			if tc.otherSource || tc.wireExplicit {
				// A source name distinct from its type, so the default registration resolves the source by type.
				src := datalayer.DataSourceConfig{
					Plugin: sourcenotifications.NewEndpointDataSource(sourcenotifications.EndpointNotificationSourceType, "endpoints"),
				}
				if tc.wireExplicit {
					src.Extractors = []plugin.Plugin{p}
				}
				cfg = &datalayer.Config{Sources: []datalayer.DataSourceConfig{src}}
			}
			require.NoError(t, rt.Configure(cfg, logr.Discard()))

			rt.NewEndpoint(ctx, newEndpoint("pod-a", "10.0.0.1").GetMetadata())
			// Each EnsureSubscriber call is recorded, so a producer wired twice would show two calls.
			assert.Len(t, subscribers.ids, tc.want)
		})
	}
}
