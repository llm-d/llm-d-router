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

package zmqmetrics

import (
	"bytes"
	"context"
	"encoding/json"
	"net"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-zeromq/zmq4"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

type testStreamingExtractor struct {
	count   int32
	payload []byte
}

func (t *testStreamingExtractor) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "test-extractor", Name: "test-extractor"}
}

func (t *testStreamingExtractor) Extract(_ context.Context, in fwkdl.StreamInput[[]byte]) error {
	t.payload = in.Payload
	atomic.AddInt32(&t.count, 1)
	return nil
}

func TestZMQDataSourceFactory(t *testing.T) {
	t.Run("default port", func(t *testing.T) {
		p, err := ZMQDataSourceFactory("zmq-test", nil, nil)
		require.NoError(t, err)
		ds, ok := p.(*ZMQDataSource)
		require.True(t, ok)
		assert.Equal(t, defaultZMQPort, ds.port)
	})

	t.Run("custom port", func(t *testing.T) {
		params := []byte(`{"port": "9999"}`)
		dec := json.NewDecoder(bytes.NewReader(params))
		p, err := ZMQDataSourceFactory("zmq-test", dec, nil)
		require.NoError(t, err)
		ds, ok := p.(*ZMQDataSource)
		require.True(t, ok)
		assert.Equal(t, "9999", ds.port)
	})

	t.Run("unquoted numeric port", func(t *testing.T) {
		params := []byte(`{"port": 5557}`)
		dec := json.NewDecoder(bytes.NewReader(params))
		p, err := ZMQDataSourceFactory("zmq-test", dec, nil)
		require.NoError(t, err)
		ds, ok := p.(*ZMQDataSource)
		require.True(t, ok)
		assert.Equal(t, "5557", ds.port)
	})

	t.Run("invalid port", func(t *testing.T) {
		for _, port := range []string{"abc", "0", "-1", "65536", "55 56"} {
			params := []byte(`{"port": "` + port + `"}`)
			dec := json.NewDecoder(bytes.NewReader(params))
			_, err := ZMQDataSourceFactory("zmq-test", dec, nil)
			assert.Error(t, err, "port %q should be rejected", port)
		}
	})
}

func TestZMQDataSource_StartEmptyAddress(t *testing.T) {
	src, err := NewZMQDataSource("", "test-zmq")
	require.NoError(t, err)

	ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{}, nil)
	err = src.Start(context.Background(), ep)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no address")
}

func TestZMQDataSource_Start(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	l, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	port := strconv.Itoa(l.Addr().(*net.TCPAddr).Port)
	l.Close()

	pub := zmq4.NewPub(ctx)
	defer pub.Close()

	addr := "tcp://127.0.0.1:" + port
	require.NoError(t, pub.Listen(addr))

	src, err := NewZMQDataSource(port, "test-zmq")
	require.NoError(t, err)

	extractor := &testStreamingExtractor{}
	require.NoError(t, src.AppendExtractor(extractor))

	ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{Address: "127.0.0.1"}, nil)

	go func() {
		_ = src.Start(ctx, ep)
	}()

	time.Sleep(50 * time.Millisecond)

	msgData := []byte("zmq-payload-test")
	msg := zmq4.NewMsg(msgData)
	require.NoError(t, pub.Send(msg))

	require.Eventually(t, func() bool {
		return atomic.LoadInt32(&extractor.count) > 0
	}, 2*time.Second, 10*time.Millisecond)

	assert.Equal(t, msgData, extractor.payload)
}
