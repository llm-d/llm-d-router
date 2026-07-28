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
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"reflect"
	"runtime/debug"
	"slices"
	"sync"
	"time"

	"github.com/go-zeromq/zmq4"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

const ZMQDataSourceType = "zmq-metrics-data-source"

const defaultZMQPort = "5556"

var ErrExtractorTypeMismatch = errors.New("extractor type mismatch")

type zmqDatasourceParams struct {
	Port string `json:"port"`
}

// ZMQDataSource is a typed streaming dispatcher that connects to a model server's
// ZMQ publisher and dispatches incoming byte frames to bound extractors.
type ZMQDataSource struct {
	typedName fwkplugin.TypedName
	port      string

	mu   sync.RWMutex
	exts []fwkdl.StreamingExtractor[[]byte]
}

var (
	_ fwkdl.StreamingDispatcher = (*ZMQDataSource)(nil)
	_ fwkdl.DataSource          = (*ZMQDataSource)(nil)
)

// NewZMQDataSource returns a new ZMQDataSource instance.
func NewZMQDataSource(port, name string) (*ZMQDataSource, error) {
	if name == "" {
		name = ZMQDataSourceType
	}
	if port == "" {
		port = defaultZMQPort
	}
	return &ZMQDataSource{
		typedName: fwkplugin.TypedName{Type: ZMQDataSourceType, Name: name},
		port:      port,
	}, nil
}

// ZMQDataSourceFactory is a factory function used to instantiate ZMQ data source plugins.
func ZMQDataSourceFactory(name string, parameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	cfg := &zmqDatasourceParams{Port: defaultZMQPort}
	if parameters != nil {
		if err := parameters.Decode(cfg); err != nil {
			return nil, err
		}
	}
	return NewZMQDataSource(cfg.Port, name)
}

func (s *ZMQDataSource) TypedName() fwkplugin.TypedName {
	return s.typedName
}

// Start connects to the endpoint's ZMQ publisher and streams incoming messages to extractors.
// If the connection drops or fails, it will attempt to reconnect with exponential backoff.
func (s *ZMQDataSource) Start(ctx context.Context, ep fwkdl.Endpoint) error {
	ip := ep.GetMetadata().GetIPAddress()
	if ip == "" {
		ip = ep.GetMetadata().Address
	}
	address := "tcp://" + net.JoinHostPort(ip, s.port)
	logger := log.FromContext(ctx).WithValues("endpoint", ip)

	backoff := 1 * time.Second
	maxBackoff := 30 * time.Second

	for {
		if ctx.Err() != nil {
			return nil
		}

		if err := s.stream(ctx, address, ep); err != nil {
			if errors.Is(err, context.Canceled) || ctx.Err() != nil {
				return nil
			}
			logger.Error(err, "ZMQ streaming error, retrying", "backoff", backoff)

			select {
			case <-ctx.Done():
				return nil
			case <-time.After(backoff):
				backoff *= 2
				if backoff > maxBackoff {
					backoff = maxBackoff
				}
			}
		} else {
			backoff = 1 * time.Second
		}
	}
}

func (s *ZMQDataSource) stream(ctx context.Context, address string, ep fwkdl.Endpoint) error {
	sub := zmq4.NewSub(ctx)
	defer sub.Close()

	if err := sub.Dial(address); err != nil {
		return fmt.Errorf("zmq dial %s: %w", address, err)
	}

	if err := sub.SetOption(zmq4.OptionSubscribe, ""); err != nil {
		return fmt.Errorf("zmq subscribe: %w", err)
	}

	for {
		msg, err := sub.Recv()
		if err != nil {
			return err
		}

		var payload []byte
		if len(msg.Frames) > 0 {
			payload = msg.Frames[len(msg.Frames)-1]
		}

		in := fwkdl.StreamInput[[]byte]{
			Payload:  payload,
			Endpoint: ep,
		}

		s.mu.RLock()
		exts := slices.Clone(s.exts)
		s.mu.RUnlock()

		for _, ext := range exts {
			if ctx.Err() != nil {
				return nil
			}
			s.runExtractor(ctx, ext, in)
		}
	}
}

func (s *ZMQDataSource) runExtractor(ctx context.Context, ext fwkdl.StreamingExtractor[[]byte], in fwkdl.StreamInput[[]byte]) {
	logger := log.FromContext(ctx)
	srcType := s.typedName.Type
	extType := ext.TypedName().Type
	defer func() {
		if r := recover(); r != nil {
			metrics.RecordDataLayerExtractError(srcType, extType)
			logger.Error(fmt.Errorf("%v", r), "extractor panicked",
				"source", s.typedName, "extractor", ext.TypedName(), "stack", string(debug.Stack()))
		}
	}()
	if err := ext.Extract(ctx, in); err != nil {
		metrics.RecordDataLayerExtractError(srcType, extType)
		logger.V(logging.DEBUG).Info("extract failed", "source", s.typedName, "extractor", ext.TypedName(), "err", err)
	}
}

// AppendExtractor binds ext as a typed StreamingExtractor[[]byte].
func (s *ZMQDataSource) AppendExtractor(ext fwkplugin.Plugin) error {
	typed, ok := ext.(fwkdl.StreamingExtractor[[]byte])
	if !ok {
		return fmt.Errorf("%w: extractor %s: expected %s, got %T",
			ErrExtractorTypeMismatch, ext.TypedName(), reflect.TypeFor[fwkdl.StreamingExtractor[[]byte]](), ext)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.exts = append(s.exts, typed)
	return nil
}
