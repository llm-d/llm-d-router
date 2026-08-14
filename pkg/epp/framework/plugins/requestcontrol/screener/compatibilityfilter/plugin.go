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

package compatibilityfilter

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
)

const (
	PluginType = "compatibility-screener"
)

// Screener groups endpoints by a metrics-derived compatibility value and
// filters to one compatible group per request.
type Screener struct {
	typedName fwkplugin.TypedName
	config    Config
	dataKey   fwkplugin.DataKey

	// chosenValues caches the compatibility value chosen during Screen() so
	// that ResponseHeader() can stamp it. Keyed by request ID.
	mu           sync.Mutex
	chosenValues map[string]string
}

var (
	_ fwkplugin.Plugin              = (*Screener)(nil)
	_ fwkplugin.ConsumerPlugin      = (*Screener)(nil)
	_ fwkrc.Screener                = (*Screener)(nil)
	_ fwkrc.ResponseHeaderProcessor = (*Screener)(nil)
)

// Factory creates a compatibility-screener from plugin parameters.
func Factory(name string, parameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	if name == "" {
		name = PluginType
	}
	config := Config{}
	if parameters == nil {
		return nil, errors.New("compatibility-screener requires parameters")
	}
	if err := parameters.Decode(&config); err != nil {
		return nil, fmt.Errorf("decode compatibility-screener parameters: %w", err)
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	dataKey := attrmetrics.StringMetricDataKey(config.AttributeKey)
	return newScreener(name, config, dataKey), nil
}

func newScreener(
	name string,
	config Config,
	dataKey fwkplugin.DataKey,
) *Screener {
	return &Screener{
		typedName:    fwkplugin.TypedName{Type: PluginType, Name: name},
		config:       config,
		dataKey:      dataKey,
		chosenValues: make(map[string]string),
	}
}

func (s *Screener) TypedName() fwkplugin.TypedName { return s.typedName }

func (s *Screener) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{
			s.dataKey: attrmetrics.StringMetricValue(""),
		},
	}
}

// ResponseHeader stamps the chosen compatibility value into the response so
// downstream consumers (e.g. a coordinator forwarding to a decode EPP) can pin
// their request to the same value.
func (s *Screener) ResponseHeader(_ context.Context, request *fwksched.InferenceRequest, response *fwkrc.Response, _ *fwkdl.EndpointMetadata) {
	if request == nil || response == nil || response.Headers == nil {
		return
	}
	s.mu.Lock()
	value, ok := s.chosenValues[request.RequestID]
	if ok {
		delete(s.chosenValues, request.RequestID)
	}
	s.mu.Unlock()

	if ok && value != "" {
		response.Headers[s.config.HeaderName] = value
	}
}

// getCompatValue reads the compatibility value from an endpoint's attributes.
func (s *Screener) getCompatValue(endpoint fwksched.Endpoint) string {
	val, ok := endpoint.Get(s.dataKey)
	if !ok {
		return ""
	}
	if v, isStr := val.(attrmetrics.StringMetricValue); isStr {
		return string(v)
	}
	return ""
}
