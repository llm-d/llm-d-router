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
	"fmt"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

const ZMQExtractorType = "zmq-state-extractor"

// Supported engine names for the extractor's payload decoder.
const (
	EngineSGLang = "sglang"
)

// extractFunc decodes an engine-specific payload and updates endpoint metrics.
type extractFunc func(ctx context.Context, in fwkdl.StreamInput[[]byte]) error

// engineExtractors maps an engine name to its payload decoder.
var engineExtractors = map[string]extractFunc{
	EngineSGLang: extractSGLang,
}

// zmqExtractorParams holds the configuration parameters for the ZMQ extractor plugin.
type zmqExtractorParams struct {
	// Engine selects the payload decoder. Required. Supported values: "sglang".
	Engine string `json:"engine"`
}

// Extractor implements ZMQ metrics extraction with an engine-specific payload decoder.
type Extractor struct {
	typedName fwkplugin.TypedName
	extract   extractFunc
}

var _ fwkdl.StreamingExtractor[[]byte] = (*Extractor)(nil)

// NewZMQMetricsExtractor returns a new ZMQ metrics extractor for the given engine.
func NewZMQMetricsExtractor(name, engine string) (*Extractor, error) {
	if name == "" {
		name = ZMQExtractorType
	}
	if engine == "" {
		return nil, fmt.Errorf("engine parameter is required for %s", ZMQExtractorType)
	}
	extract, ok := engineExtractors[engine]
	if !ok {
		return nil, fmt.Errorf("unsupported engine %q for %s", engine, ZMQExtractorType)
	}
	return &Extractor{
		typedName: fwkplugin.TypedName{
			Type: ZMQExtractorType,
			Name: name,
		},
		extract: extract,
	}, nil
}

// ZMQExtractorFactory is a factory function used to instantiate ZMQ extractor plugins.
func ZMQExtractorFactory(name string, parameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	cfg := &zmqExtractorParams{}
	if parameters != nil {
		if err := parameters.Decode(cfg); err != nil {
			return nil, err
		}
	}
	return NewZMQMetricsExtractor(name, cfg.Engine)
}

// TypedName returns the type and name of the Extractor.
func (ext *Extractor) TypedName() fwkplugin.TypedName {
	return ext.typedName
}

// Extract decodes the payload with the configured engine decoder and updates endpoint metrics.
func (ext *Extractor) Extract(ctx context.Context, in fwkdl.StreamInput[[]byte]) error {
	return ext.extract(ctx, in)
}
