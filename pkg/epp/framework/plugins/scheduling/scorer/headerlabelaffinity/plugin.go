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

// Package headerlabelaffinity provides a scorer that prefers endpoints whose
// configured label matches a request header.
package headerlabelaffinity

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const PluginType = "header-label-affinity-scorer"

type parameters struct {
	HeaderName string `json:"headerName"`
	LabelKey   string `json:"labelKey"`
}

// Factory creates a scorer for one request-header-to-endpoint-label mapping.
func Factory(name string, rawParameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	if rawParameters == nil {
		return nil, errors.New("header-label-affinity-scorer requires parameters")
	}
	params := parameters{}
	if err := rawParameters.Decode(&params); err != nil {
		return nil, fmt.Errorf("decode header-label-affinity-scorer parameters: %w", err)
	}
	if params.HeaderName == "" {
		return nil, errors.New("header-label-affinity-scorer headerName is required")
	}
	if params.LabelKey == "" {
		return nil, errors.New("header-label-affinity-scorer labelKey is required")
	}
	if name == "" {
		name = PluginType
	}
	return &Scorer{
		typedName:  fwkplugin.TypedName{Type: PluginType, Name: name},
		headerName: strings.ToLower(params.HeaderName),
		labelKey:   params.LabelKey,
	}, nil
}

// Scorer gives endpoints whose label matches the request header a soft
// affinity score without removing non-matching endpoints.
type Scorer struct {
	typedName  fwkplugin.TypedName
	headerName string
	labelKey   string
}

var _ fwksched.Scorer = (*Scorer)(nil)

func (s *Scorer) TypedName() fwkplugin.TypedName { return s.typedName }

func (s *Scorer) Category() fwksched.ScorerCategory { return fwksched.Affinity }

func (s *Scorer) Score(_ context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	scores := make(map[fwksched.Endpoint]float64, len(endpoints))
	requested := ""
	if request != nil {
		requested = request.Headers[s.headerName]
	}
	for _, endpoint := range endpoints {
		scores[endpoint] = 0
		if requested == "" || endpoint == nil || endpoint.GetMetadata() == nil {
			continue
		}
		if endpoint.GetMetadata().Labels[s.labelKey] == requested {
			scores[endpoint] = 1
		}
	}
	return scores
}
