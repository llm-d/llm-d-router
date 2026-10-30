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

// Package dprankheader provides a PreRequest plugin that writes the selected
// endpoint's data-parallel rank to a request header, for model servers that
// serve several ranks behind one port and dispatch on that header.
package dprankheader

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/llm-d/llm-d-router/pkg/common/routing"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/util/request"
)

// PluginType is the plugin type for the data-parallel rank header writer.
const PluginType = "data-parallel-rank-header"

type parameters struct {
	// HeaderName is the request header the rank is written to.
	// Defaults to routing.DataParallelRankHeader.
	HeaderName string `json:"headerName,omitempty"`
}

// Factory creates a data-parallel rank header writer.
func Factory(name string, rawParameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params := parameters{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' plugin: %w", PluginType, err)
		}
	}
	headerName := strings.ToLower(params.HeaderName)
	if headerName == "" {
		headerName = routing.DataParallelRankHeader
	}
	if request.IsSystemOwnedHeader(headerName) {
		return nil, fmt.Errorf("invalid configuration for '%s' plugin: headerName %q is EPP-owned and stripped on the way out", PluginType, params.HeaderName)
	}
	if request.InternalRoutingHeaders.Has(headerName) {
		return nil, fmt.Errorf("invalid configuration for '%s' plugin: headerName %q is an internal routing header", PluginType, params.HeaderName)
	}
	if name == "" {
		name = PluginType
	}
	return &Handler{
		typedName:  fwkplugin.TypedName{Type: PluginType, Name: name},
		headerName: headerName,
	}, nil
}

var _ fwkrc.PreRequest = &Handler{}

// Handler writes the selected endpoint's rank to a request header.
type Handler struct {
	typedName  fwkplugin.TypedName
	headerName string
}

func (h *Handler) TypedName() fwkplugin.TypedName { return h.typedName }

// PreRequest sets the header to the RankIndex of the primary profile's selected
// endpoint, replacing any client value. It no-ops when no endpoint was selected
// or the endpoint is the only rank on its port.
func (h *Handler) PreRequest(_ context.Context, req *fwksched.InferenceRequest, schedulingResult *fwksched.SchedulingResult) error {
	if req == nil || schedulingResult == nil {
		return nil
	}
	result := schedulingResult.ProfileResults[schedulingResult.PrimaryProfileName]
	if result == nil || len(result.TargetEndpoints) == 0 || result.TargetEndpoints[0] == nil {
		return nil
	}
	meta := result.TargetEndpoints[0].GetMetadata()
	if meta == nil || meta.RanksPerPort < 2 {
		return nil
	}
	if req.Headers == nil {
		req.Headers = map[string]string{}
	}
	req.Headers[h.headerName] = strconv.Itoa(meta.DataParallelRank)
	return nil
}
