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

// Package headercopy provides a response-header processor that copies a
// configured list of request headers onto the response, under the same
// names, for a coordinator to read what EPP plugins set on the request it
// forwarded.
package headercopy

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/go-logr/logr"
	"k8s.io/apimachinery/pkg/util/sets"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/routing"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	requtil "github.com/llm-d/llm-d-router/pkg/epp/util/request"
)

// PluginType is the plugin type for the request header copy handler.
const PluginType = "request-header-copy-handler"

type parameters struct {
	// Headers lists the request header names to copy onto the response. Each
	// one is copied when present on the request, under the same name; one
	// named by its current spelling, which EPP also writes under a deprecated
	// alias, is copied under both spellings. A system-owned name is rejected:
	// the director drops it from the response header mutation, so the entry
	// would copy nothing. The list is explicit: the request header map holds
	// every client header, so a copy of all of them would echo client
	// credentials onto the response.
	Headers []string `json:"headers"`
}

var _ fwkrc.ResponseHeaderProcessor = &Handler{}

// Factory creates a request header copy handler.
func Factory(name string, rawParameters *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params := parameters{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' handler - %w", PluginType, err)
		}
	}
	if len(params.Headers) == 0 {
		return nil, fmt.Errorf("invalid configuration for '%s' handler: headers must list at least one request header name", PluginType)
	}
	headers := make([]string, 0, len(params.Headers))
	seen := sets.New[string]()
	var duplicates []string
	for _, header := range params.Headers {
		header = strings.ToLower(strings.TrimSpace(header))
		if header == "" {
			return nil, fmt.Errorf("invalid configuration for '%s' handler: headers must not contain an empty name", PluginType)
		}
		if requtil.IsSystemOwnedHeader(header) {
			return nil, fmt.Errorf("invalid configuration for '%s' handler: header %q is system-owned and is dropped from the response", PluginType, header)
		}
		if seen.Has(header) {
			duplicates = append(duplicates, header)
			continue
		}
		for _, spelling := range routing.HeaderNames(header) {
			if !seen.Has(spelling) {
				seen.Insert(spelling)
				headers = append(headers, spelling)
			}
		}
	}
	if name == "" {
		name = PluginType
	}
	if len(duplicates) > 0 {
		logger := logr.Discard()
		if handle != nil {
			logger = log.FromContext(handle.Context())
		}
		logger.Info("warning: duplicate header names in the configuration are copied once",
			"plugin", name, "duplicates", duplicates)
	}
	return &Handler{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: name},
		headers:   headers,
	}, nil
}

// Handler copies configured request headers onto the response headers.
type Handler struct {
	typedName fwkplugin.TypedName
	headers   []string
}

func (h *Handler) TypedName() fwkplugin.TypedName {
	return h.typedName
}

// ResponseHeader reads the request the PreRequest plugins mutated, so a header
// a plugin set reaches the response.
func (h *Handler) ResponseHeader(_ context.Context, request *fwksched.InferenceRequest, response *fwkrc.Response, _ *fwkdl.EndpointMetadata) {
	if response == nil || response.Headers == nil || request == nil || request.Headers == nil {
		return
	}
	for _, header := range h.headers {
		if value, ok := request.Headers[header]; ok {
			response.Headers[header] = value
		}
	}
}
