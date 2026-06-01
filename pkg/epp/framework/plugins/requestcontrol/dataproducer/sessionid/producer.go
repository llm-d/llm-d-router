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

// Package sessionid provides a DataProducer that extracts a session
// identifier from a configured request header or named cookie and publishes
// it as a SessionID attribute on the InferenceRequest attribute store, so
// that affinity-aware scorers and filters can read it without knowing how
// the session was carried on the wire.
package sessionid

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrsession "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/session"
	sessionidconstants "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/sessionid/constants"
)

// SessionIDProducerType is the plugin type registered with the framework.
const SessionIDProducerType = sessionidconstants.SessionIDProducerType

// cookieHeader is the standard HTTP request header carrying cookies.
// Headers in InferenceRequest are normalized to lowercase.
const cookieHeader = "cookie"

// Parameters configures the session-id producer.
//
// Exactly one of HeaderName or CookieName must be set:
//   - HeaderName: read the value of the named request header verbatim.
//   - CookieName: parse the standard "cookie" request header and read the
//     value of the named cookie.
type Parameters struct {
	HeaderName string `json:"headerName"`
	CookieName string `json:"cookieName"`
}

var _ requestcontrol.DataProducer = &Producer{}

// Producer extracts a session identifier from each incoming request and
// publishes it as an endpoint attribute.
type Producer struct {
	typedName  fwkplugin.TypedName
	dk         fwkplugin.DataKey
	headerName string
	cookieName string
}

// Factory builds a Producer from raw plugin parameters.
func Factory(name string, rawParameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params := Parameters{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' producer: %w", SessionIDProducerType, err)
		}
	}

	header := strings.ToLower(strings.TrimSpace(params.HeaderName))
	cookie := strings.TrimSpace(params.CookieName)

	switch {
	case header == "" && cookie == "":
		return nil, fmt.Errorf("'%s' requires exactly one of headerName or cookieName to be set", SessionIDProducerType)
	case header != "" && cookie != "":
		return nil, fmt.Errorf("'%s' requires exactly one of headerName or cookieName to be set, not both", SessionIDProducerType)
	}

	return &Producer{
		typedName:  fwkplugin.TypedName{Type: SessionIDProducerType, Name: name},
		dk:         attrsession.SessionIDDataKey.WithNonEmptyProducerName(name),
		headerName: header,
		cookieName: cookie,
	}, nil
}

// TypedName returns the type and name of the plugin.
func (p *Producer) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// Produces declares the SessionID attribute key written by this producer.
func (p *Producer) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{p.dk: attrsession.SessionID("")}
}

// Produce extracts the session identifier from the request and writes it to
// the request's attribute store. When no identifier can be found the
// producer is a no-op; consumers must handle absence as "no session
// preference".
func (p *Producer) Produce(_ context.Context, request *fwksched.InferenceRequest, _ []fwksched.Endpoint) error {
	if request == nil {
		return nil
	}
	id := p.extract(request)
	if id == "" {
		return nil
	}
	request.PutAttribute(p.dk.String(), attrsession.SessionID(id))
	return nil
}

func (p *Producer) extract(request *fwksched.InferenceRequest) string {
	if request == nil || request.Headers == nil {
		return ""
	}
	if p.headerName != "" {
		return strings.TrimSpace(request.Headers[p.headerName])
	}
	return strings.TrimSpace(cookieValue(request.Headers[cookieHeader], p.cookieName))
}

// cookieValue returns the value of the named cookie within an HTTP Cookie
// header, or the empty string if the header is empty or the cookie is not
// present. The header is parsed verbatim per RFC 6265 syntax: cookies are
// separated by "; " and each pair is "name=value".
func cookieValue(header, name string) string {
	if header == "" || name == "" {
		return ""
	}
	for pair := range strings.SplitSeq(header, ";") {
		pair = strings.TrimSpace(pair)
		k, v, ok := strings.Cut(pair, "=")
		if !ok {
			continue
		}
		if k == name {
			return v
		}
	}
	return ""
}
