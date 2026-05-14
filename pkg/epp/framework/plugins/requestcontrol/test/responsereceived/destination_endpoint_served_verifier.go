/*
Copyright 2025 The Kubernetes Authors.

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

package responsereceived

import (
	"context"
	"encoding/json"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-inference-scheduler/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/requestcontrol/test"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/metadata"
)

const (
	// DestinationEndpointServedVerifierType is the ResponseReceived type that is used in plugins registry.
	DestinationEndpointServedVerifierType = "destination-endpoint-served-verifier"
)

var _ requestcontrol.ResponseHeaderProcessor = &DestinationEndpointServedVerifier{}

// DestinationEndpointServedVerifier is a test-only plugin for conformance tests.
// It verifies that the request was served by the expected endpoint.
// It works by reading Envoy's dynamic metadata, which is passed in the
// Response.ReqMetadata field. This metadata should contain the
// address of the backend endpoint that served the request. The verifier then
// writes this address to the "x-conformance-test-served-endpoint" response header.
// The conformance test client can then validate this header to ensure the request
// was routed correctly.
type DestinationEndpointServedVerifier struct {
	typedName fwkplugin.TypedName
}

// TypedName returns the type and name tuple of this plugin instance.
func (desv *DestinationEndpointServedVerifier) TypedName() fwkplugin.TypedName {
	return desv.typedName
}

// WithName sets the name of the filter.
func (desv *DestinationEndpointServedVerifier) WithName(name string) *DestinationEndpointServedVerifier {
	desv.typedName.Name = name
	return desv
}

// DestinationEndpointServedVerifierFactory defines the factory function for DestinationEndpointServedVerifier.
func DestinationEndpointServedVerifierFactory(name string, _ *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	return NewDestinationEndpointServedVerifier().WithName(name), nil
}

func NewDestinationEndpointServedVerifier() *DestinationEndpointServedVerifier {
	return &DestinationEndpointServedVerifier{}
}

// ResponseHeader is the handler for the ResponseHeader extension point.
func (desv *DestinationEndpointServedVerifier) ResponseHeader(ctx context.Context, request *fwksched.InferenceRequest, response *requestcontrol.Response, _ *fwkdl.EndpointMetadata) {
	logger := log.FromContext(ctx).WithName(desv.TypedName().String())
	logger.V(logging.DEBUG).Info("Verifying destination endpoint served")

	reqMetadata := response.ReqMetadata
	lbMetadata, ok := reqMetadata[metadata.DestinationEndpointNamespace].(map[string]any)
	if !ok {
		logger.V(logging.DEBUG).Info("Response does not contain envoy lb metadata, skipping verification")
		response.Headers[test.ConformanceTestResultHeader] = "fail: missing envoy lb metadata"
		return
	}

	actualEndpoint, ok := lbMetadata[metadata.DestinationEndpointServedKey].(string)
	if !ok {
		logger.V(logging.DEBUG).Info("Response does not contain destination endpoint served metadata, skipping verification")
		response.Headers[test.ConformanceTestResultHeader] = "fail: missing destination endpoint served metadata"
		return
	}
	response.Headers[test.ConformanceTestResultHeader] = actualEndpoint
}
