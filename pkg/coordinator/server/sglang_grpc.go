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

package server

import (
	"net/http"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/sglang/sglangwire"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

func scanSGLangGenerate(_ *http.Request, payload []byte) (grpcScanResult, error) {
	// A top-level wire scan replaces a full proto.Unmarshal. It drops client
	// bootstrap fields, and both legs then send genReq.Payload without a copy.
	// The SGLang worker validates nested fields.
	genReq, err := sglangwire.ScanGenerateRequest(payload)
	if err != nil {
		return grpcScanResult{}, err
	}
	return grpcScanResult{
		req:       &pipeline.SGLangGRPCRequest{Payload: genReq.Payload},
		stream:    true, // SglangService/Generate is always server-streaming on the wire.
		logStream: genReq.Stream,
	}, nil
}
