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

package kv

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"strconv"
	"strings"

	"github.com/google/uuid"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// mooncakeKV uses a transfer ID shared by the two legs. The producer's
// completion has no transfer descriptor; the prefill EPP reports the worker
// metadata on the response using existing configurable header plugins.
type mooncakeKV struct{}

type mooncakePrefillMetadata struct {
	Endpoint      string `json:"endpoint"`
	EngineID      string `json:"engine_id"`
	BootstrapPort int    `json:"bootstrap_port"`
}

func (mooncakeKV) Name() string { return Mooncake }

func (mooncakeKV) PreparePrefillKVParams(_ context.Context, reqCtx *pipeline.RequestContext) map[string]any {
	handoff := &pipeline.MooncakeHandoff{TransferID: "xfer-" + uuid.NewString()}
	reqCtx.MooncakeHandoff = handoff
	return map[string]any{
		reqcommon.FieldDoRemotePrefill: false,
		reqcommon.FieldDoRemoteDecode:  true,
		reqcommon.FieldTransferID:      handoff.TransferID,
	}
}

func (mooncakeKV) ReadPrefillResponse(reqCtx *pipeline.RequestContext, headers http.Header) error {
	handoff := reqCtx.MooncakeHandoff
	if handoff == nil || handoff.TransferID == "" {
		return errors.New("missing Mooncake prefill transfer ID")
	}
	handoff.EngineID, handoff.BootstrapAddress = "", ""
	values := headers.Values(routing.MooncakePrefillHandoffHeader)
	if len(values) != 1 {
		return fmt.Errorf("prefill response must carry exactly one %s header", routing.MooncakePrefillHandoffHeader)
	}
	var metadata mooncakePrefillMetadata
	if err := json.Unmarshal([]byte(values[0]), &metadata); err != nil {
		return errors.New("invalid Mooncake prefill metadata")
	}
	host, port, ok := routing.ParseHostPort(metadata.Endpoint)
	if !ok || strings.TrimSpace(metadata.EngineID) == "" || !routing.IsTCPPort(metadata.BootstrapPort) {
		return errors.New("invalid Mooncake prefill endpoint, engine ID or bootstrap port")
	}
	// The stamp names the selected endpoint, which can differ from the
	// endpoint reported as served by the gateway.
	served := false
	for _, name := range routing.HeaderNames(routing.PrefillEndpointHeader) {
		for _, value := range headers.Values(name) {
			servedHost, servedPort, valid := routing.ParseHostPort(value)
			if !valid || servedHost != host || servedPort != port {
				return errors.New("prefill metadata does not match the served endpoint")
			}
			served = true
		}
	}
	if !served {
		return errors.New("prefill response is missing the served endpoint header")
	}
	handoff.EngineID = metadata.EngineID
	handoff.BootstrapAddress = "http://" + net.JoinHostPort(host, strconv.Itoa(metadata.BootstrapPort))
	return nil
}

func (mooncakeKV) PrepareDecodeKVParams(_ context.Context, reqCtx *pipeline.RequestContext) map[string]any {
	handoff := reqCtx.MooncakeHandoff
	if handoff == nil || handoff.TransferID == "" || handoff.EngineID == "" || handoff.BootstrapAddress == "" {
		return nil
	}
	reqCtx.MooncakeHandoff = nil
	return map[string]any{
		reqcommon.FieldDoRemotePrefill:     true,
		reqcommon.FieldDoRemoteDecode:      false,
		reqcommon.FieldTransferID:          handoff.TransferID,
		reqcommon.FieldRemoteBootstrapAddr: handoff.BootstrapAddress,
		reqcommon.FieldRemoteEngineID:      handoff.EngineID,
	}
}
