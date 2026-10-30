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

package steps

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/usage"

	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/kv"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

const DecodeStepName = "decode"

// maxDecodeWrapSize bounds a decode response read whole into memory for the
// chat completions wrap. Generous for a completion body while still bounding a
// runaway or adversarial worker.
const maxDecodeWrapSize = 8 << 20 // 8 MB

func init() {
	pipeline.Register(DecodeStepName, NewDecodeStep)
}

type DecodeStep struct {
	useOpenAIFormat    bool
	outputMode         string
	wrapChatCompletion bool
	wrapResponses      bool
	wrapMessages       bool
	gwClient           *gateway.Client
	kv                 kv.Connector
}

func NewDecodeStep(gwClient *gateway.Client, params map[string]any) (pipeline.Step, error) {
	if gwClient == nil {
		return nil, errors.New("decode: gateway client is required")
	}
	useOpenAI, err := parseUseOpenAIFormat(params)
	if err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	outputMode, err := parseOutputMode(params)
	if err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	wrapChat, _, err := paramBool(params, ParamWrapChatCompletion)
	if err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	wrapResp, _, err := paramBool(params, ParamWrapResponses)
	if err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	wrapMsg, _, err := paramBool(params, ParamWrapMessages)
	if err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	if outputMode != reqcommon.OutputModeText {
		switch {
		case wrapResp && wrapMsg:
			return nil, fmt.Errorf("decode: %s and %s require output_mode %q", ParamWrapResponses, ParamWrapMessages, reqcommon.OutputModeText)
		case wrapResp:
			return nil, fmt.Errorf("decode: %s requires output_mode %q", ParamWrapResponses, reqcommon.OutputModeText)
		case wrapMsg:
			return nil, fmt.Errorf("decode: %s requires output_mode %q", ParamWrapMessages, reqcommon.OutputModeText)
		}
	}
	kvConn, err := buildKVConnector(params)
	if err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	return &DecodeStep{
		useOpenAIFormat:    useOpenAI,
		outputMode:         outputMode,
		wrapChatCompletion: wrapChat,
		wrapResponses:      wrapResp,
		wrapMessages:       wrapMsg,
		gwClient:           gwClient,
		kv:                 kvConn,
	}, nil
}

func (s *DecodeStep) Name() string { return DecodeStepName }

func (s *DecodeStep) Execute(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	logger := log.FromContext(ctx).WithName(DecodeStepName)

	format := resolveFormat(s.useOpenAIFormat, reqCtx.OriginalPath)
	body, err := s.prepareDecodeBody(ctx, reqCtx, format)
	if err != nil {
		return err
	}

	// The tokens-in format is served on the generate route whatever the client
	// called, so the leg is addressed by the format rather than by the client's
	// own path. On an OpenAI-format leg the two are the same path.
	path := reqCtx.OriginalPath
	if format == reqcommon.APITypeVLLMGenerate {
		path = reqcommon.PathVLLMGenerate
	}

	logger.V(logutil.DEFAULT).Info("sending request", "path", path, "format", format, "stream", reqCtx.Stream)

	if s.wrapsChatCompletion(reqCtx, format) {
		return s.serveWrapped(ctx, logger, reqCtx, path, body, func(b []byte, id, _, mode string) ([]byte, error) {
			return wrapChatCompletion(b, id, mode)
		}, "chat completion")
	}
	if s.wrapsResponses(reqCtx, format) {
		return s.serveWrapped(ctx, logger, reqCtx, path, body, wrapResponses, "responses")
	}
	if s.wrapsMessages(reqCtx, format) {
		return s.serveWrapped(ctx, logger, reqCtx, path, body, wrapMessages, "messages")
	}

	proxyReq, err := newDecodeProxyRequest(ctx, logger, DecodeStepName, reqCtx, s.gwClient, path, body, nil)
	if err != nil {
		return err
	}

	writer, finalize := s.responseWriter(reqCtx)
	out := serveDecode(logger, s.gwClient.Transport(), writer, proxyReq, coordmetrics.UpstreamDecode, nil)
	if err := finalize(); err != nil {
		// The buffered tail could not be forwarded, so the client holds a
		// truncated response. Nothing can be written to say so: the status and
		// the body before the tail are already on the wire.
		logger.Error(err, "flushing the rewritten usage block failed: client received a partial response")
	}
	return out.streamedError(DecodeStepName)
}

// wrapsChatCompletion reports whether this leg's response needs the chat
// completions envelope rebuilt around it.
//
// Three conditions hold together. The client must have called
// /v1/chat/completions, since that is the only envelope this step builds. The
// leg must have been rewritten to the generate route, as an OpenAI-format leg
// already answers in the client's own shape. And the request must not stream:
// the wrap rewrites one complete JSON document, while a streamed answer is a
// sequence of chat.completion.chunk frames this step does not assemble.
func (s *DecodeStep) wrapsChatCompletion(reqCtx *pipeline.RequestContext, format reqcommon.APIType) bool {
	return s.wrapChatCompletion &&
		format == reqcommon.APITypeVLLMGenerate &&
		reqcommon.DetectAPIType(reqCtx.OriginalPath) == reqcommon.APITypeChatCompletions &&
		!reqCtx.Stream
}

func (s *DecodeStep) wrapsResponses(reqCtx *pipeline.RequestContext, format reqcommon.APIType) bool {
	return s.wrapResponses &&
		format == reqcommon.APITypeVLLMGenerate &&
		reqcommon.DetectAPIType(reqCtx.OriginalPath) == reqcommon.APITypeResponses &&
		!reqCtx.Stream
}

func (s *DecodeStep) wrapsMessages(reqCtx *pipeline.RequestContext, format reqcommon.APIType) bool {
	return s.wrapMessages &&
		format == reqcommon.APITypeVLLMGenerate &&
		reqcommon.DetectAPIType(reqCtx.OriginalPath) == reqcommon.APITypeMessages &&
		!reqCtx.Stream
}

type responseWrapper func(generate []byte, requestID, model, wantMode string) ([]byte, error)

// serveWrapped sends the decode leg as a plain call, rewrites the
// generate response it returns into the client API envelope using wrap, and writes that to
// the client. The leg is not proxied: the whole body is needed before any of it
// can be rewritten.
//
// maxDecodeWrapSize bounds what is read, since this holds the response in
// memory where the proxying path streams it.
func (s *DecodeStep) serveWrapped(ctx context.Context, logger logr.Logger, reqCtx *pipeline.RequestContext, path string, body map[string]any, wrap responseWrapper, kind string) error {
	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("%s: marshal: %w", DecodeStepName, err)
	}

	resp, err := postToGateway(ctx, logger, s.gwClient, gatewayRequest{
		logMsg:   "request body",
		step:     DecodeStepName,
		upstream: coordmetrics.UpstreamDecode,
		path:     path,
		body:     bodyBytes,
		headers:  gatewayHeaders(reqCtx, gateway.PhaseDecode),
	})
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	generate, err := io.ReadAll(io.LimitReader(resp.Body, maxDecodeWrapSize+1))
	if err != nil {
		return fmt.Errorf("%s: reading response: %w", DecodeStepName, err)
	}
	if len(generate) > maxDecodeWrapSize {
		return fmt.Errorf("%s: response exceeds %d bytes", DecodeStepName, maxDecodeWrapSize)
	}
	reqCtx.CaptureResponseHeaders(resp.Header)

	wrapped, err := wrap(generate, reqCtx.RequestID, reqCtx.Model, s.outputMode)
	if err != nil {
		return fmt.Errorf("%s: %w", DecodeStepName, err)
	}

	w := reqCtx.ResponseWriter
	w.Header().Set(reqcommon.HeaderContentType, reqcommon.ContentTypeJSON)
	if _, err := w.Write(wrapped); err != nil {
		return fmt.Errorf("%s: writing response: %w", DecodeStepName, err)
	}
	logger.V(logutil.DEFAULT).Info("wrapped generate response", "kind", kind, "bodyLen", len(wrapped))
	return nil
}

// responseWriter returns the writer the decode response is streamed through,
// and the function that flushes whatever it buffered. Without a prefill count
// the client gets the decode worker's bytes untouched.
func (s *DecodeStep) responseWriter(reqCtx *pipeline.RequestContext) (http.ResponseWriter, func() error) {
	if reqCtx.PrefillCachedTokens == nil {
		return reqCtx.ResponseWriter, func() error { return nil }
	}
	return usage.NewCachedTokensResponseWriter(reqCtx.ResponseWriter, *reqCtx.PrefillCachedTokens, reqCtx.Stream)
}

// prepareDecodeBody returns the body the decode leg sends. On an OpenAI-format
// leg it mutates reqCtx.Body in place rather than on a clone (unlike prefill and
// conditional-decode). decode is the terminal pipeline step: its body is streamed
// straight to the client and no later step reads reqCtx.Body. A clone would also
// be insufficient, since injectUUIDs mutates nested values that a shallow
// maps.Clone would still share. This is sound only while the pipeline runs steps
// sequentially; if it ever goes concurrent, decode must copy like the others.
//
// The generate format is built fresh instead, since the client's body carries the
// wrong shape whenever its own path is not the generate route.
func (s *DecodeStep) prepareDecodeBody(ctx context.Context, reqCtx *pipeline.RequestContext, format reqcommon.APIType) (map[string]any, error) {
	logger := log.FromContext(ctx).WithName(DecodeStepName)

	kvParams := s.kv.PrepareDecodeKVParams(ctx, reqCtx)
	s.injectUUIDs(reqCtx, logger)

	switch format {
	case reqcommon.APITypeVLLMGenerate:
		return s.buildGenerateBody(reqCtx, kvParams), nil
	case reqcommon.APITypeChatCompletions, reqcommon.APITypeResponses, reqcommon.APITypeMessages:
		reqCtx.Body[reqcommon.FieldKVTransferParams] = kvParams
		return reqCtx.Body, nil
	case reqcommon.APITypeCompletions:
		reqCtx.Body[reqcommon.FieldKVTransferParams] = kvParams
		if len(reqCtx.TokenIDs) > 0 {
			reqCtx.Body["prompt"] = reqCtx.TokenIDs
		}
		return reqCtx.Body, nil
	default:
		// kvParams and injectUUIDs above already ran; both are harmless here
		// since the request fails on this return and no body is sent.
		return nil, unreachableFormatError(format)
	}
}

// buildGenerateBody builds the tokens-in decode body: the rendered token IDs,
// the client's own sampling_params (decode generates the real output, so its
// caps are not rewritten the way prefill's CapSingleToken rewrites them), and
// the KV handoff params at the top level.
//
// output_mode defaults to text, which asks the worker for detokenized text
// beside the token IDs so a tokens-in pipeline answers the client without a
// second /derender call. A derender step needs tokens instead, the only mode
// that endpoint accepts. A worker predating the field ignores either value and
// answers with token IDs only.
//
// A client that called the generate route keeps every other field it sent
// (cache_salt, priority, stream_options, ...); a client on another route has
// none to keep, since its body describes a different API, so only its output
// cap and stream flag carry across.
func (s *DecodeStep) buildGenerateBody(reqCtx *pipeline.RequestContext, kvParams map[string]any) map[string]any {
	clientAPI := reqcommon.DetectAPIType(reqCtx.OriginalPath)

	body := map[string]any{}
	if clientAPI == reqcommon.APITypeVLLMGenerate {
		body = maps.Clone(reqCtx.Body)
	} else {
		body[reqcommon.FieldStream] = reqCtx.Stream
	}
	body["request_id"] = reqCtx.RequestID
	body["token_ids"] = reqCtx.TokenIDs
	body[reqcommon.FieldModel] = reqCtx.Model
	body[reqcommon.FieldOutputMode] = s.outputMode
	body[reqcommon.FieldKVTransferParams] = kvParams
	if features := buildPrefillMMFeatures(reqCtx.MultimodalEntries, nil); features != nil {
		body["features"] = features
	}

	// sampling_params is required by the generate API and has no server-side
	// default, so a client whose cap sits at the top level of another API's body
	// needs it moved under here; without the move the worker falls back to its
	// own max_tokens default (16) and truncates the answer.
	sampling, _ := body[reqcommon.FieldSamplingParams].(map[string]any)
	limits := make(map[string]any, len(sampling)+1)
	maps.Copy(limits, sampling)
	if clientAPI != reqcommon.APITypeVLLMGenerate {
		if limit, ok := reqcommon.OutputTokenLimit(reqCtx.Body, clientAPI); ok {
			limits[reqcommon.FieldMaxTokens] = limit
		}
	}
	body[reqcommon.FieldSamplingParams] = limits

	// The derender mode parses the whole message server-side and vLLM refuses
	// it with stream: true, so the leg is sent unstreamed and the complete
	// answer is forwarded to the client in one write. A client that asked to
	// stream still gets its answer, in a single frame rather than a sequence.
	if s.outputMode == reqcommon.OutputModeDerender {
		body[reqcommon.FieldStream] = false
		delete(body, reqcommon.FieldStreamOptions)
	}

	return body
}

// injectUUIDs tags each media content part with the uuid the decode backend
// uses for prefix-cache keying.
//
// It keys on DetectAPIType(reqCtx.OriginalPath): decode proxies reqCtx.Body to
// reqCtx.OriginalPath, so the wire shape to walk is whatever the client sent.
// resolveFormat's answer instead reflects the encode/prefill wire-format
// setting, which can differ from the client's own shape.
//
// A Messages image block has no uuid field: vLLM's Anthropic conversion drops
// one, so the worker hashes the image itself and decode stamps nothing.
func (s *DecodeStep) injectUUIDs(reqCtx *pipeline.RequestContext, logger logr.Logger) {
	// Without entries there is nothing to stamp, and the prompt items stay raw.
	if len(reqCtx.MultimodalEntries) == 0 {
		return
	}
	apiType := reqcommon.DetectAPIType(reqCtx.OriginalPath)
	if apiType == reqcommon.APITypeMessages {
		return
	}
	if items, ok := promptItems(reqCtx.Body, apiType); ok {
		injectMediaPartUUIDs(items, apiType, reqCtx.MultimodalEntries, logger)
		setPromptItems(reqCtx.Body, apiType, items)
	}
}

// injectMediaPartUUIDs stamps each media content part with the hash of its
// corresponding multimodal entry, pairing the two by position within a modality.
// A surplus part is left unstamped, so the worker hashes the media itself rather
// than reading an entry primed under another part's hash; a surplus entry has no
// part to stamp. Neither is fatal; the branches below record what each costs.
func injectMediaPartUUIDs(items []any, apiType reqcommon.APIType, entries []pipeline.MultimodalEntry, logger logr.Logger) {
	// Grouped in entry order, so the walk below indexes by the per-modality
	// position: O(1) per part after an O(n) build.
	hashesByMod := make(map[reqcommon.Modality][]string)
	for _, entry := range entries {
		hashesByMod[entry.Modality] = append(hashesByMod[entry.Modality], entry.Hash)
	}

	modCounter := make(map[reqcommon.Modality]int)
	for _, media := range collectMediaParts(items, apiType) {
		localIdx := modCounter[media.modality]
		modCounter[media.modality]++
		hashes := hashesByMod[media.modality]
		if localIdx < len(hashes) {
			media.part["uuid"] = hashes[localIdx]
			continue
		}
		// A miss means entries and parts got out of line upstream (see
		// collectMediaParts). The request is still answered, but the worker
		// hashes and re-processes that media itself -- at the caps
		// coordinator.yaml suggests, up to 200 MB of video, redoing the encode
		// work the EPD split exists to do once elsewhere.
		logger.V(logutil.DEBUG).Info("no MultimodalEntry for media part",
			"location", media.location,
			"modality", media.modality,
			"local_index", localIdx,
			"modality_entry_count", len(hashes))
	}

	// The walk above iterates parts, so it can only see a surplus part; the
	// other direction needs its own pass, and is the worse of the two. A
	// surplus entry's hash still reaches the prefiller, since
	// PreparePrefillECParams flattens every encode response into
	// ec_transfer_params, so the prefill body describes an EC buffer no part of
	// the decode body names by uuid.
	for mod, hashes := range hashesByMod {
		if partCount := modCounter[mod]; partCount < len(hashes) {
			logger.V(logutil.DEBUG).Info("MultimodalEntry with no media part",
				"modality", mod,
				"part_count", partCount,
				"modality_entry_count", len(hashes))
		}
	}
}
