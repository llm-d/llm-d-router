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
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"slices"
	"strconv"
	"strings"

	"google.golang.org/protobuf/proto"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/gen/vllmpb"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/vllmwire"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"

	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

func scanVLLMGenerate(r *http.Request, payload []byte) (grpcScanResult, error) {
	genReq, err := vllmwire.ScanGenerateRequest(payload)
	if err != nil {
		return grpcScanResult{}, err
	}
	stream := r.URL.Path == vllmpb.Inference_GenerateStream_FullMethodName
	return grpcScanResult{
		req:                   &pipeline.VLLMGRPCRequest{Req: genReq},
		stream:                stream,
		logStream:             stream,
		model:                 genReq.Model(),
		hasModel:              true,
		droppedTransferParams: genReq.HadClientTransferParams(),
	}, nil
}

var (
	allowedHTTPGenerateFields = map[string]struct{}{
		reqcommon.FieldModel:                {},
		"request_id":                        {},
		reqcommon.FieldPrompt:               {},
		"token_ids":                         {},
		reqcommon.FieldSamplingParams:       {},
		"features":                          {},
		"content_parts":                     {},
		reqcommon.FieldStream:               {},
		reqcommon.FieldStreamOptions:        {},
		"lora_name":                         {},
		"lora_request":                      {},
		reqcommon.FieldPriority:             {},
		reqcommon.FieldCacheSalt:            {},
		"bypass_prefix_cache":               {},
		reqcommon.FieldTruncatePromptTokens: {},
		"session_id":                        {},
		"watermarking":                      {},
		"structured_outputs":                {},
		reqcommon.FieldReturnTokenIDs:       {},
		reqcommon.FieldECTransferParams:     {},
		reqcommon.FieldKVTransferParams:     {},
	}

	allowedStreamOptionsFields = map[string]struct{}{
		reqcommon.FieldIncludeUsage:         {},
		reqcommon.FieldContinuousUsageStats: {},
	}

	allowedSamplingParamsFields = map[string]struct{}{
		"temperature":                        {},
		reqcommon.FieldN:                     {},
		"top_p":                              {},
		"top_k":                              {},
		"min_p":                              {},
		"seed":                               {},
		"presence_penalty":                   {},
		"frequency_penalty":                  {},
		"repetition_penalty":                 {},
		"logit_bias":                         {},
		"allowed_token_ids":                  {},
		"bad_words_token_ids":                {},
		"bad_words":                          {},
		"structured_outputs":                 {},
		"guided_json":                        {},
		"guided_regex":                       {},
		"guided_choice":                      {},
		"guided_grammar":                     {},
		"structural_tag":                     {},
		reqcommon.FieldMaxTokens:             {},
		reqcommon.FieldMinTokens:             {},
		"stop":                               {},
		"stop_token_ids":                     {},
		"include_stop_str_in_output":         {},
		"ignore_eos":                         {},
		"thinking_token_budget":              {},
		"skip_special_tokens":                {},
		"detokenize":                         {},
		reqcommon.FieldLogprobs:              {},
		"top_logprobs":                       {},
		reqcommon.FieldPromptLogprobs:        {},
		reqcommon.FieldPromptLogprobTokenIDs: {},
		reqcommon.FieldTruncatePromptTokens:  {},
		"vllm_xargs":                         {},
	}

	allowedStructuredOutputsFields = map[string]struct{}{
		"json":           {},
		"regex":          {},
		"choice":         {},
		"grammar":        {},
		"json_object":    {},
		"structural_tag": {},
	}

	allowedVLLMXargsFields = map[string]struct{}{
		reqcommon.FieldCacheSalt:        {},
		"bypass_prefix_cache":           {},
		reqcommon.FieldKVTransferParams: {},
		reqcommon.FieldECTransferParams: {},
	}

	allowedFeaturesFields = map[string]struct{}{
		"mm_hashes":       {},
		"mm_placeholders": {},
		"kwargs_data":     {},
	}
)

func validateKnownKeys(m map[string]any, scope string, allowed map[string]struct{}) error {
	var unknown []string
	for k := range m {
		if _, ok := allowed[k]; !ok {
			unknown = append(unknown, k)
		}
	}
	if len(unknown) == 0 {
		return nil
	}
	slices.Sort(unknown)
	return fmt.Errorf("unsupported %s field %q", scope, unknown[0])
}

type httpVLLMGenerateResult struct {
	req                   vllmwire.Request
	includeUsage          bool
	returnTokenIDs        bool
	clientRequestID       string
	droppedTransferParams bool
}

// prepareVLLMGenerateFromHTTP translates an inbound HTTP POST /inference/v1/generate
// JSON request body into a scanned vllmwire.Request so the pipeline executes all
// worker legs over vllm.Inference gRPC.
func prepareVLLMGenerateFromHTTP(parsed map[string]any, fallbackRequestID string) (httpVLLMGenerateResult, error) {
	var out httpVLLMGenerateResult
	if err := validateKnownKeys(parsed, "request", allowedHTTPGenerateFields); err != nil {
		return httpVLLMGenerateResult{}, err
	}
	if _, present := parsed[reqcommon.FieldECTransferParams]; present {
		out.droppedTransferParams = true
	}

	var pbReq vllmpb.GenerateRequest
	if m, ok := parsed[reqcommon.FieldModel].(string); ok {
		pbReq.Model = m
	}

	clientReqID, hasReqID, err := decodeOptionalStringField(parsed, "request_id")
	if err != nil {
		return httpVLLMGenerateResult{}, err
	}
	if hasReqID && clientReqID != "" {
		out.clientRequestID = clientReqID
		pbReq.RequestId = clientReqID
	} else {
		pbReq.RequestId = fallbackRequestID
	}

	if loraName, ok, err := decodeOptionalStringField(parsed, "lora_name"); err != nil {
		return httpVLLMGenerateResult{}, err
	} else if ok && loraName != "" {
		pbReq.LoraName = loraName
	} else if rawLoRA, present := parsed["lora_request"]; present && rawLoRA != nil {
		val, hasVal, err := unwrapRawJSONValue(rawLoRA)
		if err != nil {
			return httpVLLMGenerateResult{}, errors.New("lora_request must be a string or object")
		}
		if hasVal {
			switch lr := val.(type) {
			case string:
				pbReq.LoraName = lr
			case map[string]any:
				name, _, err := decodeOptionalStringField(lr, "lora_name")
				if err != nil {
					return httpVLLMGenerateResult{}, err
				}
				pbReq.LoraName = name
			default:
				return httpVLLMGenerateResult{}, errors.New("lora_request must be a string or object")
			}
		}
	}

	if prio, ok, err := decodeOptionalInt32Field(parsed, reqcommon.FieldPriority); err != nil {
		return httpVLLMGenerateResult{}, err
	} else if ok {
		pbReq.Priority = prio
	}

	if sessionID, ok, err := decodeOptionalStringField(parsed, "session_id"); err != nil {
		return httpVLLMGenerateResult{}, err
	} else if ok {
		pbReq.SessionId = &sessionID
	}

	if watermarking, ok, err := decodeOptionalBoolField(parsed, "watermarking"); err != nil {
		return httpVLLMGenerateResult{}, err
	} else if ok {
		pbReq.Watermarking = &watermarking
	}

	includeUsage, err := extractHTTPStreamIncludeUsage(parsed)
	if err != nil {
		return httpVLLMGenerateResult{}, err
	}
	out.includeUsage = includeUsage

	tokenCount, err := applyHTTPPrompt(&pbReq, parsed)
	if err != nil {
		return httpVLLMGenerateResult{}, err
	}

	returnTokenIDs, _, err := decodeOptionalBoolField(parsed, reqcommon.FieldReturnTokenIDs)
	if err != nil {
		return httpVLLMGenerateResult{}, err
	}
	out.returnTokenIDs = returnTokenIDs

	droppedSamplingTransfer, err := applyHTTPSamplingAndKV(&pbReq, parsed, returnTokenIDs)
	if err != nil {
		return httpVLLMGenerateResult{}, err
	}
	if droppedSamplingTransfer {
		out.droppedTransferParams = true
	}

	mediaItems, imageCount, otherCount, err := extractHTTPMediaItems(parsed, tokenCount)
	if err != nil {
		return httpVLLMGenerateResult{}, err
	}
	pbReq.Media = mediaItems
	coordmetrics.RecordMediaItems(coordmetrics.MediaTypeImage, imageCount)
	coordmetrics.RecordMediaItems(coordmetrics.MediaTypeOther, otherCount)

	payload, err := proto.Marshal(&pbReq)
	if err != nil {
		return httpVLLMGenerateResult{}, errors.New("failed to encode GenerateRequest")
	}
	scanned, err := vllmwire.ScanGenerateRequest(payload)
	if err != nil {
		return httpVLLMGenerateResult{}, errors.New("failed to scan GenerateRequest")
	}
	out.req = scanned
	return out, nil
}

func applyHTTPPrompt(pbReq *vllmpb.GenerateRequest, parsed map[string]any) (int, error) {
	rawTokens := parsed["token_ids"]
	rawPrompt := parsed[reqcommon.FieldPrompt]
	if rawTokens != nil && rawPrompt != nil {
		return 0, errors.New("cannot specify both token_ids and prompt")
	}
	if rawTokens != nil {
		ids, err := parseUint32Array(rawTokens, "token_ids", false)
		if err != nil {
			return 0, err
		}
		pbReq.Prompt = &vllmpb.GenerateRequest_TokenIds{
			TokenIds: &vllmpb.TokenIds{Ids: ids},
		}
		return len(ids), nil
	}
	if rawPrompt != nil {
		switch p := rawPrompt.(type) {
		case string:
			pbReq.Prompt = &vllmpb.GenerateRequest_Text{Text: p}
			return -1, nil
		case []any:
			ids, err := parseUint32Array(p, "prompt", false)
			if err != nil {
				return 0, err
			}
			pbReq.Prompt = &vllmpb.GenerateRequest_TokenIds{
				TokenIds: &vllmpb.TokenIds{Ids: ids},
			}
			return len(ids), nil
		default:
			return 0, errors.New("prompt must be a string or token array")
		}
	}
	return 0, errors.New("token_ids or prompt is required")
}

func applyHTTPSamplingAndKV(pbReq *vllmpb.GenerateRequest, parsed map[string]any, returnTokenIDs bool) (bool, error) {
	droppedTransfer := false

	var (
		cacheSalt         string
		bypassPrefixCache bool
		hasKV             bool
	)
	if cs, ok, err := decodeOptionalStringField(parsed, reqcommon.FieldCacheSalt); err != nil {
		return false, err
	} else if ok && cs != "" {
		cacheSalt = cs
		hasKV = true
	}
	if bp, ok, err := decodeOptionalBoolField(parsed, "bypass_prefix_cache"); err != nil {
		return false, err
	} else if ok && bp {
		bypassPrefixCache = true
		hasKV = true
	}

	topTruncate, hasTopTruncate, err := decodeOptionalUint32Field(parsed, reqcommon.FieldTruncatePromptTokens)
	if err != nil {
		return false, err
	}
	if hasTopTruncate {
		pbReq.TruncatePromptTokens = topTruncate
	}

	var (
		decoding          vllmpb.DecodingParameters
		hasDecoding       bool
		stopStrings       []string
		skipSpecial       *bool
		detokenize        *bool
		outputLogprobs    bool
		outputCandidates  *vllmpb.CandidateTokens
		promptLogprobs    bool
		promptCandidates  *vllmpb.CandidateTokens
		promptTokenIDsOpt = returnTokenIDs
	)

	hasTopStructOut, err := applyStructuredOutputMap(&decoding, parsed, "structured_outputs", "structured_outputs")
	if err != nil {
		return false, err
	}
	if hasTopStructOut {
		hasDecoding = true
	}

	if rawSampling := parsed[reqcommon.FieldSamplingParams]; rawSampling != nil {
		sp, ok := rawSampling.(map[string]any)
		if !ok {
			return false, errors.New("sampling_params must be an object")
		}
		if err := validateKnownKeys(sp, "sampling_params", allowedSamplingParamsFields); err != nil {
			return false, err
		}

		if spTruncate, hasSPTruncate, err := decodeOptionalUint32Field(sp, reqcommon.FieldTruncatePromptTokens); err != nil {
			return false, err
		} else if hasSPTruncate {
			if hasTopTruncate && spTruncate != topTruncate {
				return false, errors.New("conflicting truncate_prompt_tokens values")
			}
			pbReq.TruncatePromptTokens = spTruncate
		}

		if temp, ok, err := decodeOptionalFloat32Field(sp, "temperature"); err != nil {
			return false, err
		} else if ok {
			pbReq.Temperature = &temp
		}

		var sampling vllmpb.RandomSampling
		hasSampling := false
		if n, ok, err := decodeOptionalUint32Field(sp, reqcommon.FieldN); err != nil {
			return false, err
		} else if ok {
			sampling.NumSequences = n
			hasSampling = true
		}
		if topP, ok, err := decodeOptionalFloat32Field(sp, "top_p"); err != nil {
			return false, err
		} else if ok {
			sampling.TopP = &topP
			hasSampling = true
		}
		if topK, ok, err := decodeOptionalInt64Field(sp, "top_k"); err != nil {
			return false, err
		} else if ok {
			if topK < -1 || topK > math.MaxUint32 {
				return false, errors.New("sampling_params.top_k out of range")
			}
			var u uint32
			if topK > 0 {
				u = uint32(topK)
			}
			sampling.TopK = &u
			hasSampling = true
		}
		if minP, ok, err := decodeOptionalFloat32Field(sp, "min_p"); err != nil {
			return false, err
		} else if ok {
			sampling.MinP = &minP
			hasSampling = true
		}
		if seed, ok, err := decodeOptionalInt64Field(sp, "seed"); err != nil {
			return false, err
		} else if ok {
			sampling.Seed = &seed
			hasSampling = true
		}
		if hasSampling {
			pbReq.Sampling = &sampling
		}

		if pp, ok, err := decodeOptionalFloat32Field(sp, "presence_penalty"); err != nil {
			return false, err
		} else if ok {
			decoding.PresencePenalty = pp
			hasDecoding = true
		}
		if fp, ok, err := decodeOptionalFloat32Field(sp, "frequency_penalty"); err != nil {
			return false, err
		} else if ok {
			decoding.FrequencyPenalty = fp
			hasDecoding = true
		}
		if rp, ok, err := decodeOptionalFloat32Field(sp, "repetition_penalty"); err != nil {
			return false, err
		} else if ok {
			decoding.RepetitionPenalty = rp
			hasDecoding = true
		}
		if bias, ok, err := parseLogitBias(sp, "logit_bias"); err != nil {
			return false, err
		} else if ok && len(bias) > 0 {
			decoding.LogitBias = bias
			hasDecoding = true
		}
		if rawAllowed, present := sp["allowed_token_ids"]; present && rawAllowed != nil {
			allowedIDs, err := parseUint32Array(rawAllowed, "sampling_params.allowed_token_ids", true)
			if err != nil {
				return false, err
			}
			if len(allowedIDs) > 0 {
				decoding.AllowedTokenIds = allowedIDs
				hasDecoding = true
			}
		}
		badWordsIDs, err := extractBadWordsTokenIDs(sp)
		if err != nil {
			return false, err
		}
		if len(badWordsIDs) > 0 {
			decoding.BadWordsTokenIds = badWordsIDs
			hasDecoding = true
		}

		hasSPStructOut, err := applySamplingStructuredOutput(&decoding, sp)
		if err != nil {
			return false, err
		}
		if hasSPStructOut {
			hasDecoding = true
		}

		var stopping vllmpb.StoppingCriteria
		hasStopping := false
		if maxToks, ok, err := decodeOptionalUint32Field(sp, reqcommon.FieldMaxTokens); err != nil {
			return false, err
		} else if ok {
			stopping.MaxNewTokens = maxToks
			hasStopping = true
		}
		if minToks, ok, err := decodeOptionalUint32Field(sp, reqcommon.FieldMinTokens); err != nil {
			return false, err
		} else if ok {
			stopping.MinNewTokens = minToks
			hasStopping = true
		}
		if rawStop, present := sp["stop"]; present && rawStop != nil {
			parsedStops, err := parseStopStrings(rawStop)
			if err != nil {
				return false, err
			}
			if len(parsedStops) > 0 {
				stopStrings = parsedStops
				stopping.StopStrings = parsedStops
				hasStopping = true
			}
		}
		if rawStopIDs, present := sp["stop_token_ids"]; present && rawStopIDs != nil {
			stopIDs, err := parseUint32Array(rawStopIDs, "sampling_params.stop_token_ids", true)
			if err != nil {
				return false, err
			}
			if len(stopIDs) > 0 {
				stopping.StopTokenIds = stopIDs
				hasStopping = true
			}
		}
		if incStop, ok, err := decodeOptionalBoolField(sp, "include_stop_str_in_output"); err != nil {
			return false, err
		} else if ok && incStop {
			stopping.IncludeStopStrings = true
			hasStopping = true
		}
		if ignoreEOS, ok, err := decodeOptionalBoolField(sp, "ignore_eos"); err != nil {
			return false, err
		} else if ok && ignoreEOS {
			stopping.IgnoreEos = true
			hasStopping = true
		}
		if thinkBudget, ok, err := decodeOptionalInt64Field(sp, "thinking_token_budget"); err != nil {
			return false, err
		} else if ok {
			stopping.ThinkingTokenBudget = &thinkBudget
			hasStopping = true
		}
		if hasStopping {
			pbReq.Stopping = &stopping
		}

		if skip, ok, err := decodeOptionalBoolField(sp, "skip_special_tokens"); err != nil {
			return false, err
		} else if ok {
			skipSpecial = &skip
		}
		if detok, ok, err := decodeOptionalBoolField(sp, "detokenize"); err != nil {
			return false, err
		} else if ok {
			detokenize = &detok
		}

		outputLogprobs, outputCandidates, err = extractOutputLogprobs(sp)
		if err != nil {
			return false, err
		}
		promptLogprobs, promptCandidates, err = extractPromptLogprobs(sp)
		if err != nil {
			return false, err
		}
		if promptLogprobs {
			promptTokenIDsOpt = true
		}

		if rawXargs, present := sp["vllm_xargs"]; present && rawXargs != nil {
			xargs, ok, err := decodeOptionalObjectField(sp, "vllm_xargs")
			if err != nil || !ok {
				return false, errors.New("sampling_params.vllm_xargs must be an object")
			}
			if err := validateKnownKeys(xargs, "sampling_params.vllm_xargs", allowedVLLMXargsFields); err != nil {
				return false, err
			}
			if _, hasKVTransfer := xargs[reqcommon.FieldKVTransferParams]; hasKVTransfer {
				droppedTransfer = true
			}
			if _, hasECTransfer := xargs[reqcommon.FieldECTransferParams]; hasECTransfer {
				droppedTransfer = true
			}
			if cs, ok, err := decodeOptionalStringField(xargs, reqcommon.FieldCacheSalt); err != nil {
				return false, err
			} else if ok && cs != "" {
				cacheSalt = cs
				hasKV = true
			}
			if bp, ok, err := decodeOptionalBoolField(xargs, "bypass_prefix_cache"); err != nil {
				return false, err
			} else if ok && bp {
				bypassPrefixCache = true
				hasKV = true
			}
		}
	}

	if hasDecoding {
		pbReq.Decoding = &decoding
	}
	if hasKV {
		pbReq.Kv = &vllmpb.KVCacheParameters{
			CacheSalt:         cacheSalt,
			BypassPrefixCache: bypassPrefixCache,
		}
	}

	// /inference/v1/generate defaults to token-only output (output_mode="tokens")
	// so --tokens-only decode workers do not detokenize unless stop strings or
	// explicit sampling_params.detokenize require it.
	outputText := len(stopStrings) > 0
	if detokenize != nil {
		if !*detokenize && len(stopStrings) > 0 {
			return false, errors.New("sampling_params.detokenize=false cannot be combined with stop strings")
		}
		outputText = *detokenize
	}

	pbReq.Response = &vllmpb.ResponseOptions{
		PromptTokenIds:    promptTokenIDsOpt,
		PromptLogprobs:    promptLogprobs,
		PromptCandidates:  promptCandidates,
		OutputTokenIds:    true,
		OutputText:        proto.Bool(outputText),
		OutputLogprobs:    outputLogprobs,
		OutputCandidates:  outputCandidates,
		SkipSpecialTokens: skipSpecial,
	}
	return droppedTransfer, nil
}

func extractHTTPStreamIncludeUsage(parsed map[string]any) (bool, error) {
	opts, ok, err := decodeOptionalObjectField(parsed, reqcommon.FieldStreamOptions)
	if err != nil {
		return false, err
	}
	if !ok {
		return false, nil
	}
	if err := validateKnownKeys(opts, "stream_options", allowedStreamOptionsFields); err != nil {
		return false, err
	}
	if cont, ok, err := decodeOptionalBoolField(opts, reqcommon.FieldContinuousUsageStats); err != nil {
		return false, err
	} else if ok && cont {
		return false, errors.New("stream_options.continuous_usage_stats is not supported with worker_protocol=grpc")
	}
	inc, _, err := decodeOptionalBoolField(opts, reqcommon.FieldIncludeUsage)
	if err != nil {
		return false, err
	}
	return inc, nil
}

func extractOutputLogprobs(sp map[string]any) (bool, *vllmpb.CandidateTokens, error) {
	topLP, hasTopLP, err := decodeOptionalUint32Field(sp, "top_logprobs")
	if err != nil {
		return false, nil, fmt.Errorf("sampling_params.%w", err)
	}
	enabled, explicitFalse, cand, err := parseLogprobSpec(sp, reqcommon.FieldLogprobs, "sampling_params.logprobs")
	if err != nil {
		return false, nil, err
	}
	if hasTopLP && topLP > 0 {
		if explicitFalse {
			return false, nil, errors.New("sampling_params.top_logprobs cannot be set when sampling_params.logprobs is false")
		}
		if cand != nil {
			if existingTop, ok := cand.Select.(*vllmpb.CandidateTokens_TopN); !ok || existingTop.TopN != topLP {
				return false, nil, errors.New("conflicting sampling_params.logprobs and sampling_params.top_logprobs")
			}
		} else {
			cand = &vllmpb.CandidateTokens{Select: &vllmpb.CandidateTokens_TopN{TopN: topLP}}
		}
		enabled = true
	}
	return enabled, cand, nil
}

func extractPromptLogprobs(sp map[string]any) (bool, *vllmpb.CandidateTokens, error) {
	enabled, explicitFalse, cand, err := parseLogprobSpec(sp, reqcommon.FieldPromptLogprobs, "sampling_params.prompt_logprobs")
	if err != nil {
		return false, nil, err
	}
	if rawIDs, present := sp[reqcommon.FieldPromptLogprobTokenIDs]; present && rawIDs != nil {
		val, hasVal, unwrapErr := unwrapRawJSONValue(rawIDs)
		if unwrapErr != nil {
			return false, nil, errors.New("sampling_params.prompt_logprob_token_ids must be an array")
		}
		if hasVal {
			ids, err := parseUint32Array(val, "sampling_params.prompt_logprob_token_ids", false)
			if err != nil {
				return false, nil, err
			}
			if explicitFalse {
				return false, nil, errors.New("sampling_params.prompt_logprob_token_ids cannot be set when sampling_params.prompt_logprobs is false")
			}
			if cand != nil {
				return false, nil, errors.New("cannot combine numeric sampling_params.prompt_logprobs candidates with prompt_logprob_token_ids")
			}
			enabled = true
			cand = &vllmpb.CandidateTokens{
				Select: &vllmpb.CandidateTokens_TokenIds{
					TokenIds: &vllmpb.TokenIds{Ids: ids},
				},
			}
		}
	}
	return enabled, cand, nil
}

func parseLogprobSpec(m map[string]any, field, qualified string) (enabled bool, explicitFalse bool, cand *vllmpb.CandidateTokens, err error) {
	raw, present := m[field]
	if !present || raw == nil {
		return false, false, nil, nil
	}
	val, hasVal, err := unwrapRawJSONValue(raw)
	if err != nil {
		return false, false, nil, fmt.Errorf("%s must be a boolean or integer", qualified)
	}
	if !hasVal {
		return false, false, nil, nil
	}
	if b, ok := val.(bool); ok {
		if !b {
			return false, true, nil, nil
		}
		return true, false, nil, nil
	}
	n, _, intErr := decodeOptionalInt64Field(map[string]any{qualified: val}, qualified)
	if intErr != nil {
		return false, false, nil, fmt.Errorf("%s must be a boolean or integer", qualified)
	}
	if n < -1 || n > math.MaxUint32 {
		return false, false, nil, fmt.Errorf("%s must be >= -1 and <= %d", qualified, uint32(math.MaxUint32))
	}
	if n == -1 {
		return true, false, &vllmpb.CandidateTokens{Select: &vllmpb.CandidateTokens_All{All: true}}, nil
	}
	if n == 0 {
		return true, false, nil, nil
	}
	return true, false, &vllmpb.CandidateTokens{Select: &vllmpb.CandidateTokens_TopN{TopN: uint32(n)}}, nil //#nosec G115 -- guarded above
}

func parseLogitBias(m map[string]any, field string) (map[uint32]float32, bool, error) {
	obj, ok, err := decodeOptionalObjectField(m, field)
	if err != nil {
		return nil, false, fmt.Errorf("sampling_params.%w", err)
	}
	if !ok {
		return nil, false, nil
	}
	out := make(map[uint32]float32, len(obj))
	for k := range obj {
		tokID, parseErr := strconv.ParseUint(k, 10, 32)
		if parseErr != nil {
			return nil, false, fmt.Errorf("sampling_params.logit_bias key %q must be a uint32 token ID", k)
		}
		bias, hasVal, fErr := decodeOptionalFloat32Field(obj, k)
		if fErr != nil || !hasVal {
			return nil, false, fmt.Errorf("sampling_params.logit_bias[%q] must be a finite float32 number", k)
		}
		out[uint32(tokID)] = bias //#nosec G115 -- bitSize 32 bounds tokID
	}
	return out, true, nil
}

func extractBadWordsTokenIDs(sp map[string]any) ([]*vllmpb.TokenIds, error) {
	explicit, hasExplicit, err := parseTokenIDSequenceList(sp, "bad_words_token_ids", "sampling_params.bad_words_token_ids", false)
	if err != nil {
		return nil, err
	}
	legacy, hasLegacy, err := parseTokenIDSequenceList(sp, "bad_words", "sampling_params.bad_words", true)
	if err != nil {
		return nil, err
	}
	if hasExplicit && hasLegacy {
		return nil, errors.New("cannot specify both sampling_params.bad_words and sampling_params.bad_words_token_ids")
	}
	if hasExplicit {
		return explicit, nil
	}
	return legacy, nil
}

func parseTokenIDSequenceList(m map[string]any, field, qualified string, rejectStrings bool) ([]*vllmpb.TokenIds, bool, error) {
	arr, ok, err := decodeOptionalArrayField(m, field)
	if err != nil {
		return nil, false, fmt.Errorf("%s must be an array of token ID sequences", qualified)
	}
	if !ok || len(arr) == 0 {
		return nil, false, nil
	}
	out := make([]*vllmpb.TokenIds, 0, len(arr))
	for i, elem := range arr {
		if rejectStrings {
			if _, isStr := elem.(string); isStr {
				return nil, false, fmt.Errorf("%s requires token ID sequences (use bad_words_token_ids)", qualified)
			}
		}
		seq, seqErr := parseUint32Array(elem, fmt.Sprintf("%s[%d]", qualified, i), false)
		if seqErr != nil {
			return nil, false, seqErr
		}
		out = append(out, &vllmpb.TokenIds{Ids: seq})
	}
	return out, true, nil
}

func applyStructuredOutputMap(decoding *vllmpb.DecodingParameters, m map[string]any, field, qualified string) (bool, error) {
	obj, ok, err := decodeOptionalObjectField(m, field)
	if err != nil {
		return false, fmt.Errorf("%s must be an object", qualified)
	}
	if !ok {
		return false, nil
	}
	if err := validateKnownKeys(obj, qualified, allowedStructuredOutputsFields); err != nil {
		return false, err
	}

	applied := false
	ensureUnset := func() error {
		if decoding.StructuredOutput != nil {
			return errors.New("at most one structured output constraint may be specified")
		}
		applied = true
		return nil
	}

	if s, ok, err := decodeStringOrJSONField(obj, "json", qualified+".json"); err != nil {
		return false, err
	} else if ok && s != "" {
		if err := ensureUnset(); err != nil {
			return false, err
		}
		decoding.StructuredOutput = &vllmpb.DecodingParameters_Json{Json: s}
	}
	if s, ok, err := decodeOptionalStringField(obj, "regex"); err != nil {
		return false, fmt.Errorf("%s.%w", qualified, err)
	} else if ok && s != "" {
		if err := ensureUnset(); err != nil {
			return false, err
		}
		decoding.StructuredOutput = &vllmpb.DecodingParameters_Regex{Regex: s}
	}
	if choices, ok, err := parseStringChoiceArray(obj, "choice", qualified+".choice"); err != nil {
		return false, err
	} else if ok {
		if err := ensureUnset(); err != nil {
			return false, err
		}
		decoding.StructuredOutput = &vllmpb.DecodingParameters_Choice{
			Choice: &vllmpb.DecodingParameters_StringChoices{Choices: choices},
		}
	}
	if s, ok, err := decodeOptionalStringField(obj, "grammar"); err != nil {
		return false, fmt.Errorf("%s.%w", qualified, err)
	} else if ok && s != "" {
		if err := ensureUnset(); err != nil {
			return false, err
		}
		decoding.StructuredOutput = &vllmpb.DecodingParameters_Grammar{Grammar: s}
	}
	if b, ok, err := decodeOptionalBoolField(obj, "json_object"); err != nil {
		return false, fmt.Errorf("%s.%w", qualified, err)
	} else if ok && b {
		if err := ensureUnset(); err != nil {
			return false, err
		}
		decoding.StructuredOutput = &vllmpb.DecodingParameters_JsonObject{JsonObject: true}
	}
	if s, ok, err := decodeStringOrJSONField(obj, "structural_tag", qualified+".structural_tag"); err != nil {
		return false, err
	} else if ok && s != "" {
		if err := ensureUnset(); err != nil {
			return false, err
		}
		decoding.StructuredOutput = &vllmpb.DecodingParameters_StructuralTag{StructuralTag: s}
	}

	return applied, nil
}

func applySamplingStructuredOutput(decoding *vllmpb.DecodingParameters, sp map[string]any) (bool, error) {
	applied, err := applyStructuredOutputMap(decoding, sp, "structured_outputs", "sampling_params.structured_outputs")
	if err != nil {
		return false, err
	}
	ensureUnset := func() error {
		if decoding.StructuredOutput != nil {
			return errors.New("at most one structured output constraint may be specified")
		}
		applied = true
		return nil
	}

	if s, ok, err := decodeStringOrJSONField(sp, "guided_json", "sampling_params.guided_json"); err != nil {
		return false, err
	} else if ok && s != "" {
		if err := ensureUnset(); err != nil {
			return false, err
		}
		decoding.StructuredOutput = &vllmpb.DecodingParameters_Json{Json: s}
	}
	if s, ok, err := decodeOptionalStringField(sp, "guided_regex"); err != nil {
		return false, fmt.Errorf("sampling_params.%w", err)
	} else if ok && s != "" {
		if err := ensureUnset(); err != nil {
			return false, err
		}
		decoding.StructuredOutput = &vllmpb.DecodingParameters_Regex{Regex: s}
	}
	if choices, ok, err := parseStringChoiceArray(sp, "guided_choice", "sampling_params.guided_choice"); err != nil {
		return false, err
	} else if ok {
		if err := ensureUnset(); err != nil {
			return false, err
		}
		decoding.StructuredOutput = &vllmpb.DecodingParameters_Choice{
			Choice: &vllmpb.DecodingParameters_StringChoices{Choices: choices},
		}
	}
	if s, ok, err := decodeOptionalStringField(sp, "guided_grammar"); err != nil {
		return false, fmt.Errorf("sampling_params.%w", err)
	} else if ok && s != "" {
		if err := ensureUnset(); err != nil {
			return false, err
		}
		decoding.StructuredOutput = &vllmpb.DecodingParameters_Grammar{Grammar: s}
	}
	if s, ok, err := decodeStringOrJSONField(sp, "structural_tag", "sampling_params.structural_tag"); err != nil {
		return false, err
	} else if ok && s != "" {
		if err := ensureUnset(); err != nil {
			return false, err
		}
		decoding.StructuredOutput = &vllmpb.DecodingParameters_StructuralTag{StructuralTag: s}
	}

	return applied, nil
}

func decodeStringOrJSONField(m map[string]any, field, qualified string) (string, bool, error) {
	raw, present := m[field]
	if !present || raw == nil {
		return "", false, nil
	}
	if rm, isRaw := raw.(json.RawMessage); isRaw {
		trimmed := bytes.TrimSpace(rm)
		if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
			return "", false, nil
		}
		if trimmed[0] == '"' {
			var s string
			if err := json.Unmarshal(trimmed, &s); err != nil {
				return "", false, fmt.Errorf("%s must be a JSON schema string or object", qualified)
			}
			return s, true, nil
		}
		if trimmed[0] == '{' {
			var obj map[string]any
			if err := json.Unmarshal(trimmed, &obj); err != nil {
				return "", false, fmt.Errorf("%s must be a JSON schema string or object", qualified)
			}
			return string(trimmed), true, nil
		}
		return "", false, fmt.Errorf("%s must be a JSON schema string or object", qualified)
	}
	switch v := raw.(type) {
	case string:
		return v, true, nil
	case map[string]any:
		encoded, err := json.Marshal(v)
		if err != nil {
			return "", false, fmt.Errorf("%s must be a JSON schema string or object", qualified)
		}
		return string(encoded), true, nil
	default:
		return "", false, fmt.Errorf("%s must be a JSON schema string or object", qualified)
	}
}

func parseStringChoiceArray(m map[string]any, field, qualified string) ([]string, bool, error) {
	arr, ok, err := decodeOptionalArrayField(m, field)
	if err != nil {
		return nil, false, fmt.Errorf("%s must be an array of strings", qualified)
	}
	if !ok {
		return nil, false, nil
	}
	if len(arr) == 0 {
		return nil, false, fmt.Errorf("%s must not be empty", qualified)
	}
	out := make([]string, 0, len(arr))
	for _, elem := range arr {
		s, isStr := elem.(string)
		if !isStr {
			return nil, false, fmt.Errorf("%s elements must be strings", qualified)
		}
		out = append(out, s)
	}
	return out, true, nil
}

func extractHTTPMediaItems(parsed map[string]any, tokenCount int) ([]*vllmpb.MediaItem, int, int, error) {
	features, hasFeatures, err := decodeOptionalObjectField(parsed, "features")
	if err != nil {
		return nil, 0, 0, err
	}
	rawParts, hasParts, err := decodeOptionalArrayField(parsed, "content_parts")
	if err != nil {
		return nil, 0, 0, err
	}
	if hasFeatures && hasParts {
		return nil, 0, 0, errors.New("cannot specify both features and content_parts")
	}

	var (
		items      []*vllmpb.MediaItem
		imageCount int
		otherCount int
	)

	if hasFeatures {
		if err := validateKnownKeys(features, "features", allowedFeaturesFields); err != nil {
			return nil, 0, 0, err
		}
		featItems, imgN, othN, err := convertHTTPFeaturesToMedia(features, tokenCount)
		if err != nil {
			return nil, 0, 0, err
		}
		items = append(items, featItems...)
		imageCount += imgN
		otherCount += othN
	}

	if hasParts {
		for _, elem := range rawParts {
			part, ok := elem.(map[string]any)
			if !ok {
				return nil, 0, 0, errors.New("content_parts elements must be objects")
			}
			partType, ok, err := decodeOptionalStringField(part, reqcommon.FieldType)
			if err != nil || !ok {
				return nil, 0, 0, errors.New("content_parts.type must be a string")
			}
			rawURL, ok, err := decodeOptionalStringField(part, reqcommon.FieldURL)
			if err != nil || !ok || rawURL == "" {
				return nil, 0, 0, errors.New("content_parts.url is required")
			}
			itemUUID, _, err := decodeOptionalStringField(part, reqcommon.FieldUUID)
			if err != nil {
				return nil, 0, 0, err
			}
			mod, isImage, err := partTypeToProtoModality(partType)
			if err != nil {
				return nil, 0, 0, err
			}
			if isImage {
				imageCount++
			} else {
				otherCount++
			}
			mediaItem := &vllmpb.MediaItem{
				Modality: mod,
				Uuid:     itemUUID,
			}
			if len(rawURL) >= 5 && strings.EqualFold(rawURL[:5], "data:") {
				mediaItem.Source = &vllmpb.MediaItem_DataUri{DataUri: rawURL}
			} else {
				mediaItem.Source = &vllmpb.MediaItem_Url{Url: rawURL}
			}
			items = append(items, mediaItem)
		}
	}

	return items, imageCount, otherCount, nil
}

func convertHTTPFeaturesToMedia(features map[string]any, tokenCount int) ([]*vllmpb.MediaItem, int, int, error) {
	hashesMap, ok, err := decodeOptionalObjectField(features, "mm_hashes")
	if err != nil {
		return nil, 0, 0, err
	}
	if !ok || len(hashesMap) == 0 {
		return nil, 0, 0, nil
	}
	placeholdersMap, hasPlaceholders, err := decodeOptionalObjectField(features, "mm_placeholders")
	if err != nil {
		return nil, 0, 0, err
	}
	kwargsMap, hasKwargsMap, err := decodeOptionalObjectField(features, "kwargs_data")
	if err != nil {
		return nil, 0, 0, err
	}

	modalities := make([]string, 0, len(hashesMap))
	for k, v := range hashesMap {
		if k == "" {
			return nil, 0, 0, errors.New("features.mm_hashes has an empty modality key")
		}
		if v == nil {
			continue
		}
		modalities = append(modalities, k)
	}
	slices.Sort(modalities)

	var (
		items      []*vllmpb.MediaItem
		imageCount int
		otherCount int
	)
	for _, modKey := range modalities {
		protoMod, isImage, err := modalityKeyToProto(modKey)
		if err != nil {
			return nil, 0, 0, err
		}
		rawHashes, ok := hashesMap[modKey].([]any)
		if !ok {
			return nil, 0, 0, errors.New("features.mm_hashes modality value must be an array")
		}
		n := len(rawHashes)
		if n == 0 {
			continue
		}
		if !hasPlaceholders {
			return nil, 0, 0, errors.New("features.mm_placeholders is required when mm_hashes is set")
		}
		rawPlaceholders, ok := placeholdersMap[modKey].([]any)
		if !ok || len(rawPlaceholders) != n {
			return nil, 0, 0, errors.New("features.mm_placeholders length must match mm_hashes")
		}
		var rawKwargs []any
		hasModKwargs := false
		if hasKwargsMap && kwargsMap[modKey] != nil {
			rawKwargs, ok = kwargsMap[modKey].([]any)
			if !ok || len(rawKwargs) != n {
				return nil, 0, 0, errors.New("features.kwargs_data length must match mm_hashes")
			}
			hasModKwargs = true
		}

		for i := 0; i < n; i++ {
			hash, ok := rawHashes[i].(string)
			if !ok {
				return nil, 0, 0, errors.New("features.mm_hashes entries must be strings")
			}
			pMap, ok := rawPlaceholders[i].(map[string]any)
			if !ok {
				return nil, 0, 0, errors.New("features.mm_placeholders entries must be objects")
			}
			offset, err := reqcommon.NonNegativeInt(pMap["offset"])
			if err != nil {
				return nil, 0, 0, errors.New("features.mm_placeholders.offset must be a non-negative integer")
			}
			length, err := reqcommon.NonNegativeInt(pMap["length"])
			if err != nil {
				return nil, 0, 0, errors.New("features.mm_placeholders.length must be a non-negative integer")
			}
			if tokenCount >= 0 && (offset >= tokenCount || length > tokenCount-offset) {
				return nil, 0, 0, errors.New("features.mm_placeholders span exceeds token_ids length")
			}

			var kwargsBytes []byte
			if hasModKwargs && rawKwargs[i] != nil {
				kwargStr, ok := rawKwargs[i].(string)
				if !ok {
					return nil, 0, 0, errors.New("features.kwargs_data entries must be strings or null")
				}
				if kwargStr != "" {
					kwargsBytes, err = decodeBase64Kwargs(kwargStr)
					if err != nil {
						return nil, 0, 0, errors.New("features.kwargs_data entry is not valid base64")
					}
				}
			}

			if isImage {
				imageCount++
			} else {
				otherCount++
			}
			items = append(items, &vllmpb.MediaItem{
				Modality: protoMod,
				Uuid:     hash,
				Source: &vllmpb.MediaItem_Features{
					Features: &vllmpb.PreprocessedMediaFeatures{
						Kwargs:     kwargsBytes,
						Identifier: hash,
						Offset:     uint64(offset), //#nosec G115 -- NonNegativeInt verified >= 0
						Length:     uint64(length), //#nosec G115 -- NonNegativeInt verified >= 0
						MmHash:     proto.String(hash),
					},
				},
			})
		}
	}
	return items, imageCount, otherCount, nil
}

func decodeBase64Kwargs(s string) ([]byte, error) {
	b, err := base64.StdEncoding.DecodeString(s)
	if err == nil {
		return b, nil
	}
	return base64.RawStdEncoding.DecodeString(s)
}

func modalityKeyToProto(mod string) (vllmpb.Modality, bool, error) {
	switch reqcommon.Modality(mod) {
	case reqcommon.ModalityImage:
		return vllmpb.Modality_MODALITY_IMAGE, true, nil
	case reqcommon.ModalityVideo:
		return vllmpb.Modality_MODALITY_VIDEO, false, nil
	case reqcommon.ModalityAudio:
		return vllmpb.Modality_MODALITY_AUDIO, false, nil
	default:
		return vllmpb.Modality_MODALITY_UNSPECIFIED, false, errors.New("unsupported multimodal modality")
	}
}

func partTypeToProtoModality(partType string) (vllmpb.Modality, bool, error) {
	switch partType {
	case reqcommon.PartTypeImageURL:
		return vllmpb.Modality_MODALITY_IMAGE, true, nil
	case reqcommon.PartTypeVideoURL:
		return vllmpb.Modality_MODALITY_VIDEO, false, nil
	case reqcommon.PartTypeAudioURL:
		return vllmpb.Modality_MODALITY_AUDIO, false, nil
	default:
		return vllmpb.Modality_MODALITY_UNSPECIFIED, false, errors.New("unsupported content_parts type")
	}
}

func parseUint32Array(raw any, fieldName string, allowEmpty bool) ([]uint32, error) {
	arr, ok := raw.([]any)
	if !ok {
		return nil, fmt.Errorf("%s must be an array", fieldName)
	}
	if !allowEmpty && len(arr) == 0 {
		return nil, fmt.Errorf("%s must not be empty", fieldName)
	}
	out := make([]uint32, 0, len(arr))
	for _, elem := range arr {
		n, err := reqcommon.NonNegativeInt(elem)
		if err != nil || n > math.MaxUint32 {
			return nil, fmt.Errorf("%s elements must be uint32 integers", fieldName)
		}
		out = append(out, uint32(n)) //#nosec G115 -- guarded by [0, MaxUint32] above
	}
	return out, nil
}

func parseStopStrings(raw any) ([]string, error) {
	switch v := raw.(type) {
	case string:
		if v == "" {
			return nil, nil
		}
		return []string{v}, nil
	case []any:
		out := make([]string, 0, len(v))
		for _, elem := range v {
			s, ok := elem.(string)
			if !ok {
				return nil, errors.New("sampling_params.stop elements must be strings")
			}
			out = append(out, s)
		}
		return out, nil
	default:
		return nil, errors.New("sampling_params.stop must be a string or array of strings")
	}
}

func unwrapRawJSONValue(raw any) (any, bool, error) {
	if raw == nil {
		return nil, false, nil
	}
	rm, ok := raw.(json.RawMessage)
	if !ok {
		return raw, true, nil
	}
	trimmed := bytes.TrimSpace(rm)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil, false, nil
	}
	dec := json.NewDecoder(bytes.NewReader(trimmed))
	dec.UseNumber()
	var decoded any
	if err := dec.Decode(&decoded); err != nil {
		return nil, false, err
	}
	if decoded == nil {
		return nil, false, nil
	}
	return decoded, true, nil
}

func decodeOptionalStringField(m map[string]any, field string) (string, bool, error) {
	raw, present := m[field]
	if !present || raw == nil {
		return "", false, nil
	}
	val, hasVal, err := unwrapRawJSONValue(raw)
	if err != nil {
		return "", false, fmt.Errorf("%s must be a string", field)
	}
	if !hasVal {
		return "", false, nil
	}
	s, ok := val.(string)
	if !ok {
		return "", false, fmt.Errorf("%s must be a string", field)
	}
	return s, true, nil
}

func decodeOptionalBoolField(m map[string]any, field string) (bool, bool, error) {
	raw, present := m[field]
	if !present || raw == nil {
		return false, false, nil
	}
	val, hasVal, err := unwrapRawJSONValue(raw)
	if err != nil {
		return false, false, fmt.Errorf("%s must be a boolean", field)
	}
	if !hasVal {
		return false, false, nil
	}
	b, ok := val.(bool)
	if !ok {
		return false, false, fmt.Errorf("%s must be a boolean", field)
	}
	return b, true, nil
}

func decodeOptionalObjectField(m map[string]any, field string) (map[string]any, bool, error) {
	raw, present := m[field]
	if !present || raw == nil {
		return nil, false, nil
	}
	val, hasVal, err := unwrapRawJSONValue(raw)
	if err != nil {
		return nil, false, fmt.Errorf("%s must be an object", field)
	}
	if !hasVal {
		return nil, false, nil
	}
	obj, ok := val.(map[string]any)
	if !ok {
		return nil, false, fmt.Errorf("%s must be an object", field)
	}
	return obj, true, nil
}

func decodeOptionalArrayField(m map[string]any, field string) ([]any, bool, error) {
	raw, present := m[field]
	if !present || raw == nil {
		return nil, false, nil
	}
	val, hasVal, err := unwrapRawJSONValue(raw)
	if err != nil {
		return nil, false, fmt.Errorf("%s must be an array", field)
	}
	if !hasVal {
		return nil, false, nil
	}
	arr, ok := val.([]any)
	if !ok {
		return nil, false, fmt.Errorf("%s must be an array", field)
	}
	return arr, true, nil
}

func decodeOptionalFloat32Field(m map[string]any, field string) (float32, bool, error) {
	raw, present := m[field]
	if !present || raw == nil {
		return 0, false, nil
	}
	val, hasVal, err := unwrapRawJSONValue(raw)
	if err != nil {
		return 0, false, fmt.Errorf("%s must be a number", field)
	}
	if !hasVal {
		return 0, false, nil
	}
	var f float64
	switch n := val.(type) {
	case float64:
		f = n
	case json.Number:
		f, err = n.Float64()
		if err != nil {
			return 0, false, fmt.Errorf("%s must be a number", field)
		}
	case int:
		f = float64(n)
	default:
		return 0, false, fmt.Errorf("%s must be a number", field)
	}
	if math.IsNaN(f) || math.IsInf(f, 0) || math.Abs(f) > math.MaxFloat32 {
		return 0, false, fmt.Errorf("%s out of range", field)
	}
	return float32(f), true, nil
}

func decodeOptionalInt64Field(m map[string]any, field string) (int64, bool, error) {
	raw, present := m[field]
	if !present || raw == nil {
		return 0, false, nil
	}
	val, hasVal, err := unwrapRawJSONValue(raw)
	if err != nil {
		return 0, false, fmt.Errorf("%s must be an integer", field)
	}
	if !hasVal {
		return 0, false, nil
	}
	switch n := val.(type) {
	case int:
		return int64(n), true, nil
	case int64:
		return n, true, nil
	case float64:
		if n != math.Trunc(n) || n < math.MinInt64 || n >= math.MaxInt64 {
			return 0, false, fmt.Errorf("%s must be an integer", field)
		}
		return int64(n), true, nil
	case json.Number:
		i, err := n.Int64()
		if err != nil {
			return 0, false, fmt.Errorf("%s must be an integer", field)
		}
		return i, true, nil
	default:
		return 0, false, fmt.Errorf("%s must be an integer", field)
	}
}

func decodeOptionalInt32Field(m map[string]any, field string) (int32, bool, error) {
	i64, ok, err := decodeOptionalInt64Field(m, field)
	if err != nil || !ok {
		return 0, ok, err
	}
	if i64 < math.MinInt32 || i64 > math.MaxInt32 {
		return 0, false, fmt.Errorf("%s out of range", field)
	}
	return int32(i64), true, nil //#nosec G115 -- guarded by [MinInt32, MaxInt32] above
}

func decodeOptionalUint32Field(m map[string]any, field string) (uint32, bool, error) {
	i64, ok, err := decodeOptionalInt64Field(m, field)
	if err != nil || !ok {
		return 0, ok, err
	}
	if i64 < 0 || i64 > math.MaxUint32 {
		return 0, false, fmt.Errorf("%s must be a non-negative uint32 integer", field)
	}
	return uint32(i64), true, nil //#nosec G115 -- guarded by [0, MaxUint32] above
}
