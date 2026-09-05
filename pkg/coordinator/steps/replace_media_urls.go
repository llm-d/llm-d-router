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
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"maps"
	"math"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"syscall"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"

	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
	"golang.org/x/sync/errgroup"
)

const ReplaceMediaURLsStepName = "replace-media-urls"

// OpenAI chat content-part type strings the coordinator recognizes as
// multimodal. URL-based parts (image_url, audio_url, video_url) carry a
// `url` field that may be an https URL or a data URI. input_audio is
// always inline: `data` (base64) + `format` (audio codec name).
const (
	imageURLPartType   = "image_url"
	audioURLPartType   = "audio_url"
	videoURLPartType   = "video_url"
	inputAudioPartType = "input_audio"
)

// partTypeModality maps each recognized content-part type to the Modality
// it represents. Parts whose type is not in this map (text, tool_use,
// image_embeds, unknown types) are passed through untouched.
var partTypeModality = map[string]string{
	imageURLPartType:   ModalityImage,
	audioURLPartType:   ModalityAudio,
	videoURLPartType:   ModalityVideo,
	inputAudioPartType: ModalityAudio,
}

// mediaPartIsWellFormed reports whether partMap carries the fields
// replace_media_urls needs to produce a MultimodalEntry for it. Downstream
// steps that pair entries with parts (encode.collectMediaParts,
// decode.injectUUIDs) must apply the same predicate so entry i pairs with
// the i-th well-formed part of its modality; a silently-skipped part must
// not shift the pairing.
func mediaPartIsWellFormed(partMap map[string]any, partType string) bool {
	inner, ok := partMap[partType].(map[string]any)
	if !ok {
		return false
	}
	if partType == inputAudioPartType {
		data, _ := inner["data"].(string)
		return data != ""
	}
	_, ok = inner["url"].(string)
	return ok
}

const defaultContentType = "application/octet-stream"

// defaultMaxDownloadSize is the default cap for max_download_size, in megabytes.
const defaultMaxDownloadSize = 10 // 10 MB

func init() {
	pipeline.Register(ReplaceMediaURLsStepName, NewReplaceMediaURLsStep)
}

type ReplaceMediaURLsStep struct {
	downloadTimeout        time.Duration
	maxConcurrentDownloads int
	maxMultimodalEntries   int
	// maxDownloadSize is the default cap applied when a modality has no
	// per-modality override. Also the fallback used by input_audio size
	// validation when max_audio_download_size is unset.
	maxDownloadSize int64
	// maxDownloadSizeByMod optionally overrides maxDownloadSize per modality
	// (keys are the ModalityImage / ModalityAudio / ModalityVideo constants).
	// A missing key means "fall back to maxDownloadSize".
	maxDownloadSizeByMod map[string]int64
	// allowedContentTypes is the per-modality MIME allowlist applied to
	// data URIs and to input_audio inline items. Keys are Modality* constants;
	// values are lowercase MIME strings.
	allowedContentTypes map[string]map[string]struct{}
	guard               *addressGuard
	client              *http.Client
}

func NewReplaceMediaURLsStep(_ *gateway.Client, params map[string]any) (pipeline.Step, error) {
	timeout := 10 * time.Second
	if v, ok, err := paramDuration(params, "download_timeout"); err != nil {
		return nil, err
	} else if ok {
		timeout = v
	}

	maxConcurrent := 10
	if v, ok, err := paramInt(params, "max_concurrent_downloads"); err != nil {
		return nil, err
	} else if ok {
		if v <= 0 {
			return nil, fmt.Errorf("max_concurrent_downloads must be positive, got %d", v)
		}
		maxConcurrent = v
	}

	maxEntries := 0
	if v, ok, err := paramInt(params, "max_multimodal_entries"); err != nil {
		return nil, err
	} else if ok {
		if v < 0 {
			return nil, fmt.Errorf("max_multimodal_entries must be non-negative, got %d", v)
		}
		maxEntries = v
	}

	maxDownloadSize := int64(defaultMaxDownloadSize) * config.BytesPerMB
	if v, ok, err := paramInt(params, "max_download_size"); err != nil {
		return nil, err
	} else if ok {
		// Guard against overflow: maxDownloadSize+1 is used as the io.LimitReader
		// sentinel; an MB value that overflows int64 when converted to bytes would
		// cause LimitReader to receive a negative limit and return immediate EOF.
		if v <= 0 || v > (math.MaxInt-1)/config.BytesPerMB {
			return nil, fmt.Errorf("max_download_size must be positive and at most %d MB, got %d", (math.MaxInt-1)/config.BytesPerMB, v)
		}
		maxDownloadSize = int64(v) * config.BytesPerMB
	}

	// Optional per-modality download caps. Each param, when set, overrides
	// maxDownloadSize for URLs / inline payloads of that modality. Same
	// units (MB) and same overflow guard as max_download_size.
	maxDownloadSizeByMod, err := parsePerModalityDownloadSizes(params)
	if err != nil {
		return nil, err
	}

	// Optional per-modality MIME allowlist overrides. Each param, when set,
	// replaces the built-in default set for that modality.
	allowedContentTypes, err := parsePerModalityContentTypes(params)
	if err != nil {
		return nil, err
	}

	guard := &addressGuard{}
	if v, ok, err := paramBool(params, "allow_private_networks"); err != nil {
		return nil, err
	} else if ok {
		guard.allowPrivate = v
	}
	if raw, present := params["allowed_domains"]; present {
		domains, err := parseAllowedDomains(raw)
		if err != nil {
			return nil, err
		}
		guard.allowedDomains = domains
	}

	step := &ReplaceMediaURLsStep{
		downloadTimeout:        timeout,
		maxConcurrentDownloads: maxConcurrent,
		maxMultimodalEntries:   maxEntries,
		maxDownloadSize:        maxDownloadSize,
		maxDownloadSizeByMod:   maxDownloadSizeByMod,
		allowedContentTypes:    allowedContentTypes,
		guard:                  guard,
	}
	step.client = guard.newClient(timeout)
	return step, nil
}

func (s *ReplaceMediaURLsStep) Name() string { return ReplaceMediaURLsStepName }

func (s *ReplaceMediaURLsStep) Execute(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	logger := log.FromContext(ctx).WithName(ReplaceMediaURLsStepName)

	messages, ok := reqCtx.Body["messages"].([]any)
	if !ok {
		return nil
	}

	// Collect every media part into one walker-order slice, tagged by kind.
	// MultimodalEntries and the download-result slots below both index by
	// this walker position, so encode.collectMediaParts and
	// decode.injectUUIDs (which walk parts in the same order) pair entry i
	// with the i-th part of its modality. Splitting URL and inline parts
	// into separate append passes would reorder audio entries whenever a
	// request mixes audio_url and input_audio.
	var refs []mediaRef
	for msgIdx, msg := range messages {
		msgMap, ok := msg.(map[string]any)
		if !ok {
			continue
		}
		content, ok := msgMap["content"].([]any)
		if !ok {
			continue
		}
		for partIdx, part := range content {
			partMap, ok := part.(map[string]any)
			if !ok {
				continue
			}
			partType, _ := partMap["type"].(string)
			modality, isMedia := partTypeModality[partType]
			if !isMedia {
				continue
			}
			if partType == inputAudioPartType {
				innerMap, ok := partMap[inputAudioPartType].(map[string]any)
				if !ok {
					continue
				}
				data, _ := innerMap["data"].(string)
				if data == "" {
					continue
				}
				format, _ := innerMap["format"].(string)
				refs = append(refs, mediaRef{
					msgIdx:   msgIdx,
					partIdx:  partIdx,
					modality: modality,
					isInline: true,
					data:     data,
					format:   format,
				})
				continue
			}
			// URL-based parts: image_url, audio_url, video_url
			innerMap, ok := partMap[partType].(map[string]any)
			if !ok {
				continue
			}
			url, ok := innerMap["url"].(string)
			if !ok {
				continue
			}
			refs = append(refs, mediaRef{
				msgIdx:   msgIdx,
				partIdx:  partIdx,
				modality: modality,
				url:      url,
				urlMap:   innerMap,
			})
		}
	}

	if len(refs) == 0 {
		return nil
	}

	if s.maxMultimodalEntries > 0 && len(refs) > s.maxMultimodalEntries {
		return fmt.Errorf("too many multimodal entries: got %d, max %d: %w", len(refs), s.maxMultimodalEntries, pipeline.ErrBadRequest)
	}

	// Cancel any in-flight downloads when Execute returns early (cancelled
	// context or a rejected data URI), so goroutines do not outlive the step.
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	g, gCtx := errgroup.WithContext(ctx)
	g.SetLimit(s.maxConcurrentDownloads)

	// results parallels refs: results[i] is the outcome of processing
	// refs[i]. URL refs may be filled synchronously (data URI) or by a
	// download goroutine; inline refs are validated in the walker-order
	// append pass after g.Wait.
	results := make([]mediaResult, len(refs))
	urlCount := 0
	for i, ref := range refs {
		if err := gCtx.Err(); err != nil {
			break
		}
		if ref.isInline {
			// Skip inline refs here; they are validated in the walker-order
			// pass below along with the download results.
			continue
		}
		urlCount++
		if strings.HasPrefix(ref.url, "data:") {
			contentType, b64, err := parseDataURI(ref.url)
			if err != nil {
				return fmt.Errorf("parsing data URI at message %d part %d: %w: %w", ref.msgIdx, ref.partIdx, err, pipeline.ErrBadRequest)
			}
			if !s.allowedContentTypeForModality(contentType, ref.modality) {
				return fmt.Errorf("data URI content type %q not allowed for %s at message %d part %d: %w", contentType, ref.modality, ref.msgIdx, ref.partIdx, pipeline.ErrBadRequest)
			}
			results[i] = mediaResult{base64Data: b64, contentType: contentType}
			continue
		}
		g.Go(func() error {
			data, contentType, err := s.download(gCtx, ref.url, ref.modality)
			if err != nil {
				return fmt.Errorf("downloading %s: %w", ref.url, err)
			}
			// Audio and video decoders have historically carried more CVEs
			// than image decoders, so the origin's Content-Type is checked
			// against the per-modality allowlist for those two. image_url
			// keeps its pre-existing permissive treatment on downloads (data
			// URIs are still checked).
			if ref.modality != ModalityImage && !s.allowedContentTypeForModality(contentType, ref.modality) {
				return fmt.Errorf("downloaded content type %q not allowed for %s at message %d part %d: %w",
					contentType, ref.modality, ref.msgIdx, ref.partIdx, pipeline.ErrBadRequest)
			}
			results[i] = mediaResult{
				base64Data:  base64.StdEncoding.EncodeToString(data),
				contentType: contentType,
			}
			return nil
		})
	}

	// Log proxy presence only: HTTP(S)_PROXY URLs can carry basic-auth
	// credentials (http://user:pass@host) that must not reach logs.
	logger.V(logutil.TRACE).Info("downloading media", "count", urlCount, "http_proxy_set", os.Getenv("HTTP_PROXY") != "", "https_proxy_set", os.Getenv("HTTPS_PROXY") != "")

	if err := g.Wait(); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	// Walker-order pass. URL refs get their data URI written back in
	// place; inline refs are validated (MIME + size) now that downloads
	// have settled. Every ref appends exactly one MultimodalEntry, in
	// walker order.
	for i, ref := range refs {
		if ref.isInline {
			contentType, err := audioFormatToMIME(ref.format)
			if err != nil {
				return fmt.Errorf("input_audio at message %d part %d: %w: %w",
					ref.msgIdx, ref.partIdx, err, pipeline.ErrBadRequest)
			}
			if !s.allowedContentTypeForModality(contentType, ref.modality) {
				return fmt.Errorf("input_audio content type %q not allowed at message %d part %d: %w",
					contentType, ref.msgIdx, ref.partIdx, pipeline.ErrBadRequest)
			}
			// Padded base64 for n bytes has length 4 * ceil(n/3). Compute
			// the cap and reject when the string alone exceeds it, so an
			// oversized payload is caught before decoding. input_audio size
			// is bounded by the audio-modality cap.
			sizeCap := s.downloadSizeFor(ref.modality)
			maxBase64Len := 4 * ((sizeCap + 2) / 3)
			if int64(len(ref.data)) > maxBase64Len {
				return fmt.Errorf("input_audio at message %d part %d exceeds size limit: %w",
					ref.msgIdx, ref.partIdx, pipeline.ErrBadRequest)
			}
			// The inline "data" field already carries the base64 payload
			// the backend needs, so the body passes through unmodified.
			appendMultimodalEntry(reqCtx, ref.modality, contentType, ref.data)
			continue
		}
		r := results[i]
		// r.contentType is set on every URL ref the download/parse path
		// populated. Skip a zero-valued slot defensively so a future path
		// that leaves results[i] unset after g.Wait cannot panic on a
		// nil-map write.
		if ref.urlMap == nil {
			continue
		}
		if !strings.HasPrefix(ref.url, "data:") {
			ref.urlMap["url"] = fmt.Sprintf("data:%s;base64,%s", r.contentType, r.base64Data)
		}
		appendMultimodalEntry(reqCtx, ref.modality, r.contentType, r.base64Data)
	}

	return nil
}

func appendMultimodalEntry(reqCtx *pipeline.RequestContext, modality, contentType, b64 string) {
	reqCtx.MultimodalEntries = append(reqCtx.MultimodalEntries, pipeline.MultimodalEntry{
		Index:       len(reqCtx.MultimodalEntries),
		Modality:    modality,
		Base64Data:  b64,
		ContentType: contentType,
	})
}

func (s *ReplaceMediaURLsStep) download(ctx context.Context, rawURL, modality string) ([]byte, string, error) {
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return nil, "", fmt.Errorf("invalid URL: %w: %w", err, pipeline.ErrBadRequest)
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return nil, "", fmt.Errorf("scheme %q not allowed: %w", parsed.Scheme, pipeline.ErrBadRequest)
	}
	if !s.guard.hostAllowed(parsed.Hostname()) {
		return nil, "", fmt.Errorf("host %q not allowed: %w", parsed.Hostname(), pipeline.ErrBadRequest)
	}

	sizeCap := s.downloadSizeFor(modality)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, "", err
	}
	call := coordmetrics.StartUpstreamCall(coordmetrics.UpstreamReplaceMediaURLs)
	resp, err := s.client.Do(req)
	call.Done()
	if err != nil {
		return nil, "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody := readErrorBody(resp.Body)
		return nil, "", upstreamError(ReplaceMediaURLsStepName, resp.StatusCode, respBody)
	}

	if resp.ContentLength > sizeCap {
		return nil, "", fmt.Errorf("response too large: Content-Length %d exceeds max %d: %w", resp.ContentLength, sizeCap, pipeline.ErrBadRequest)
	}

	data, err := io.ReadAll(io.LimitReader(resp.Body, sizeCap+1))
	if err != nil {
		return nil, "", err
	}
	if int64(len(data)) > sizeCap {
		return nil, "", fmt.Errorf("response too large: body exceeds max %d: %w", sizeCap, pipeline.ErrBadRequest)
	}
	contentType := resp.Header.Get("Content-Type")
	if contentType == "" {
		contentType = defaultContentType
	}
	return data, contentType, nil
}

// mediaRef locates one media content part in the request body, tagged by
// kind. isInline discriminates the two variants:
//
//   - URL-based parts (image_url / audio_url / video_url) set isInline=false
//     and fill url + urlMap. urlMap is the inner map carrying the "url" key
//     so the download result can be inlined in place.
//   - input_audio parts set isInline=true and fill data + format. Their
//     base64 payload is already inline in the request body; only MIME and
//     size validation are needed.
//
// Refs are collected in walker order so MultimodalEntries append order
// matches encode.collectMediaParts / decode.injectUUIDs walk order.
type mediaRef struct {
	msgIdx   int
	partIdx  int
	modality string
	isInline bool
	// URL variant:
	url    string
	urlMap map[string]any
	// Inline variant:
	data   string // base64 payload
	format string // "wav", "mp3", ...
}

// mediaResult carries the outcome of processing one URL-based mediaRef:
// the MIME type and base64 payload the entry ends up carrying. Inline
// refs do not use a result slot; their fields are validated and appended
// straight to MultimodalEntries in the walker-order pass.
type mediaResult struct {
	base64Data  string
	contentType string
}

// defaultAllowedContentTypesByModality is the built-in per-modality MIME
// allowlist applied to data URIs and to input_audio inline items. Operators
// override individual modality sets via the allowed_{image,audio,video}_content_types
// params; a modality with no override falls back to the entry here. The map
// is intentionally permissive for common containers; codec-level restrictions
// are the backend's job.
var defaultAllowedContentTypesByModality = map[string]map[string]struct{}{
	ModalityImage: {
		"image/jpeg": {},
		"image/png":  {},
		"image/gif":  {},
		"image/webp": {},
	},
	ModalityAudio: {
		"audio/wav":    {},
		"audio/x-wav":  {},
		"audio/mpeg":   {},
		"audio/mp3":    {},
		"audio/flac":   {},
		"audio/x-flac": {},
		"audio/ogg":    {},
		"audio/opus":   {},
		"audio/webm":   {},
	},
	ModalityVideo: {
		"video/mp4":       {},
		"video/webm":      {},
		"video/quicktime": {},
		"video/mpeg":      {},
		"video/ogg":       {},
	},
}

// allowedContentTypeForModality reports whether contentType is allowed for
// modality per the step's configured allowlist. Comparison is
// case-insensitive with whitespace trimmed.
func (s *ReplaceMediaURLsStep) allowedContentTypeForModality(contentType, modality string) bool {
	allowed, ok := s.allowedContentTypes[modality]
	if !ok {
		return false
	}
	_, ok = allowed[strings.ToLower(strings.TrimSpace(contentType))]
	return ok
}

// downloadSizeFor returns the per-modality download cap when the operator set
// one, else the global default. Callers use this both for HTTP downloads
// (Content-Length + LimitReader bound) and for input_audio inline size checks.
func (s *ReplaceMediaURLsStep) downloadSizeFor(modality string) int64 {
	if v, ok := s.maxDownloadSizeByMod[modality]; ok {
		return v
	}
	return s.maxDownloadSize
}

// perModalityDownloadSizeParams names the config keys that override
// max_download_size on a per-modality basis. Ordering matches Modality*.
var perModalityDownloadSizeParams = map[string]string{
	ModalityImage: "max_image_download_size",
	ModalityAudio: "max_audio_download_size",
	ModalityVideo: "max_video_download_size",
}

// parsePerModalityDownloadSizes reads the three optional per-modality cap
// params. Each uses the same validation as max_download_size (positive,
// bounded by MaxInt/BytesPerMB). Returns nil when no override is set so the
// step can distinguish "no override" from "override set to zero".
func parsePerModalityDownloadSizes(params map[string]any) (map[string]int64, error) {
	var out map[string]int64
	for mod, key := range perModalityDownloadSizeParams {
		v, ok, err := paramInt(params, key)
		if err != nil {
			return nil, err
		}
		if !ok {
			continue
		}
		if v <= 0 || v > (math.MaxInt-1)/config.BytesPerMB {
			return nil, fmt.Errorf("%s must be positive and at most %d MB, got %d", key, (math.MaxInt-1)/config.BytesPerMB, v)
		}
		if out == nil {
			out = make(map[string]int64, len(perModalityDownloadSizeParams))
		}
		out[mod] = int64(v) * config.BytesPerMB
	}
	return out, nil
}

// perModalityContentTypeParams names the config keys that override the
// built-in MIME allowlist on a per-modality basis.
var perModalityContentTypeParams = map[string]string{
	ModalityImage: "allowed_image_content_types",
	ModalityAudio: "allowed_audio_content_types",
	ModalityVideo: "allowed_video_content_types",
}

// parsePerModalityContentTypes builds the final per-modality allowlist map,
// starting from defaultAllowedContentTypesByModality and replacing any
// modality whose config param is set. Each override is a list of MIME
// strings; non-list roots or non-string entries are rejected so a
// misconfiguration fails loudly instead of silently disabling the check.
func parsePerModalityContentTypes(params map[string]any) (map[string]map[string]struct{}, error) {
	out := make(map[string]map[string]struct{}, len(defaultAllowedContentTypesByModality))
	for mod, set := range defaultAllowedContentTypesByModality {
		// Clone so a caller mutating the returned per-modality set never
		// leaks into defaultAllowedContentTypesByModality (visible to every
		// future ReplaceMediaURLsStep in the process).
		out[mod] = maps.Clone(set)
	}
	for mod, key := range perModalityContentTypeParams {
		raw, present := params[key]
		if !present || raw == nil {
			continue
		}
		types, err := parseContentTypeSet(raw, key)
		if err != nil {
			return nil, err
		}
		out[mod] = types
	}
	return out, nil
}

// parseContentTypeSet accepts a list of MIME strings as either []any (YAML
// decode path) or []string (programmatic callers), returning them as a set
// keyed by lowercase MIME. fieldName is used only for error messages.
func parseContentTypeSet(raw any, fieldName string) (map[string]struct{}, error) {
	var entries []any
	switch v := raw.(type) {
	case []any:
		entries = v
	case []string:
		entries = make([]any, len(v))
		for i, s := range v {
			entries[i] = s
		}
	default:
		return nil, fmt.Errorf("%s must be a list of strings, got %T", fieldName, raw)
	}
	set := make(map[string]struct{}, len(entries))
	for _, e := range entries {
		mime, ok := e.(string)
		if !ok {
			return nil, fmt.Errorf("%s entries must be strings, got %T", fieldName, e)
		}
		set[strings.ToLower(strings.TrimSpace(mime))] = struct{}{}
	}
	return set, nil
}

// audioFormatMIME maps OpenAI's input_audio.format values to canonical MIME
// types. OpenAI documents "wav" and "mp3" today; the additional entries
// mirror what the backend commonly accepts.
var audioFormatMIME = map[string]string{
	"wav":  "audio/wav",
	"mp3":  "audio/mpeg",
	"flac": "audio/flac",
	"opus": "audio/opus",
	"ogg":  "audio/ogg",
	"webm": "audio/webm",
}

func audioFormatToMIME(format string) (string, error) {
	normalized := strings.ToLower(strings.TrimSpace(format))
	mime, ok := audioFormatMIME[normalized]
	if !ok {
		return "", fmt.Errorf("unsupported input_audio format %q", format)
	}
	return mime, nil
}

func parseDataURI(uri string) (contentType, b64 string, err error) {
	rest := strings.TrimPrefix(uri, "data:")
	meta, payload, ok := strings.Cut(rest, ",")
	if !ok {
		return "", "", errors.New("missing comma in data URI")
	}
	ct, params, _ := strings.Cut(meta, ";")
	hasBase64 := false
	for _, p := range strings.Split(params, ";") {
		if strings.EqualFold(strings.TrimSpace(p), "base64") {
			hasBase64 = true
			break
		}
	}
	if !hasBase64 {
		return "", "", errors.New("data URI must be base64-encoded")
	}
	if ct == "" {
		return "", "", errors.New("data URI missing media type")
	}
	return strings.ToLower(strings.TrimSpace(ct)), payload, nil
}

// addressGuard enforces SSRF protections for outbound image downloads. The IP
// check runs at dial time, so it covers every connection a single request
// makes, including each redirect hop. The hostname allowlist is enforced
// separately because a redirect target's hostname is only known per hop.
type addressGuard struct {
	allowPrivate   bool
	allowedDomains map[string]struct{}

	// allowLoopback relaxes the loopback block for in-package tests, whose
	// httptest servers bind to 127.0.0.1. Never set in production.
	allowLoopback bool
}

// errBlockedAddress marks a dial to a forbidden address. It wraps
// pipeline.ErrBadRequest so the connection failure surfaced by http.Client.Do
// (wrapped in *url.Error/*net.OpError, both of which Unwrap) classifies as a
// client 4xx rather than a 502.
var errBlockedAddress = fmt.Errorf("address resolves to a blocked range: %w", pipeline.ErrBadRequest)

// cgnatBlock is the RFC 6598 carrier-grade NAT range, which net.IP has no
// dedicated predicate for.
var cgnatBlock = &net.IPNet{IP: net.IPv4(100, 64, 0, 0), Mask: net.CIDRMask(10, 32)}

func (g *addressGuard) newClient(timeout time.Duration) *http.Client {
	// Clone DefaultTransport to keep Proxy: http.ProxyFromEnvironment, so image
	// fetches still honor HTTP(S)_PROXY, and attach the dial-time IP guard.
	transport := http.DefaultTransport.(*http.Transport).Clone()
	dialer := &net.Dialer{Control: g.dialControl}
	transport.DialContext = dialer.DialContext

	return &http.Client{
		Timeout:   timeout,
		Transport: transport,
		CheckRedirect: func(req *http.Request, _ []*http.Request) error {
			if !g.hostAllowed(req.URL.Hostname()) {
				return fmt.Errorf("redirect host %q not allowed: %w", req.URL.Hostname(), pipeline.ErrBadRequest)
			}
			return nil
		},
	}
}

// dialControl runs against the resolved IP the dialer is about to connect to,
// defeating DNS-rebinding bypasses that a hostname check would miss.
func (g *addressGuard) dialControl(_, address string, _ syscall.RawConn) error {
	host, _, err := net.SplitHostPort(address)
	if err != nil {
		return err
	}
	ip := net.ParseIP(host)
	if ip == nil {
		return fmt.Errorf("cannot parse dial address %q: %w", address, pipeline.ErrBadRequest)
	}
	if g.blockedIP(ip) {
		return errBlockedAddress
	}
	return nil
}

func (g *addressGuard) blockedIP(ip net.IP) bool {
	// Normalize IPv4-mapped IPv6 (e.g. ::ffff:169.254.169.254) so the IPv4
	// predicates below see the embedded address.
	if v4 := ip.To4(); v4 != nil {
		ip = v4
	}
	if ip.IsUnspecified() || ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() {
		return true
	}
	if ip.IsLoopback() {
		return !g.allowLoopback
	}
	if cgnatBlock.Contains(ip) {
		return true
	}
	if ip.IsPrivate() {
		// IsPrivate covers RFC1918 (IPv4) and unique-local fc00::/7 (IPv6).
		// Only RFC1918 is configurable; unique-local is never a valid image
		// origin and stays blocked even when allowPrivate is set.
		if ip.To4() != nil {
			return !g.allowPrivate
		}
		return true
	}
	return false
}

func (g *addressGuard) hostAllowed(host string) bool {
	if len(g.allowedDomains) == 0 {
		return true
	}
	_, ok := g.allowedDomains[strings.ToLower(host)]
	return ok
}

// parseAllowedDomains accepts a list of hostnames as either []any (the YAML
// decode path) or []string (programmatic callers). It returns an error on any
// other type rather than silently disabling the allowlist, which would be an
// open-by-default downgrade of a security control.
func parseAllowedDomains(raw any) (map[string]struct{}, error) {
	var entries []any
	switch v := raw.(type) {
	case []any:
		entries = v
	case []string:
		entries = make([]any, len(v))
		for i, s := range v {
			entries[i] = s
		}
	default:
		return nil, fmt.Errorf("allowed_domains must be a list of strings, got %T", raw)
	}

	domains := make(map[string]struct{}, len(entries))
	for _, e := range entries {
		host, ok := e.(string)
		if !ok {
			return nil, fmt.Errorf("allowed_domains entries must be strings, got %T", e)
		}
		domains[strings.ToLower(host)] = struct{}{}
	}
	return domains, nil
}
