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

// classifyMediaPart extracts the fields replace_media_urls needs from a media
// content part of the given type, returning them as a partially filled
// mediaRef (the caller supplies msgIdx and partIdx). ok is false when the part
// is missing those fields, and reason then names what was wrong so the caller
// can log it. partType must already be known to be a media type; modality is
// the one partTypeModality maps it to.
//
// mediaPartIsWellFormed wraps this function for callers that need only the
// yes/no answer, so every walker accepts and rejects the same parts.
// mediaPartIsWellFormed records why that has to hold.
func classifyMediaPart(partMap map[string]any, partType, modality string) (ref mediaRef, ok bool, reason string) {
	inner, isObject := partMap[partType].(map[string]any)
	if !isObject {
		return mediaRef{}, false, "inner object missing or wrong type"
	}
	if partType == inputAudioPartType {
		data, _ := inner["data"].(string)
		if data == "" {
			return mediaRef{}, false, "data is empty or not a string"
		}
		format, _ := inner["format"].(string)
		return mediaRef{modality: modality, isInline: true, data: data, format: format}, true, ""
	}
	// URL-based parts: image_url, audio_url, video_url.
	url, isString := inner["url"].(string)
	if !isString {
		return mediaRef{}, false, "url is missing or not a string"
	}
	return mediaRef{modality: modality, url: url, urlMap: inner}, true, ""
}

// mediaPartIsWellFormed reports whether partMap carries the fields
// replace_media_urls needs to produce a MultimodalEntry for it.
//
// How entries and parts stay lined up:
// replace_media_urls creates one entry for each media part that passes
// this check, in the order the parts appear in the request. Later
// steps (encode.collectMediaParts, the encode fanout, decode.injectUUIDs)
// walk the same request in the same order, apply the same check, and
// pair the Nth entry of a modality with the Nth part of that modality.
// If a bad part is dropped here but kept there (or the other way
// around), the pairing shifts and the wrong bytes get attached to the
// wrong entry. That is why this delegates to classifyMediaPart rather
// than restating the rule: replace_media_urls, which fixes the order,
// runs the same code as the walkers that follow it.
func mediaPartIsWellFormed(partMap map[string]any, partType string) bool {
	_, ok, _ := classifyMediaPart(partMap, partType, partTypeModality[partType])
	return ok
}

const defaultContentType = "application/octet-stream"

// normalizeMediaType reduces a Content-Type header value or a data URI media
// type to the bare type, lowercased and trimmed. Everything from the first
// ";" (MIME parameters such as codecs= or charset=) or the first "," is
// dropped, and the result is what both the allowlist lookup and the emitted
// data URI use.
//
// Cutting at the comma keeps the emitted URI parseable. RFC 2397 ends the
// metadata at the first comma, so a type carrying one (a header holding more
// than one value, or a codec list) would move the payload boundary and hand
// the reader a truncated media type with the rest of the metadata prepended
// to the base64.
//
// Returns "" for an empty value and for one that is only parameters
// ("; charset=utf-8"). Callers that need a type substitute defaultContentType,
// which keeps the empty case off the wire: "data:;base64,..." has no media
// type at all.
func normalizeMediaType(raw string) string {
	if i := strings.IndexAny(raw, ";,"); i >= 0 {
		raw = raw[:i]
	}
	return strings.ToLower(strings.TrimSpace(raw))
}

// dataURIPrefix is the scheme prefix of a data URI. Scheme names are
// case-insensitive (RFC 3986 section 3.1), so every comparison against it goes
// through isDataURI rather than strings.HasPrefix.
const dataURIPrefix = "data:"

// isDataURI reports whether s carries the data: scheme, ignoring case.
// A case-sensitive check would send "DATA:image/png;base64,..." down the
// download path, where the scheme guard rejects anything that is not http(s).
//
// The payload is forwarded with the client's casing intact. That is safe
// because vLLM resolves the scheme through urlparse, which lowercases it
// before the comparison exactly as url.Parse does here.
func isDataURI(s string) bool {
	return len(s) >= len(dataURIPrefix) && strings.EqualFold(s[:len(dataURIPrefix)], dataURIPrefix)
}

// defaultMaxDownloadSize is the default cap for max_download_size, in megabytes.
const defaultMaxDownloadSize = 10 // 10 MB

func init() {
	pipeline.Register(ReplaceMediaURLsStepName, NewReplaceMediaURLsStep)
}

type ReplaceMediaURLsStep struct {
	downloadTimeout        time.Duration
	maxConcurrentDownloads int
	maxMultimodalEntries   int
	// maxDownloadSize is the cap applied when a modality has no per-modality
	// override. Reached through downloadSizeFor.
	maxDownloadSize int64
	// maxDownloadSizeByMod optionally overrides maxDownloadSize per modality
	// (keys are the ModalityImage / ModalityAudio / ModalityVideo constants).
	// A missing key means "fall back to maxDownloadSize".
	maxDownloadSizeByMod map[string]int64
	// allowedContentTypes is the per-modality MIME allowlist applied to
	// data URIs and to input_audio inline items. Keys are Modality* constants;
	// values are lowercase MIME strings.
	allowedContentTypes map[string]map[string]struct{}
	// contentTypeOverrides records which modalities the operator configured
	// an allowed_<modality>_content_types param for, so the step can tell an
	// explicit allowlist from the built-in default. Only the image modality
	// reads it; see enforceDownloadContentType.
	contentTypeOverrides map[string]struct{}
	guard                *addressGuard
	client               *http.Client
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
	// replaces the built-in default set for that modality. overrides names
	// the modalities that were set explicitly.
	allowedContentTypes, overrides, err := parsePerModalityContentTypes(params)
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
		contentTypeOverrides:   overrides,
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

	// Collect every media part into one slice, in the order they appear
	// in the request, tagged by kind. The request order matters for
	// pairing entries with parts later; see mediaPartIsWellFormed.
	// Splitting URL and inline parts into two passes would reorder audio
	// entries whenever a request mixes audio_url and input_audio.
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
			ref, ok, reason := classifyMediaPart(partMap, partType, modality)
			if !ok {
				logger.V(logutil.DEBUG).Info("skipping malformed media part",
					"reason", reason, "msg_index", msgIdx, "part_index", partIdx, "part_type", partType)
				continue
			}
			ref.msgIdx = msgIdx
			ref.partIdx = partIdx
			refs = append(refs, ref)
		}
	}

	if len(refs) == 0 {
		return nil
	}

	if s.maxMultimodalEntries > 0 && len(refs) > s.maxMultimodalEntries {
		return fmt.Errorf("too many multimodal entries: got %d, max %d: %w", len(refs), s.maxMultimodalEntries, pipeline.ErrBadRequest)
	}

	// Validate every inline input_audio ref before a single download starts.
	// These checks are local (format name, MIME allowlist, payload length),
	// so running them first means a request that is going to be rejected
	// anyway does not first pull megabytes over the network for its
	// audio_url / video_url refs. Doing this inside the download loop below
	// would not help: an inline ref late in walker order would still land
	// after earlier downloads had been kicked off.
	for _, ref := range refs {
		if !ref.isInline {
			continue
		}
		if err := s.validateInlineAudio(ref); err != nil {
			return err
		}
	}

	// Cancel any in-flight downloads when Execute returns early (cancelled
	// context or a rejected data URI), so goroutines do not outlive the step.
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	g, gCtx := errgroup.WithContext(ctx)
	g.SetLimit(s.maxConcurrentDownloads)

	// dataURIs parallels refs: dataURIs[i] is the rewritten "url" value for
	// refs[i], built by that ref's download goroutine. Only downloaded refs
	// get a slot. Inline input_audio refs, and URL refs that already are data
	// URIs, carry their payload in the body already and leave the slot empty.
	dataURIs := make([]string, len(refs))
	downloadCount := 0
	for i, ref := range refs {
		if err := gCtx.Err(); err != nil {
			break
		}
		if ref.isInline {
			continue
		}
		if isDataURI(ref.url) {
			// Validate only. The slot is already the payload, so there is
			// nothing to rewrite and nothing worth retaining.
			contentType, b64, err := parseDataURI(ref.url)
			if err != nil {
				return fmt.Errorf("parsing data URI at message %d part %d: %w: %w", ref.msgIdx, ref.partIdx, err, pipeline.ErrBadRequest)
			}
			if !s.allowedContentTypeForModality(contentType, ref.modality) {
				return fmt.Errorf("data URI content type %q not allowed for %s at message %d part %d: %w", contentType, ref.modality, ref.msgIdx, ref.partIdx, pipeline.ErrBadRequest)
			}
			if s.enforceInlineSize(ref.modality) && s.inlineSizeExceeded(b64, ref.modality) {
				return fmt.Errorf("data URI at message %d part %d exceeds size limit for %s: %w",
					ref.msgIdx, ref.partIdx, ref.modality, pipeline.ErrBadRequest)
			}
			continue
		}
		downloadCount++
		g.Go(func() error {
			data, contentType, err := s.download(gCtx, ref.url, ref.modality)
			if err != nil {
				return fmt.Errorf("downloading %s at message %d part %d: %w", ref.url, ref.msgIdx, ref.partIdx, err)
			}
			// Encode to the final data URI here rather than stashing the
			// base64 for the walker pass below to wrap: holding both the
			// payload and the URI containing it doubles what the step
			// retains, and that retention is per request, not per download.
			dataURIs[i] = encodeDataURI(contentType, data)
			return nil
		})
	}

	// Log proxy presence only: HTTP(S)_PROXY URLs can carry basic-auth
	// credentials (http://user:pass@host) that must not reach logs.
	// count is the number of refs that go over the network. Data URIs and
	// inline input_audio are excluded, so the number can be read against
	// egress volume.
	logger.V(logutil.TRACE).Info("downloading media", "count", downloadCount, "http_proxy_set", os.Getenv("HTTP_PROXY") != "", "https_proxy_set", os.Getenv("HTTPS_PROXY") != "")

	if err := g.Wait(); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	// Walk refs in the order they appeared in the request: rewrite URL
	// slots in place and add one MultimodalEntry per ref. Entry order is
	// the whole point of this pass; see mediaPartIsWellFormed.
	for i, ref := range refs {
		// A non-empty slot is exactly a ref that was downloaded. Inline
		// input_audio refs and refs that were already data URIs need no
		// rewrite: their payload is in the body as it arrived.
		if dataURIs[i] != "" {
			ref.urlMap["url"] = dataURIs[i]
		}
		appendMultimodalEntry(reqCtx, ref.modality)
	}

	return nil
}

// validateInlineAudio checks one input_audio ref: its format must be a
// recognized audio codec name, the MIME that maps to must pass the audio
// allowlist, and its payload must fit the audio size cap. Nothing here
// touches the network or the request body, so Execute runs it on every
// inline ref before starting any download.
func (s *ReplaceMediaURLsStep) validateInlineAudio(ref mediaRef) error {
	contentType, err := audioFormatToMIME(ref.format)
	if err != nil {
		return fmt.Errorf("input_audio at message %d part %d: %w: %w",
			ref.msgIdx, ref.partIdx, err, pipeline.ErrBadRequest)
	}
	if !s.allowedContentTypeForModality(contentType, ref.modality) {
		// Name the format alongside the MIME it maps to. The allowlist is
		// written in MIME types and the request in formats, so an operator
		// reading the MIME alone cannot tell which format was refused.
		return fmt.Errorf("input_audio format %q (content type %q) not allowed at message %d part %d: %w",
			ref.format, contentType, ref.msgIdx, ref.partIdx, pipeline.ErrBadRequest)
	}
	// input_audio is capped by the audio modality.
	if s.inlineSizeExceeded(ref.data, ref.modality) {
		return fmt.Errorf("input_audio at message %d part %d exceeds size limit: %w",
			ref.msgIdx, ref.partIdx, pipeline.ErrBadRequest)
	}
	return nil
}

// inlineSizeExceeded reports whether a base64 payload that arrived inline in
// the request body is larger than the modality's cap allows.
//
// The bound is on the length of the base64 string, checked before any decode,
// so an oversized payload is never allocated. It bounds the allocation to
// within 2 bytes of the cap: when the cap is not a multiple of 3, a string
// right at the bound decodes to up to 2 bytes over.
func (s *ReplaceMediaURLsStep) inlineSizeExceeded(b64, modality string) bool {
	return int64(len(b64)) > base64LenForBytes(s.downloadSizeFor(modality))
}

// enforceInlineSize reports whether the per-modality cap is applied to a data
// URI in a URL slot, whose bytes arrive in the request body instead of over
// the network.
//
// Audio and video are enforced so the per-modality cap holds however the
// payload arrived. coordinator.yaml's max_download_size comment records the
// memory bound these caps set. input_audio is only ever inline and is capped by
// validateInlineAudio, so enforcing here keeps the two ways of sending the same
// audio in agreement.
//
// Images are exempt, on the same grounds as their exemption from the
// download-path Content-Type check in enforceDownloadContentType.
// server.max_request_body_size is what bounds an image data URI.
func (s *ReplaceMediaURLsStep) enforceInlineSize(modality string) bool {
	return modality != ModalityImage
}

// base64LenForBytes returns the padded-base64 encoded length of a sizeCap-byte
// payload, which is 4*ceil(sizeCap/3) chars.
//
// The multiply saturates at MaxInt64 instead of wrapping. A per-modality cap is
// validated only against MaxInt/BytesPerMB, so sizeCap can legitimately reach
// ~9.2e18 bytes, and 4*ceil(sizeCap/3) overflows int64 above ~6.9e18, turning
// the bound negative and rejecting every input_audio. Saturated, the bound sits
// above any request body a server will accept, so a cap that large stops
// binding.
func base64LenForBytes(sizeCap int64) int64 {
	if sizeCap > math.MaxInt64-2 {
		return math.MaxInt64
	}
	groups := (sizeCap + 2) / 3
	if groups > math.MaxInt64/4 {
		return math.MaxInt64
	}
	return 4 * groups
}

func appendMultimodalEntry(reqCtx *pipeline.RequestContext, modality string) {
	reqCtx.MultimodalEntries = append(reqCtx.MultimodalEntries, pipeline.MultimodalEntry{
		Modality: modality,
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

	// Normalize before falling back to defaultContentType, so a header that
	// is only parameters ("; charset=utf-8") lands on the default the same
	// way an absent header does.
	contentType := normalizeMediaType(resp.Header.Get("Content-Type"))
	if contentType == "" {
		contentType = defaultContentType
	}

	// Check the origin's Content-Type against the per-modality allowlist
	// before reading the body, so a response that cannot be accepted costs
	// its headers rather than up to sizeCap bytes read and held. Which
	// modalities this covers on the download path is decided by
	// enforceDownloadContentType; the allowlist always applies to data URIs
	// regardless.
	if s.enforceDownloadContentType(modality) && !s.allowedContentTypeForModality(contentType, modality) {
		return nil, "", fmt.Errorf("downloaded content type %q not allowed for %s: %w", contentType, modality, pipeline.ErrBadRequest)
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
// Refs are collected in the order the parts appear in the request. That
// order is part of how entries and parts stay lined up; see
// mediaPartIsWellFormed.
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

// encodeDataURI returns data as "data:<contentType>;base64,<payload>".
//
// It encodes straight into the returned string instead of calling
// base64.StdEncoding.EncodeToString and concatenating a prefix onto the
// result, which would hold two full-size copies of the encoded payload at
// once. strings.Builder hands its buffer to the string without copying, so
// the payload is encoded exactly once. For a 200 MB video the difference is
// ~270 MB of peak per concurrent download.
func encodeDataURI(contentType string, data []byte) string {
	prefix := dataURIPrefix + contentType + ";base64,"
	var sb strings.Builder
	sb.Grow(len(prefix) + base64.StdEncoding.EncodedLen(len(data)))
	sb.WriteString(prefix)
	enc := base64.NewEncoder(base64.StdEncoding, &sb)
	// Neither call can fail: strings.Builder.Write never returns an error, and
	// Close only flushes the final partial base64 block into it.
	_, _ = enc.Write(data)
	_ = enc.Close()
	return sb.String()
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
// modality per the step's configured allowlist. contentType must already be
// normalized by normalizeMediaType, which every caller uses, so the lookup is
// a plain map hit against the allowlist's normalized keys.
//
// A nil value for the modality means the operator opted out of the
// per-modality allowlist (allowed_<modality>_content_types: []), and
// any content type is accepted for that modality.
func (s *ReplaceMediaURLsStep) allowedContentTypeForModality(contentType, modality string) bool {
	allowed, ok := s.allowedContentTypes[modality]
	if !ok {
		return false
	}
	if allowed == nil {
		return true
	}
	_, ok = allowed[contentType]
	return ok
}

// enforceDownloadContentType reports whether the per-modality allowlist is
// applied to the Content-Type an HTTP origin returned. Data URIs are always
// checked; this governs the download path only.
//
// Audio and video are always enforced, for the reason recorded on
// coordinator.yaml's max_audio_download_size.
//
// Images are enforced only when the operator set allowed_image_content_types
// explicitly. coordinator.yaml's comment on that param records why leaving it
// unset leaves image downloads unchecked. An origin that sends no Content-Type
// at all lands on defaultContentType.
func (s *ReplaceMediaURLsStep) enforceDownloadContentType(modality string) bool {
	if modality != ModalityImage {
		return true
	}
	_, explicit := s.contentTypeOverrides[ModalityImage]
	return explicit
}

// downloadSizeFor returns the per-modality cap when the operator set one, else
// the global default. Every byte bound in this step resolves through it;
// coordinator.yaml's max_download_size comment records which payloads each
// modality's cap reaches.
func (s *ReplaceMediaURLsStep) downloadSizeFor(modality string) int64 {
	if v, ok := s.maxDownloadSizeByMod[modality]; ok {
		return v
	}
	return s.maxDownloadSize
}

// modalityParam pairs a modality with the config key that overrides one of
// its defaults.
type modalityParam struct {
	modality string
	param    string
}

// perModalityDownloadSizeParams names the config keys that override
// max_download_size on a per-modality basis. A slice rather than a map, so a
// config with more than one bad value always reports the same one and the
// operator does not chase a different message on each restart.
var perModalityDownloadSizeParams = []modalityParam{
	{ModalityImage, "max_image_download_size"},
	{ModalityAudio, "max_audio_download_size"},
	{ModalityVideo, "max_video_download_size"},
}

// parsePerModalityDownloadSizes reads the three optional per-modality cap
// params. Each uses the same validation as max_download_size (positive,
// bounded by MaxInt/BytesPerMB). Returns nil when no override is set so the
// step can distinguish "no override" from "override set to zero".
func parsePerModalityDownloadSizes(params map[string]any) (map[string]int64, error) {
	var out map[string]int64
	for _, mp := range perModalityDownloadSizeParams {
		v, ok, err := paramInt(params, mp.param)
		if err != nil {
			return nil, err
		}
		if !ok {
			continue
		}
		if v <= 0 || v > (math.MaxInt-1)/config.BytesPerMB {
			return nil, fmt.Errorf("%s must be positive and at most %d MB, got %d", mp.param, (math.MaxInt-1)/config.BytesPerMB, v)
		}
		if out == nil {
			out = make(map[string]int64, len(perModalityDownloadSizeParams))
		}
		out[mp.modality] = int64(v) * config.BytesPerMB
	}
	return out, nil
}

// perModalityContentTypeParams names the config keys that override the
// built-in MIME allowlist on a per-modality basis. A slice for the same
// reason as perModalityDownloadSizeParams.
var perModalityContentTypeParams = []modalityParam{
	{ModalityImage, "allowed_image_content_types"},
	{ModalityAudio, "allowed_audio_content_types"},
	{ModalityVideo, "allowed_video_content_types"},
}

// parsePerModalityContentTypes builds the final per-modality allowlist map,
// starting from defaultAllowedContentTypesByModality and replacing any
// modality whose config param is set. Each override is a list of MIME
// strings; non-list roots or non-string entries are rejected so a
// misconfiguration fails loudly instead of silently disabling the check.
//
// A present-but-empty list is used to mean "unrestricted for this
// modality", matching allowed_domains's convention. That maps to a nil
// value in the returned map, which allowedContentTypeForModality treats
// as "accept anything".
//
// A present key with a null value ("allowed_audio_content_types:" and
// nothing after it, which a template renders whenever its variable is
// unset) is rejected. Reading it as an absent key would restore the built-in
// default and, for image, leave enforceDownloadContentType off, so the
// operator would get no download-path check from a line they wrote to add
// one. allowed_domains rejects a null value for the same reason.
//
// The second return value names the modalities that were configured
// explicitly, empty lists included. A default set and an override that
// happens to match it are indistinguishable in the first return value, and
// enforceDownloadContentType needs to tell them apart.
func parsePerModalityContentTypes(params map[string]any) (map[string]map[string]struct{}, map[string]struct{}, error) {
	out := make(map[string]map[string]struct{}, len(defaultAllowedContentTypesByModality))
	for mod, set := range defaultAllowedContentTypesByModality {
		// Clone so a caller mutating the returned per-modality set never
		// leaks into defaultAllowedContentTypesByModality (visible to every
		// future ReplaceMediaURLsStep in the process).
		out[mod] = maps.Clone(set)
	}
	overrides := make(map[string]struct{}, len(perModalityContentTypeParams))
	for _, mp := range perModalityContentTypeParams {
		raw, present := params[mp.param]
		if !present {
			continue
		}
		types, err := parseContentTypeSet(raw, mp.param)
		if err != nil {
			return nil, nil, err
		}
		overrides[mp.modality] = struct{}{}
		if len(types) == 0 {
			// Empty list = accept anything for this modality.
			out[mp.modality] = nil
			continue
		}
		out[mp.modality] = types
	}
	return out, overrides, nil
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
		// Same normalization the checked types get, so an entry written with
		// a parameter attached still matches the bare type it names.
		set[normalizeMediaType(mime)] = struct{}{}
	}
	return set, nil
}

// audioFormatMIME maps OpenAI's input_audio.format values to canonical MIME
// types, one per format, for the allowlist check validateInlineAudio runs.
// "wav" and "mp3" match OpenAI's chat-completions API; the other entries cover
// formats backends commonly accept. coordinator.yaml's
// allowed_audio_content_types comment records what one MIME per format means
// for an operator narrowing that list.
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
	rest := uri
	if isDataURI(rest) {
		rest = rest[len(dataURIPrefix):]
	}
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
	contentType = normalizeMediaType(ct)
	if contentType == "" {
		return "", "", errors.New("data URI missing media type")
	}
	return contentType, payload, nil
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
