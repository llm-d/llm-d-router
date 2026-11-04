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
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"

	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
	"golang.org/x/sync/errgroup"
)

const ReplaceMediaURLsStepName = "replace-media-urls"

const defaultContentType = "application/octet-stream"

// normalizeMediaType reduces a Content-Type header or data URI media type to
// the bare lowercase type, dropping everything from the first ";" (MIME
// parameters) or ",". Cutting at the comma keeps the emitted data URI
// parseable, since RFC 2397 ends the metadata there.
//
// Returns "" for an empty value or one that is only parameters; callers needing
// a type substitute defaultContentType.
func normalizeMediaType(raw string) string {
	if i := strings.IndexAny(raw, ";,"); i >= 0 {
		raw = raw[:i]
	}
	return strings.ToLower(strings.TrimSpace(raw))
}

const dataURIPrefix = "data:"

// isDataURI reports whether s carries the data: scheme, ignoring case: scheme
// names are case-insensitive (RFC 3986 section 3.1), so a case-sensitive check
// would send "DATA:image/png;base64,..." down the download path. The payload
// keeps the client's casing, which vLLM's urlparse lowercases as url.Parse does.
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
	// maxDownloadSize is the fallback cap, reached through downloadSizeFor.
	maxDownloadSize int64
	// maxDownloadSizeByMod optionally overrides maxDownloadSize per modality.
	maxDownloadSizeByMod map[reqcommon.Modality]int64
	// allowedContentTypes is the per-modality MIME allowlist (lowercase)
	// applied to data URIs and input_audio items.
	allowedContentTypes map[reqcommon.Modality]map[string]struct{}
	// contentTypeOverrides names the modalities with an explicit
	// allowed_<modality>_content_types param, telling one from the built-in
	// default. Only image reads it; see enforceDownloadContentType.
	contentTypeOverrides map[reqcommon.Modality]struct{}
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

	// Optional per-modality download caps. Same units (MB) and overflow guard
	// as max_download_size.
	maxDownloadSizeByMod, err := parsePerModalityDownloadSizes(params)
	if err != nil {
		return nil, err
	}

	// Optional per-modality MIME allowlist overrides; overrides names the ones
	// set explicitly.
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

	// One slice, tagged by modality, in request order: that order is what pairs
	// entries with parts later (see collectMediaParts). Separate passes over URL
	// and inline parts would reorder a request mixing audio_url and input_audio.
	var refs []mediaRef
	apiType := reqcommon.DetectAPIType(reqCtx.OriginalPath)
	if items, ok := promptItems(reqCtx.Body, apiType); ok {
		var err error
		refs, err = collectMediaRefs(items, apiType)
		if err != nil {
			return err
		}
		// The refs alias parts inside items, which promptItems may have decoded
		// out of a json.RawMessage; store the slice back so the rewrites below
		// land in the body. Skipped when nothing will be rewritten, which leaves
		// the field raw and its key order intact.
		if len(refs) > 0 {
			setPromptItems(reqCtx.Body, apiType, items)
		}
	}

	// boundMediaType folds audio and video into MediaTypeOther, so count the
	// two buckets separately. Still above the return below, so a request
	// carrying no media observes zero as it did.
	var imageItems, otherItems int
	for _, ref := range refs {
		if ref.modality == reqcommon.ModalityImage {
			imageItems++
			continue
		}
		otherItems++
	}
	coordmetrics.RecordMediaItems(coordmetrics.MediaTypeImage, imageItems)
	coordmetrics.RecordMediaItems(coordmetrics.MediaTypeOther, otherItems)

	if len(refs) == 0 {
		return nil
	}

	if s.maxMultimodalEntries > 0 && len(refs) > s.maxMultimodalEntries {
		return fmt.Errorf("too many multimodal entries: got %d, max %d: %w", len(refs), s.maxMultimodalEntries, pipeline.ErrBadRequest)
	}

	// Every inline input_audio ref is validated before any download starts:
	// these checks are local, so a request that will be rejected anyway does not
	// first pull megabytes for its audio_url / video_url refs. Checking inside
	// the download loop would let a late inline ref land after earlier downloads
	// had already started.
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

	// dataURIs parallels refs: dataURIs[i] is the rewritten "url" for refs[i],
	// built by that ref's download goroutine. Inline refs and refs already
	// carrying a data URI leave their slot empty.
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
			// Validate only: the slot is already the payload.
			contentType, b64, err := parseDataURI(ref.url)
			if err != nil {
				return fmt.Errorf("parsing data URI at %s: %w: %w", ref.location, err, pipeline.ErrBadRequest)
			}
			if !s.allowedContentTypeForModality(contentType, ref.modality) {
				return fmt.Errorf("data URI content type %q not allowed for %s at %s: %w", contentType, ref.modality, ref.location, pipeline.ErrBadRequest)
			}
			if s.enforceInlineSize(ref.modality) && s.inlineSizeExceeded(b64, ref.modality) {
				return fmt.Errorf("data URI at %s exceeds size limit for %s: %w",
					ref.location, ref.modality, pipeline.ErrBadRequest)
			}
			continue
		}
		downloadCount++
		g.Go(func() error {
			data, contentType, err := s.download(gCtx, ref.url, ref.modality)
			if err != nil {
				return fmt.Errorf("downloading %s at %s: %w", ref.url, ref.location, err)
			}
			// Encode the final URI here rather than stashing base64 for the
			// walker pass to wrap: holding both the payload and the URI
			// containing it doubles what the step retains per request.
			dataURIs[i] = encodeDataURI(contentType, data)
			return nil
		})
	}

	// Log proxy presence only: HTTP(S)_PROXY URLs can carry basic-auth
	// credentials (http://user:pass@host) that must not reach logs.
	// count covers only refs that go over the network, so it can be read
	// against egress volume.
	logger.V(logutil.TRACE).Info("downloading media", "count", downloadCount, "http_proxy_set", os.Getenv("HTTP_PROXY") != "", "https_proxy_set", os.Getenv("HTTPS_PROXY") != "")

	if err := g.Wait(); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	// Walk refs in request order: rewrite URL slots in place and append one
	// entry per ref. That order is the point; see collectMediaParts.
	for i, ref := range refs {
		// A non-empty slot is exactly a downloaded ref; every other ref
		// already has its payload in the body.
		if dataURIs[i] != "" {
			ref.setURL(dataURIs[i])
		}
		appendMultimodalEntry(reqCtx, ref.modality)
	}

	return nil
}

// collectMediaRefs returns a ref per media content part, in walk order.
//
// A part that carries no usable media is rejected rather than skipped: encode
// and decode pair this same walk with reqCtx.MultimodalEntries by position
// within a modality, so skipping one would shift every later part of that
// modality onto another part's hash, priming the encoder from a part it cannot
// fetch under a hash the prefiller then misses.
//
// Usable differs by part type: a URL-based part needs a readable URL, an
// input_audio part its inline base64 data (format, MIME and size come later, in
// validateInlineAudio).
func collectMediaRefs(items []any, apiType reqcommon.APIType) ([]mediaRef, error) {
	parts := collectMediaParts(items, apiType)
	refs := make([]mediaRef, 0, len(parts))
	for _, media := range parts {
		partType, _ := media.part[reqcommon.FieldType].(string)
		if partType == reqcommon.PartTypeInputAudio {
			ref, err := inlineAudioRef(media)
			if err != nil {
				return nil, err
			}
			refs = append(refs, ref)
			continue
		}
		url, setURL := reqcommon.MediaPartURLRef(media.part)
		if setURL == nil || url == "" {
			return nil, fmt.Errorf("%s: %s part carries no fetchable URL: %w",
				media.location, media.modality, pipeline.ErrBadRequest)
		}
		refs = append(refs, mediaRef{
			location: media.location,
			modality: media.modality,
			url:      url,
			setURL:   setURL,
		})
	}
	return refs, nil
}

// inlineAudioRef builds the ref for an input_audio part, whose payload sits in
// the request body rather than behind a URL. Both fields are read here so
// Execute can check them before it starts any download.
func inlineAudioRef(media mediaPart) (mediaRef, error) {
	data, format, ok := inlineAudioData(media.part)
	if !ok {
		return mediaRef{}, fmt.Errorf("%s: input_audio part carries no base64 data: %w",
			media.location, pipeline.ErrBadRequest)
	}
	return mediaRef{
		location: media.location,
		modality: media.modality,
		isInline: true,
		data:     data,
		format:   format,
	}, nil
}

// validateInlineAudio checks one input_audio ref: a recognized format name, a
// MIME that passes the audio allowlist, and a payload within the audio cap.
func (s *ReplaceMediaURLsStep) validateInlineAudio(ref mediaRef) error {
	contentType, err := audioFormatToMIME(ref.format)
	if err != nil {
		return fmt.Errorf("input_audio at %s: %w: %w", ref.location, err, pipeline.ErrBadRequest)
	}
	if !s.allowedContentTypeForModality(contentType, ref.modality) {
		// Name the format alongside its MIME: the allowlist is written in
		// MIMEs and the request in formats, so the MIME alone does not say
		// which format was refused.
		return fmt.Errorf("input_audio format %q (content type %q) not allowed at %s: %w",
			ref.format, contentType, ref.location, pipeline.ErrBadRequest)
	}
	if s.inlineSizeExceeded(ref.data, ref.modality) {
		return fmt.Errorf("input_audio at %s exceeds size limit: %w", ref.location, pipeline.ErrBadRequest)
	}
	return nil
}

// inlineSizeExceeded reports whether a base64 payload that arrived inline in
// the request body exceeds the modality's cap. The payload is measured, never
// decoded, so an oversized one is rejected without being allocated.
//
// Measuring the payload rather than encoding the cap keeps this exact at the
// boundary -- a cap that is not a multiple of 3 used to leave up to 2 bytes
// slack -- and keeps the arithmetic on an int bounded by
// server.max_request_body_size, so nothing can overflow.
func (s *ReplaceMediaURLsStep) inlineSizeExceeded(b64 string, modality reqcommon.Modality) bool {
	return int64(reqcommon.Base64DecodedLen(b64)) > s.downloadSizeFor(modality)
}

// enforceInlineSize reports whether the per-modality cap applies to a data URI
// in a URL slot, whose bytes arrive in the request body instead of over the
// network. Audio and video always, so the cap holds however the payload arrived.
// Image only when max_image_download_size is set, the same opt-in
// enforceDownloadContentType makes: falling back to max_download_size would
// start rejecting data URIs every such config has always accepted. Unset leaves
// max_request_body_size as the bound.
func (s *ReplaceMediaURLsStep) enforceInlineSize(modality reqcommon.Modality) bool {
	if modality != reqcommon.ModalityImage {
		return true
	}
	_, explicit := s.maxDownloadSizeByMod[reqcommon.ModalityImage]
	return explicit
}

func appendMultimodalEntry(reqCtx *pipeline.RequestContext, modality reqcommon.Modality) {
	reqCtx.MultimodalEntries = append(reqCtx.MultimodalEntries, pipeline.MultimodalEntry{
		Modality: modality,
	})
}

func (s *ReplaceMediaURLsStep) download(ctx context.Context, rawURL string, modality reqcommon.Modality) (data []byte, contentType string, err error) {
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return nil, "", fmt.Errorf("invalid URL: %w: %w", err, pipeline.ErrBadRequest)
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return nil, "", fmt.Errorf("scheme %q not allowed: %w", parsed.Scheme, pipeline.ErrBadRequest)
	}
	if !s.guard.hostAllowed(parsed.Hostname()) {
		log.FromContext(ctx).WithName(ReplaceMediaURLsStepName).V(logutil.DEBUG).Info(
			"rejecting media URL: host not in allowed_domains", "host", parsed.Hostname())
		return nil, "", fmt.Errorf("host %q not allowed: %w", parsed.Hostname(), pipeline.ErrBadRequest)
	}

	sizeCap := s.downloadSizeFor(modality)

	// Timing starts after pre-dial validation: URL parse, scheme, and
	// allowed_domains rejections are not download attempts, so they must not
	// drag the error bucket's duration distribution toward zero.
	start := time.Now()
	defer func() {
		coordmetrics.RecordMediaDownloadDuration(coordmetrics.ClassifyDownloadResult(ctx, err), time.Since(start))
	}()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, "", err
	}
	call := coordmetrics.StartUpstreamCall(coordmetrics.UpstreamReplaceMediaURLs)
	// rawURL's host is checked against allowed_domains above, and s.client's
	// dialer (addressGuard.dialControl) blocks the resolved IP if it is
	// loopback, link-local, CGNAT, or private, closing the DNS-rebinding gap
	// a hostname check alone would miss.
	// codeql[go/request-forgery]
	resp, err := s.client.Do(req)
	call.Done()
	if err != nil {
		return nil, "", err
	}
	defer resp.Body.Close()

	if err := checkStatus(ReplaceMediaURLsStepName, resp); err != nil {
		return nil, "", err
	}

	// Normalize before the fallback, so a header that is only parameters
	// ("; charset=utf-8") lands on the default like an absent one does.
	contentType = normalizeMediaType(resp.Header.Get("Content-Type"))
	if contentType == "" {
		contentType = defaultContentType
	}

	// Checked before the body is read, so a response that cannot be accepted
	// costs its headers rather than up to sizeCap bytes read and held.
	if s.enforceDownloadContentType(modality) && !s.allowedDownloadContentType(contentType, modality) {
		return nil, "", fmt.Errorf("downloaded content type %q not allowed for %s: %w", contentType, modality, pipeline.ErrBadRequest)
	}

	if resp.ContentLength > sizeCap {
		return nil, "", fmt.Errorf("response too large: Content-Length %d exceeds max %d: %w", resp.ContentLength, sizeCap, pipeline.ErrBadRequest)
	}

	data, err = io.ReadAll(io.LimitReader(resp.Body, sizeCap+1))
	if err != nil {
		return nil, "", err
	}
	if int64(len(data)) > sizeCap {
		return nil, "", fmt.Errorf("response too large: body exceeds max %d: %w", sizeCap, pipeline.ErrBadRequest)
	}
	return data, contentType, nil
}

// mediaRef locates one media content part in the request body. isInline
// discriminates the two variants: a URL-based part (image_url / audio_url /
// video_url / input_image) fills url + setURL, so the download result can be
// inlined in place; an input_audio part fills data + format, its payload already
// in the body. Refs are collected in request order, part of how entries and
// parts stay lined up; see collectMediaParts.
type mediaRef struct {
	// location names where this ref's part sits in the client body, for error
	// messages: "message 0 content part 2", "input item 1 output part 0".
	location string
	modality reqcommon.Modality
	isInline bool
	// URL variant:
	url string
	// setURL writes the rewritten data URI back to wherever this ref's URL
	// lives in reqCtx.Body, since that location's shape differs by API
	// format (chat-completions nests it at image_url.url; Responses stores
	// it as a bare string field on the part itself).
	setURL func(string)
	// Inline variant:
	data   string // base64 payload
	format string // "wav", "mp3", ...
}

// encodeDataURI returns data as "data:<contentType>;base64,<payload>". It
// encodes straight into the returned string rather than concatenating a prefix
// onto base64.StdEncoding.EncodeToString's result, which would hold two
// full-size copies at once: strings.Builder hands its buffer over without
// copying, saving ~270 MB of peak per concurrent 200 MB download.
func encodeDataURI(contentType string, data []byte) string {
	prefix := dataURIPrefix + contentType + ";base64,"
	var sb strings.Builder
	sb.Grow(len(prefix) + base64.StdEncoding.EncodedLen(len(data)))
	sb.WriteString(prefix)
	enc := base64.NewEncoder(base64.StdEncoding, &sb)
	// Neither can fail: Builder.Write never errors and Close only flushes the
	// final partial base64 block into it.
	_, _ = enc.Write(data)
	_ = enc.Close()
	return sb.String()
}

// defaultAllowedContentTypesByModality is the built-in per-modality MIME
// allowlist, used for any modality the allowed_{image,audio,video}_content_types
// params do not override. Intentionally permissive: codec-level restrictions
// are the backend's job.
var defaultAllowedContentTypesByModality = map[reqcommon.Modality]map[string]struct{}{
	reqcommon.ModalityImage: {
		"image/jpeg": {},
		"image/png":  {},
		"image/gif":  {},
		"image/webp": {},
	},
	reqcommon.ModalityAudio: {
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
	reqcommon.ModalityVideo: {
		"video/mp4":       {},
		"video/webm":      {},
		"video/quicktime": {},
		"video/mpeg":      {},
		"video/ogg":       {},
	},
}

// allowedContentTypeForModality reports whether contentType is allowed for
// modality under the step's allowlist. The lookup is a plain map hit, so
// contentType must already be normalizeMediaType's bare lowercase shape; a
// parameter or mixed case would miss every entry. A nil value means the
// operator opted out (allowed_<modality>_content_types: []).
func (s *ReplaceMediaURLsStep) allowedContentTypeForModality(contentType string, modality reqcommon.Modality) bool {
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

// allowedDownloadContentType is allowedContentTypeForModality plus
// defaultContentType, for the built-in list only.
//
// An unlabeled origin and a lying one both send application/octet-stream, so
// refusing it buys nothing against the liar (who can send audio/wav instead) and
// costs every S3, GCS or presigned URL that serves its objects unlabeled. What
// the check still catches is an origin that honestly returns something else: an
// HTML error page, a login redirect, a JSON error body.
//
// A data URI gets no such allowance (the client wrote the type itself), and
// neither does an explicit allowed_<modality>_content_types -- coordinator.yaml
// tells an operator to list application/octet-stream when they want unlabeled
// origins.
func (s *ReplaceMediaURLsStep) allowedDownloadContentType(contentType string, modality reqcommon.Modality) bool {
	if _, explicit := s.contentTypeOverrides[modality]; !explicit && contentType == defaultContentType {
		return true
	}
	return s.allowedContentTypeForModality(contentType, modality)
}

// enforceDownloadContentType reports whether the allowlist is applied to the
// Content-Type an HTTP origin returned; data URIs are always checked, so this
// governs the download path only. Audio and video always; images only when
// allowed_image_content_types is set explicitly. coordinator.yaml's comments on
// those two params record why they differ.
func (s *ReplaceMediaURLsStep) enforceDownloadContentType(modality reqcommon.Modality) bool {
	if modality != reqcommon.ModalityImage {
		return true
	}
	_, explicit := s.contentTypeOverrides[reqcommon.ModalityImage]
	return explicit
}

// downloadSizeFor returns the per-modality cap when the operator set one, else
// the global default. Every byte bound in this step resolves through it; which
// payloads each cap reaches is recorded on coordinator.yaml's max_download_size.
func (s *ReplaceMediaURLsStep) downloadSizeFor(modality reqcommon.Modality) int64 {
	if v, ok := s.maxDownloadSizeByMod[modality]; ok {
		return v
	}
	return s.maxDownloadSize
}

// modalityParam pairs a modality with a config key that overrides one of its
// defaults.
type modalityParam struct {
	modality reqcommon.Modality
	param    string
}

// perModalityDownloadSizeParams names the config keys overriding
// max_download_size per modality. A slice, not a map, so a config with more
// than one bad value reports the same one on every restart.
var perModalityDownloadSizeParams = []modalityParam{
	{reqcommon.ModalityImage, "max_image_download_size"},
	{reqcommon.ModalityAudio, "max_audio_download_size"},
	{reqcommon.ModalityVideo, "max_video_download_size"},
}

// parsePerModalityDownloadSizes reads the three optional cap params, each
// validated like max_download_size (positive, at most MaxInt/BytesPerMB).
// Returns nil when none is set, telling "no override" from "set to zero".
//
// A key present with a null value -- the shape a template renders from an unset
// variable -- is rejected rather than read as absent, which paramInt alone would
// do: the fallback is the global max_download_size, so a line written to tighten
// one modality's cap would leave it untightened. Same reasoning as the null
// allowlist value on parsePerModalityContentTypes.
func parsePerModalityDownloadSizes(params map[string]any) (map[reqcommon.Modality]int64, error) {
	var out map[reqcommon.Modality]int64
	for _, mp := range perModalityDownloadSizeParams {
		if raw, present := params[mp.param]; present && raw == nil {
			return nil, fmt.Errorf("%s must be a number, got <nil>", mp.param)
		}
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
			out = make(map[reqcommon.Modality]int64, len(perModalityDownloadSizeParams))
		}
		out[mp.modality] = int64(v) * config.BytesPerMB
	}
	return out, nil
}

// perModalityContentTypeParams names the config keys overriding the built-in
// MIME allowlist per modality. A slice, for the reason on
// perModalityDownloadSizeParams.
var perModalityContentTypeParams = []modalityParam{
	{reqcommon.ModalityImage, "allowed_image_content_types"},
	{reqcommon.ModalityAudio, "allowed_audio_content_types"},
	{reqcommon.ModalityVideo, "allowed_video_content_types"},
}

// parsePerModalityContentTypes builds the final per-modality allowlist from
// defaultAllowedContentTypesByModality, replacing any modality whose config
// param is set. A non-list root or non-string entry is rejected rather than
// silently disabling the check -- including a key present with a null value,
// which read as absent would leave image's enforceDownloadContentType off: no
// download check from a line written to add one. An empty list means
// "unrestricted for this modality", matching allowed_domains, and maps to nil.
//
// The second return value names the modalities set explicitly, empty lists
// included: a default set and an override matching it are indistinguishable in
// the first, and enforceDownloadContentType must tell them apart.
func parsePerModalityContentTypes(params map[string]any) (map[reqcommon.Modality]map[string]struct{}, map[reqcommon.Modality]struct{}, error) {
	out := make(map[reqcommon.Modality]map[string]struct{}, len(defaultAllowedContentTypesByModality))
	for mod, set := range defaultAllowedContentTypesByModality {
		// Clone so a caller mutating a returned set cannot leak into
		// defaultAllowedContentTypesByModality, which every future step sees.
		out[mod] = maps.Clone(set)
	}
	overrides := make(map[reqcommon.Modality]struct{}, len(perModalityContentTypeParams))
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

// parseStringSet accepts list entries as []any (the YAML decode path) or
// []string (programmatic callers) and returns them as a set, each entry run
// through normalize so it is stored in the form the lookup compares against.
// fieldName appears only in error messages.
//
// Any other root type -- including the null a template renders from an unset
// variable -- is an error rather than a silently disabled allowlist, which would
// be an open-by-default downgrade of a security control. An entry that
// normalizes to nothing is rejected for that reason inverted: no real value
// normalizes to "", so it would be a live key nothing can match, and a one-entry
// list would reject everything where the documented empty list accepts it.
func parseStringSet(raw any, fieldName string, normalize func(string) string) (map[string]struct{}, error) {
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
		entry, ok := e.(string)
		if !ok {
			return nil, fmt.Errorf("%s entries must be strings, got %T", fieldName, e)
		}
		normalized := normalize(entry)
		if normalized == "" {
			return nil, fmt.Errorf("%s entry %q normalizes to nothing", fieldName, entry)
		}
		set[normalized] = struct{}{}
	}
	return set, nil
}

// parseContentTypeSet reads one allowed_<modality>_content_types param. Entries
// get the normalization the checked types get, so one written with a parameter
// attached still matches the bare type it names.
func parseContentTypeSet(raw any, fieldName string) (map[string]struct{}, error) {
	return parseStringSet(raw, fieldName, normalizeMediaType)
}

// audioFormatMIME maps OpenAI's input_audio.format values to one canonical MIME
// each, for the allowlist check validateInlineAudio runs. "wav" and "mp3" match
// OpenAI's chat-completions API; the rest cover formats backends commonly accept.
// coordinator.yaml's allowed_audio_content_types comment records what one MIME
// per format means for an operator narrowing that list.
//
// Each value must be a bare lowercase type, since validateInlineAudio hands it
// straight to allowedContentTypeForModality's equality match
// (TestAudioFormatMIMEValuesAreNormalized).
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

// parseAllowedDomains reads the allowed_domains param. Entries are lowercased,
// the form hostAllowed compares against. Case is all that is folded: a hostname
// never legitimately carries surrounding whitespace, and trimming it here would
// widen what this guard matches rather than narrow it.
func parseAllowedDomains(raw any) (map[string]struct{}, error) {
	return parseStringSet(raw, "allowed_domains", strings.ToLower)
}
