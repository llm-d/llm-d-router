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

	"github.com/llm-d/coordinator/pkg/config"
	"github.com/llm-d/coordinator/pkg/gateway"
	"github.com/llm-d/coordinator/pkg/pipeline"
	"golang.org/x/sync/errgroup"
)

const ReplaceMediaURLsStepName = "replace-media-urls"

const imageURLPartType = "image_url"

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
	maxDownloadSize        int64
	guard                  *addressGuard
	client                 *http.Client
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

	guard := &addressGuard{}
	if v, ok := params["allow_private_networks"].(bool); ok {
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

	var imageURLs []imageRef
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
			if partMap["type"] != imageURLPartType {
				continue
			}
			imageURL, ok := partMap[imageURLPartType].(map[string]any)
			if !ok {
				continue
			}
			url, ok := imageURL["url"].(string)
			if !ok {
				continue
			}
			imageURLs = append(imageURLs, imageRef{
				msgIdx:   msgIdx,
				partIdx:  partIdx,
				url:      url,
				imageURL: imageURL,
			})
		}
	}

	if len(imageURLs) == 0 {
		return nil
	}

	if s.maxMultimodalEntries > 0 && len(imageURLs) > s.maxMultimodalEntries {
		return fmt.Errorf("too many multimodal entries: got %d, max %d: %w", len(imageURLs), s.maxMultimodalEntries, pipeline.ErrBadRequest)
	}

	// Cancel any in-flight downloads when Execute returns early (cancelled
	// context or a rejected data URI), so goroutines do not outlive the step.
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	g, gCtx := errgroup.WithContext(ctx)
	g.SetLimit(s.maxConcurrentDownloads)

	results := make([]downloadResult, len(imageURLs))
	for i, ref := range imageURLs {
		if err := gCtx.Err(); err != nil {
			break
		}
		if strings.HasPrefix(ref.url, "data:") {
			contentType, b64, err := parseDataURI(ref.url)
			if err != nil {
				return fmt.Errorf("parsing data URI at message %d part %d: %w: %w", ref.msgIdx, ref.partIdx, err, pipeline.ErrBadRequest)
			}
			if !allowedImageContentType(contentType) {
				return fmt.Errorf("data URI content type %q not allowed at message %d part %d: %w", contentType, ref.msgIdx, ref.partIdx, pipeline.ErrBadRequest)
			}
			results[i] = downloadResult{ref: ref, base64Data: b64, contentType: contentType}
			continue
		}
		g.Go(func() error {
			data, contentType, err := s.download(gCtx, ref.url)
			if err != nil {
				return fmt.Errorf("downloading %s: %w", ref.url, err)
			}
			results[i] = downloadResult{
				ref:         ref,
				base64Data:  base64.StdEncoding.EncodeToString(data),
				contentType: contentType,
			}
			return nil
		})
	}

	// Log proxy presence only: HTTP(S)_PROXY URLs can carry basic-auth
	// credentials (http://user:pass@host) that must not reach logs.
	logger.V(logutil.TRACE).Info("downloading images", "count", len(imageURLs), "http_proxy_set", os.Getenv("HTTP_PROXY") != "", "https_proxy_set", os.Getenv("HTTPS_PROXY") != "")

	if err := g.Wait(); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	for _, r := range results {
		if !strings.HasPrefix(r.ref.url, "data:") {
			r.ref.imageURL["url"] = fmt.Sprintf("data:%s;base64,%s", r.contentType, r.base64Data)
		}

		appendMultimodalEntry(reqCtx, r.contentType, r.base64Data)
	}

	return nil
}

func appendMultimodalEntry(reqCtx *pipeline.RequestContext, contentType, b64 string) {
	reqCtx.MultimodalEntries = append(reqCtx.MultimodalEntries, pipeline.MultimodalEntry{
		Index:       len(reqCtx.MultimodalEntries),
		Base64Data:  b64,
		ContentType: contentType,
	})
}

func (s *ReplaceMediaURLsStep) download(ctx context.Context, rawURL string) ([]byte, string, error) {
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

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, "", err
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return nil, "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, "", fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	if resp.ContentLength > s.maxDownloadSize {
		return nil, "", fmt.Errorf("response too large: Content-Length %d exceeds max %d: %w", resp.ContentLength, s.maxDownloadSize, pipeline.ErrBadRequest)
	}

	data, err := io.ReadAll(io.LimitReader(resp.Body, s.maxDownloadSize+1))
	if err != nil {
		return nil, "", err
	}
	if int64(len(data)) > s.maxDownloadSize {
		return nil, "", fmt.Errorf("response too large: body exceeds max %d: %w", s.maxDownloadSize, pipeline.ErrBadRequest)
	}
	contentType := resp.Header.Get("Content-Type")
	if contentType == "" {
		contentType = defaultContentType
	}
	return data, contentType, nil
}

type imageRef struct {
	msgIdx   int
	partIdx  int
	url      string
	imageURL map[string]any
}

type downloadResult struct {
	ref         imageRef
	base64Data  string
	contentType string
}

// allowedImageContentTypes is the set of data URI media types a vision model
// accepts. Non-image payloads (HTML, SVG, scripts, audio, video) are rejected
// so they are not inlined and forwarded downstream.
var allowedImageContentTypes = map[string]struct{}{
	"image/jpeg": {},
	"image/png":  {},
	"image/gif":  {},
	"image/webp": {},
}

func allowedImageContentType(contentType string) bool {
	_, ok := allowedImageContentTypes[strings.ToLower(strings.TrimSpace(contentType))]
	return ok
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
