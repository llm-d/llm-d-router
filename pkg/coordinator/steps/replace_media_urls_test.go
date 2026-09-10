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
	"io"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// MIME literals reused across tests. Extracted so goconst does not flag
// their repetition and so a rename or typo lands in one place.
const (
	testAudioWAVMIME  = "audio/wav"
	testImagePNGMIME  = "image/png"
	testImageJPEGMIME = "image/jpeg"
	testVideoMP4MIME  = "video/mp4"
)

// newLoopbackStep builds a step whose SSRF guard permits loopback. httptest
// servers bind to 127.0.0.1, which the guard blocks by default, so download
// tests that talk to a local server must opt loopback back in.
func newLoopbackStep(t *testing.T, params map[string]any) *ReplaceMediaURLsStep {
	t.Helper()
	step, err := NewReplaceMediaURLsStep(nil, params)
	if err != nil {
		t.Fatal(err)
	}
	rmu := step.(*ReplaceMediaURLsStep)
	rmu.guard.allowLoopback = true
	return rmu
}

func TestReplaceMediaURLsStep_DownloadsAndInlines(t *testing.T) {
	imageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", testImageJPEGMIME)
		_, _ = w.Write([]byte("jpeg-bytes"))
	}))
	defer imageServer.Close()

	step := newLoopbackStep(t, map[string]any{"download_timeout": "5s"})

	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{"type": "text", "text": "describe this"},
						map[string]any{
							"type":      "image_url",
							"image_url": map[string]any{"url": imageServer.URL + "/photo.jpg"},
						},
					},
				},
			},
		},
	}

	err := step.Execute(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(reqCtx.MultimodalEntries) != 1 {
		t.Fatalf("expected 1 multimodal entry, got %d", len(reqCtx.MultimodalEntries))
	}

	msgs := reqCtx.Body["messages"].([]any)
	content := msgs[0].(map[string]any)["content"].([]any)
	imgPart := content[1].(map[string]any)["image_url"].(map[string]any)
	url := imgPart["url"].(string)
	if url[:len("data:image/jpeg;base64,")] != "data:image/jpeg;base64," {
		t.Fatalf("expected data URI, got %s", url)
	}
}

func TestReplaceMediaURLsStep_NoImages(t *testing.T) {
	step, _ := NewReplaceMediaURLsStep(nil, map[string]any{})

	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{"role": "user", "content": "just text"},
			},
		},
	}

	err := step.Execute(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(reqCtx.MultimodalEntries) != 0 {
		t.Fatalf("expected 0 multimodal entries, got %d", len(reqCtx.MultimodalEntries))
	}
}

func TestReplaceMediaURLsStep_DownloadFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	step := newLoopbackStep(t, map[string]any{})

	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":      "image_url",
							"image_url": map[string]any{"url": server.URL + "/missing.png"},
						},
					},
				},
			},
		},
	}

	err := step.Execute(context.Background(), reqCtx)
	if err == nil {
		t.Fatal("expected error for failed download")
	}
	var upstreamErr *pipeline.UpstreamError
	if !errors.As(err, &upstreamErr) {
		t.Fatalf("expected *pipeline.UpstreamError, got %T: %v", err, err)
	}
	if upstreamErr.StatusCode != http.StatusNotFound {
		t.Errorf("StatusCode = %d, want %d", upstreamErr.StatusCode, http.StatusNotFound)
	}
	if upstreamErr.Step != ReplaceMediaURLsStepName {
		t.Errorf("Step = %q, want %q", upstreamErr.Step, ReplaceMediaURLsStepName)
	}
}

func TestReplaceMediaURLsStep_DataURIInput(t *testing.T) {
	step, _ := NewReplaceMediaURLsStep(nil, map[string]any{})

	const dataURI = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{"type": "text", "text": "describe this"},
						map[string]any{
							"type":      "image_url",
							"image_url": map[string]any{"url": dataURI},
						},
					},
				},
			},
		},
	}

	err := step.Execute(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(reqCtx.MultimodalEntries) != 1 {
		t.Fatalf("expected 1 multimodal entry, got %d", len(reqCtx.MultimodalEntries))
	}

	msgs := reqCtx.Body["messages"].([]any)
	content := msgs[0].(map[string]any)["content"].([]any)
	imgPart := content[1].(map[string]any)["image_url"].(map[string]any)
	if imgPart["url"].(string) != dataURI {
		t.Fatalf("expected url unchanged, got %s", imgPart["url"])
	}
}

// TestReplaceMediaURLsStep_UppercaseDataURIScheme covers an uppercase
// data: scheme, which RFC 3986 allows. It must be recognized as inline
// data and left alone. A case-sensitive prefix check would send it to the
// download path, where the scheme guard rejects it for not being http(s).
//
// The url is asserted unchanged, casing included: the step does not rewrite an
// inline payload, and vLLM lowercases the scheme through urlparse before
// matching it, so the client's casing reaches the backend and still parses.
func TestReplaceMediaURLsStep_UppercaseDataURIScheme(t *testing.T) {
	step, _ := NewReplaceMediaURLsStep(nil, map[string]any{})

	const dataURI = "DATA:image/png;base64,iVBORw0KGgo="
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":      "image_url",
							"image_url": map[string]any{"url": dataURI},
						},
					},
				},
			},
		},
	}

	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("expected uppercase data: scheme accepted, got %v", err)
	}
	if len(reqCtx.MultimodalEntries) != 1 {
		t.Fatalf("expected 1 multimodal entry, got %d", len(reqCtx.MultimodalEntries))
	}
	msgs := reqCtx.Body["messages"].([]any)
	content := msgs[0].(map[string]any)["content"].([]any)
	imgPart := content[0].(map[string]any)["image_url"].(map[string]any)
	if imgPart["url"].(string) != dataURI {
		t.Fatalf("expected url unchanged, got %s", imgPart["url"])
	}
}

// One MultimodalEntry must be appended per media part, in request order,
// regardless of whether the part came from a download or an inline data: URI.
// The encode fanout and decode.injectUUIDs pair the Nth entry of a modality
// with the Nth part of that modality (see mediaPartIsWellFormed), so drift
// here attaches the wrong bytes to the wrong entry. Asserted in both source
// orderings.
func TestReplaceMediaURLsStep_MixedHTTPAndDataURIOrdering(t *testing.T) {
	imageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", testImagePNGMIME)
		_, _ = w.Write([]byte("downloaded-image-bytes"))
	}))
	defer imageServer.Close()

	const dataURI = "data:image/jpeg;base64,SU5MSU5F"
	httpURL := imageServer.URL + "/img.png"

	httpPart := map[string]any{"type": "image_url", "image_url": map[string]any{"url": httpURL}}
	dataPart := map[string]any{"type": "image_url", "image_url": map[string]any{"url": dataURI}}

	httpAsDataURI := "data:image/png;base64," + base64.StdEncoding.EncodeToString([]byte("downloaded-image-bytes"))
	tests := []struct {
		name     string
		parts    []any
		wantURLs []string
	}{
		{
			name:     "http then data",
			parts:    []any{httpPart, dataPart},
			wantURLs: []string{httpAsDataURI, dataURI},
		},
		{
			name:     "data then http",
			parts:    []any{dataPart, httpPart},
			wantURLs: []string{dataURI, httpAsDataURI},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			step := newLoopbackStep(t, map[string]any{})
			reqCtx := &pipeline.RequestContext{
				Body: map[string]any{
					"messages": []any{
						map[string]any{"role": "user", "content": tt.parts},
					},
				},
			}

			if err := step.Execute(context.Background(), reqCtx); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(reqCtx.MultimodalEntries) != len(tt.wantURLs) {
				t.Fatalf("expected %d multimodal entries, got %d", len(tt.wantURLs), len(reqCtx.MultimodalEntries))
			}
			for i, want := range tt.wantURLs {
				if got := reqCtx.MultimodalEntries[i].Modality; got != ModalityImage {
					t.Errorf("entry[%d].Modality = %q, want %q", i, got, ModalityImage)
				}
				content := reqCtx.Body["messages"].([]any)[0].(map[string]any)["content"].([]any)
				gotURL := content[i].(map[string]any)["image_url"].(map[string]any)["url"].(string)
				if gotURL != want {
					t.Errorf("content[%d] url = %q, want %q", i, gotURL, want)
				}
			}
		})
	}
}

// Execute and mediaPartIsWellFormed must agree on exactly which parts count.
// Execute fixes the entry order that the encode fanout and decode.injectUUIDs
// later index into, and those two use the predicate, so a part accepted by one
// and rejected by the other shifts the pairing. Both delegate to
// classifyMediaPart; this asserts the agreement end to end rather than trusting
// the delegation. Every malformed shape and every recognized part type is
// represented.
func TestReplaceMediaURLsStep_ExecuteAgreesWithWellFormedPredicate(t *testing.T) {
	parts := []any{
		map[string]any{"type": "text", "text": "hi"},
		// image_url: one good, then every way to be malformed.
		map[string]any{"type": imageURLPartType, imageURLPartType: map[string]any{"url": "data:image/png;base64,aGk="}},
		map[string]any{"type": imageURLPartType},
		map[string]any{"type": imageURLPartType, imageURLPartType: map[string]any{}},
		map[string]any{"type": imageURLPartType, imageURLPartType: map[string]any{"url": 123}},
		// audio_url, including a non-object inner.
		map[string]any{"type": audioURLPartType, audioURLPartType: map[string]any{"url": "data:audio/wav;base64,aGk="}},
		map[string]any{"type": audioURLPartType, audioURLPartType: "not-an-object"},
		// input_audio is inline and carries its payload under data rather
		// than url.
		map[string]any{"type": inputAudioPartType, inputAudioPartType: map[string]any{"data": "aGk=", "format": "wav"}},
		map[string]any{"type": inputAudioPartType, inputAudioPartType: map[string]any{"data": "", "format": "wav"}},
		map[string]any{"type": inputAudioPartType, inputAudioPartType: map[string]any{"format": "wav"}},
		map[string]any{"type": videoURLPartType, videoURLPartType: map[string]any{"url": "data:video/mp4;base64,aGk="}},
		// Recognized by neither: passed through untouched.
		map[string]any{"type": "image_embeds", "image_embeds": map[string]any{}},
	}

	// What the predicate says the answer is, walking the same body Execute does.
	var wantModalities []string
	for _, part := range parts {
		partMap := part.(map[string]any)
		partType, _ := partMap["type"].(string)
		modality, isMedia := partTypeModality[partType]
		if isMedia && mediaPartIsWellFormed(partMap, partType) {
			wantModalities = append(wantModalities, modality)
		}
	}
	// Guard the guard: a typo that made every part malformed would otherwise
	// let this test pass with both sides at zero.
	if len(wantModalities) != 4 {
		t.Fatalf("fixture drift: predicate accepted %d parts, want 4 (%v)", len(wantModalities), wantModalities)
	}

	step, _ := NewReplaceMediaURLsStep(nil, map[string]any{})
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{map[string]any{"role": "user", "content": parts}},
		},
	}
	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	gotModalities := make([]string, 0, len(reqCtx.MultimodalEntries))
	for _, e := range reqCtx.MultimodalEntries {
		gotModalities = append(gotModalities, e.Modality)
	}
	if !equalStringSlices(gotModalities, wantModalities) {
		t.Errorf("Execute produced modalities %v, predicate expects %v", gotModalities, wantModalities)
	}
}

// base64LenForBytes must agree with the real encoder for ordinary caps and must
// stay positive for every cap the config validation permits. The naive
// 4*((cap+2)/3) overflows int64 for a large max_audio_download_size, and a
// negative bound would reject every input_audio instead of accepting more.
func TestBase64LenForBytes(t *testing.T) {
	// Agreement with the encoder, including the non-multiple-of-3 sizes where
	// padding decides the answer.
	for _, n := range []int{0, 1, 2, 3, 4, 5, 6, 100, 1024, 1024 * 1024} {
		want := int64(len(base64.StdEncoding.EncodeToString(make([]byte, n))))
		if got := base64LenForBytes(int64(n)); got != want {
			t.Errorf("base64LenForBytes(%d) = %d, want %d", n, got, want)
		}
	}

	// Never negative, never zero, for anything reachable through config.
	maxConfigurable := int64((math.MaxInt-1)/config.BytesPerMB) * config.BytesPerMB
	for _, cap64 := range []int64{
		maxConfigurable,
		math.MaxInt64 / 4,
		math.MaxInt64 / 2,
		math.MaxInt64 - 2,
		math.MaxInt64,
	} {
		if got := base64LenForBytes(cap64); got <= 0 {
			t.Errorf("base64LenForBytes(%d) = %d, want a positive bound", cap64, got)
		}
	}
}

// A cap so large that the encoded-length bound saturates must still accept a
// normal payload. Before the saturating multiply this rejected everything.
func TestReplaceMediaURLsStep_InputAudio_HugeCapStillAccepts(t *testing.T) {
	hugeMB := (math.MaxInt - 1) / config.BytesPerMB
	step, err := NewReplaceMediaURLsStep(nil, map[string]any{"max_audio_download_size": hugeMB})
	if err != nil {
		t.Fatalf("expected the maximum permitted cap to be accepted: %v", err)
	}
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":             inputAudioPartType,
							inputAudioPartType: map[string]any{"data": "aGk=", "format": "wav"},
						},
					},
				},
			},
		},
	}
	if err := step.(*ReplaceMediaURLsStep).Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("expected payload accepted under a saturating cap, got %v", err)
	}
	if got := len(reqCtx.MultimodalEntries); got != 1 {
		t.Fatalf("expected 1 entry, got %d", got)
	}
}

// encodeDataURI streams into a strings.Builder instead of encoding to a string
// and concatenating, so it must still agree with the obvious implementation
// byte for byte. Payload lengths 0-4 cover every base64 padding case, which is
// where a missed Close() flush would show up.
func TestEncodeDataURI_MatchesNaiveEncoding(t *testing.T) {
	for n := 0; n <= 4; n++ {
		data := make([]byte, n)
		for i := range data {
			data[i] = byte('a' + i)
		}
		got := encodeDataURI(testImagePNGMIME, data)
		want := "data:" + testImagePNGMIME + ";base64," + base64.StdEncoding.EncodeToString(data)
		if got != want {
			t.Errorf("encodeDataURI(%d bytes) = %q, want %q", n, got, want)
		}
		// The emitted URI must survive the parser the step uses on input.
		ct, payload, err := parseDataURI(got)
		if err != nil {
			t.Fatalf("parseDataURI(%q): %v", got, err)
		}
		if ct != testImagePNGMIME {
			t.Errorf("round-tripped content type = %q, want %q", ct, testImagePNGMIME)
		}
		decoded, err := base64.StdEncoding.DecodeString(payload)
		if err != nil {
			t.Fatalf("decoding round-tripped payload: %v", err)
		}
		if string(decoded) != string(data) {
			t.Errorf("round-tripped payload = %q, want %q", decoded, data)
		}
	}
}

func TestParseDataURI(t *testing.T) {
	tests := []struct {
		name        string
		uri         string
		wantType    string
		wantPayload string
		wantErr     bool
	}{
		{
			name:        "jpeg base64",
			uri:         "data:image/jpeg;base64,/9j/4AAQ",
			wantType:    testImageJPEGMIME,
			wantPayload: "/9j/4AAQ",
		},
		{
			name:        "png base64",
			uri:         "data:image/png;base64,iVBORw0K",
			wantType:    testImagePNGMIME,
			wantPayload: "iVBORw0K",
		},
		{
			name:    "missing media type",
			uri:     "data:;base64,YWJj",
			wantErr: true,
		},
		{
			name:        "content type normalized to lowercase and trimmed",
			uri:         "data:IMAGE/PNG ;base64,iVBORw0K",
			wantType:    testImagePNGMIME,
			wantPayload: "iVBORw0K",
		},
		{
			name:        "uppercase scheme",
			uri:         "DATA:image/png;base64,iVBORw0K",
			wantType:    testImagePNGMIME,
			wantPayload: "iVBORw0K",
		},
		{
			name:    "missing comma",
			uri:     "data:image/jpeg;base64",
			wantErr: true,
		},
		{
			name:    "missing base64 marker",
			uri:     "data:image/jpeg,raw",
			wantErr: true,
		},
		{
			name:    "no semicolon before comma",
			uri:     "data:image/jpeg,abc",
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ct, b64, err := parseDataURI(tt.uri)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("expected error, got contentType=%q payload=%q", ct, b64)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if ct != tt.wantType {
				t.Fatalf("contentType: want %q, got %q", tt.wantType, ct)
			}
			if b64 != tt.wantPayload {
				t.Fatalf("payload: want %q, got %q", tt.wantPayload, b64)
			}
		})
	}
}

func TestReplaceMediaURLsStep_RejectsTooManyEntries(t *testing.T) {
	var hits atomic.Int32
	imageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		w.Header().Set("Content-Type", testImagePNGMIME)
		_, _ = w.Write([]byte("png-data"))
	}))
	defer imageServer.Close()

	step, err := NewReplaceMediaURLsStep(nil, map[string]any{"max_multimodal_entries": 2})
	if err != nil {
		t.Fatal(err)
	}

	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{"type": "image_url", "image_url": map[string]any{"url": imageServer.URL + "/a.png"}},
						map[string]any{"type": "image_url", "image_url": map[string]any{"url": imageServer.URL + "/b.png"}},
						map[string]any{"type": "image_url", "image_url": map[string]any{"url": imageServer.URL + "/c.png"}},
					},
				},
			},
		},
	}

	err = step.Execute(context.Background(), reqCtx)
	if err == nil {
		t.Fatal("expected error for exceeding max_multimodal_entries")
	}
	if !strings.Contains(err.Error(), "too many multimodal entries") {
		t.Fatalf("unexpected error message: %v", err)
	}
	if !strings.Contains(err.Error(), "got 3") || !strings.Contains(err.Error(), "max 2") {
		t.Fatalf("error should include counts: %v", err)
	}
	if hits.Load() != 0 {
		t.Fatalf("expected no downloads on rejection, got %d hits", hits.Load())
	}
	if len(reqCtx.MultimodalEntries) != 0 {
		t.Fatalf("expected no entries populated on rejection, got %d", len(reqCtx.MultimodalEntries))
	}
}

func TestReplaceMediaURLsStep_RejectsNegativeMaxEntries(t *testing.T) {
	_, err := NewReplaceMediaURLsStep(nil, map[string]any{"max_multimodal_entries": -1})
	if err == nil {
		t.Fatal("expected error for negative max_multimodal_entries")
	}
}

func TestReplaceMediaURLsStep_AllowsAtLimit(t *testing.T) {
	imageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", testImagePNGMIME)
		_, _ = w.Write([]byte("png-data"))
	}))
	defer imageServer.Close()

	step := newLoopbackStep(t, map[string]any{"max_multimodal_entries": 2})

	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{"type": "image_url", "image_url": map[string]any{"url": imageServer.URL + "/a.png"}},
						map[string]any{"type": "image_url", "image_url": map[string]any{"url": imageServer.URL + "/b.png"}},
					},
				},
			},
		},
	}

	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error at limit: %v", err)
	}
	if len(reqCtx.MultimodalEntries) != 2 {
		t.Fatalf("expected 2 entries, got %d", len(reqCtx.MultimodalEntries))
	}
}

func TestReplaceMediaURLsStep_MultipleImages(t *testing.T) {
	imageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", testImagePNGMIME)
		_, _ = w.Write([]byte("png-data"))
	}))
	defer imageServer.Close()

	step := newLoopbackStep(t, map[string]any{})

	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":      "image_url",
							"image_url": map[string]any{"url": imageServer.URL + "/a.png"},
						},
						map[string]any{
							"type":      "image_url",
							"image_url": map[string]any{"url": imageServer.URL + "/b.png"},
						},
						map[string]any{
							"type":      "image_url",
							"image_url": map[string]any{"url": imageServer.URL + "/c.png"},
						},
					},
				},
			},
		},
	}

	err := step.Execute(context.Background(), reqCtx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(reqCtx.MultimodalEntries) != 3 {
		t.Fatalf("expected 3 entries, got %d", len(reqCtx.MultimodalEntries))
	}
	content := reqCtx.Body["messages"].([]any)[0].(map[string]any)["content"].([]any)
	for i, part := range content {
		url, _ := part.(map[string]any)["image_url"].(map[string]any)["url"].(string)
		if !strings.HasPrefix(url, "data:image/png;base64,") {
			t.Fatalf("part %d not inlined: %s", i, url)
		}
	}
}

func TestReplaceMediaURLsStep_RejectsNonPositiveMaxConcurrent(t *testing.T) {
	for _, v := range []int{0, -1} {
		if _, err := NewReplaceMediaURLsStep(nil, map[string]any{"max_concurrent_downloads": v}); err == nil {
			t.Fatalf("expected error for max_concurrent_downloads=%d", v)
		}
	}
	if _, err := NewReplaceMediaURLsStep(nil, map[string]any{"max_concurrent_downloads": 5}); err != nil {
		t.Fatalf("unexpected error for positive max_concurrent_downloads: %v", err)
	}
}

func TestReplaceMediaURLsStep_Name(t *testing.T) {
	step, err := NewReplaceMediaURLsStep(nil, map[string]any{})
	if err != nil {
		t.Fatal(err)
	}
	if step.Name() != ReplaceMediaURLsStepName {
		t.Fatalf("Name() = %q, want %q", step.Name(), ReplaceMediaURLsStepName)
	}
}

func TestReplaceMediaURLsStep_MalformedBody(t *testing.T) {
	tests := []struct {
		name string
		body map[string]any
	}{
		{
			name: "no messages key",
			body: map[string]any{"model": "x"},
		},
		{
			name: "message not a map",
			body: map[string]any{"messages": []any{"not-a-map"}},
		},
		{
			name: "content part not a map",
			body: map[string]any{
				"messages": []any{
					map[string]any{"role": "user", "content": []any{"not-a-map"}},
				},
			},
		},
		{
			name: "image_url field not a map",
			body: map[string]any{
				"messages": []any{
					map[string]any{"role": "user", "content": []any{
						map[string]any{"type": "image_url", "image_url": "not-a-map"},
					}},
				},
			},
		},
		{
			name: "url field not a string",
			body: map[string]any{
				"messages": []any{
					map[string]any{"role": "user", "content": []any{
						map[string]any{"type": "image_url", "image_url": map[string]any{"url": 123}},
					}},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			step, _ := NewReplaceMediaURLsStep(nil, map[string]any{})
			reqCtx := &pipeline.RequestContext{Body: tt.body}
			if err := step.Execute(context.Background(), reqCtx); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(reqCtx.MultimodalEntries) != 0 {
				t.Fatalf("expected 0 multimodal entries, got %d", len(reqCtx.MultimodalEntries))
			}
		})
	}
}

func TestReplaceMediaURLsStep_InvalidDataURI(t *testing.T) {
	step, _ := NewReplaceMediaURLsStep(nil, map[string]any{})
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":      "image_url",
							"image_url": map[string]any{"url": "data:image/jpeg;base64"},
						},
					},
				},
			},
		},
	}
	err := step.Execute(context.Background(), reqCtx)
	if err == nil {
		t.Fatal("expected error for malformed data URI")
	}
	if !strings.Contains(err.Error(), "parsing data URI") {
		t.Fatalf("unexpected error message: %v", err)
	}
}

func TestReplaceMediaURLsStep_EmptyContentType(t *testing.T) {
	imageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header()["Content-Type"] = nil // suppress net/http content sniffing
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("raw-bytes"))
	}))
	defer imageServer.Close()

	step := newLoopbackStep(t, map[string]any{})
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":      "image_url",
							"image_url": map[string]any{"url": imageServer.URL + "/raw"},
						},
					},
				},
			},
		},
	}
	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(reqCtx.MultimodalEntries) != 1 {
		t.Fatalf("expected 1 multimodal entry, got %d", len(reqCtx.MultimodalEntries))
	}
	// The rewritten data URI carries the fallback type.
	part := reqCtx.Body["messages"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)
	url, _ := part["image_url"].(map[string]any)["url"].(string)
	if !strings.HasPrefix(url, "data:"+defaultContentType+";base64,") {
		t.Fatalf("expected url with default type, got %s", url)
	}
}

func TestReplaceMediaURLsStep_DownloadUnreachable(t *testing.T) {
	imageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {}))
	deadURL := imageServer.URL + "/gone.png"
	imageServer.Close() // nothing is listening on this address now

	step := newLoopbackStep(t, map[string]any{})
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":      "image_url",
							"image_url": map[string]any{"url": deadURL},
						},
					},
				},
			},
		},
	}
	err := step.Execute(context.Background(), reqCtx)
	if err == nil {
		t.Fatal("expected error for unreachable download host")
	}
	if !strings.Contains(err.Error(), "downloading") {
		t.Fatalf("unexpected error message: %v", err)
	}
}

// Structural guard, not a behavioral proxy test. The SSRF dial guard requires a
// custom transport, so the downloader clones http.DefaultTransport to retain
// its Proxy: http.ProxyFromEnvironment. That is the only reason image fetches
// honor HTTP_PROXY/HTTPS_PROXY. A custom transport without a Proxy field (as in
// pkg/gateway/client.go) would silently bypass the proxy; this test fails if
// that regression is introduced here.
func TestReplaceMediaURLsStep_ClientPreservesProxy(t *testing.T) {
	step, err := NewReplaceMediaURLsStep(nil, map[string]any{})
	if err != nil {
		t.Fatal(err)
	}
	rmu, ok := step.(*ReplaceMediaURLsStep)
	if !ok {
		t.Fatalf("expected *ReplaceMediaURLsStep, got %T", step)
	}
	transport, ok := rmu.client.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("expected *http.Transport, got %T", rmu.client.Transport)
	}
	if transport.Proxy == nil {
		t.Fatal("downloader transport must keep Proxy (http.ProxyFromEnvironment) so HTTP(S)_PROXY is honored")
	}
}

func TestReplaceMediaURLsStep_DownloadInvalidURL(t *testing.T) {
	step, _ := NewReplaceMediaURLsStep(nil, map[string]any{})
	rmu := step.(*ReplaceMediaURLsStep)

	// 0x7f (DEL) is an invalid control character in a URL; NewRequestWithContext
	// fails before any network call.
	_, _, err := rmu.download(context.Background(), "http://\x7f/control-char", ModalityImage)
	if err == nil {
		t.Fatal("expected error building request for URL with control character")
	}
}

func TestReplaceMediaURLsStep_RejectsOversizedBody(t *testing.T) {
	var hits atomic.Int32
	imageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		w.Header().Set("Content-Type", testImagePNGMIME)
		// No Content-Length set: force the size check to happen during the read.
		w.(http.Flusher).Flush()
		_, _ = w.Write(make([]byte, config.BytesPerMB+1))
	}))
	defer imageServer.Close()

	step := newLoopbackStep(t, map[string]any{"max_download_size": 1})

	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{"type": "image_url", "image_url": map[string]any{"url": imageServer.URL + "/big.png"}},
					},
				},
			},
		},
	}

	err := step.Execute(context.Background(), reqCtx)
	if err == nil {
		t.Fatal("expected error for oversized download")
	}
	if !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("expected ErrBadRequest, got %v", err)
	}
	if len(reqCtx.MultimodalEntries) != 0 {
		t.Fatalf("expected no entries populated on rejection, got %d", len(reqCtx.MultimodalEntries))
	}
}

func TestReplaceMediaURLsStep_RejectsOversizedContentLength(t *testing.T) {
	imageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", testImagePNGMIME)
		w.Header().Set("Content-Length", "1048577") // config.BytesPerMB + 1
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(make([]byte, config.BytesPerMB+1))
	}))
	defer imageServer.Close()

	rmu := newLoopbackStep(t, map[string]any{"max_download_size": 1})

	_, _, err := rmu.download(context.Background(), imageServer.URL+"/big.png", ModalityImage)
	if err == nil {
		t.Fatal("expected error for oversized Content-Length")
	}
	if !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("expected ErrBadRequest, got %v", err)
	}
}

func TestReplaceMediaURLsStep_AllowsBodyAtCap(t *testing.T) {
	const capMB = 1
	const capBytes = capMB * config.BytesPerMB
	imageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", testImagePNGMIME)
		_, _ = w.Write(make([]byte, capBytes))
	}))
	defer imageServer.Close()

	step := newLoopbackStep(t, map[string]any{"max_download_size": capMB})
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{"type": "image_url", "image_url": map[string]any{"url": imageServer.URL + "/atcap.png"}},
					},
				},
			},
		},
	}

	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error for body exactly at cap: %v", err)
	}
	if len(reqCtx.MultimodalEntries) != 1 {
		t.Fatalf("expected 1 entry, got %d", len(reqCtx.MultimodalEntries))
	}
	// The rewritten data URI carries the exact cap-sized payload.
	part := reqCtx.Body["messages"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)
	url, _ := part["image_url"].(map[string]any)["url"].(string)
	want := "data:image/png;base64," + base64.StdEncoding.EncodeToString(make([]byte, capBytes))
	if url != want {
		t.Fatalf("url mismatch: got %q want %q", url, want)
	}
}

// A request may carry several image_url entries. The per-download cap must
// bound each one independently: a single oversized entry rejects the whole
// request even when the others are within the cap.
func TestReplaceMediaURLsStep_RejectsOneOversizedAmongMany(t *testing.T) {
	imageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", testImagePNGMIME)
		if strings.HasPrefix(r.URL.Path, "/big") {
			_, _ = w.Write(make([]byte, config.BytesPerMB+1))
			return
		}
		_, _ = w.Write(make([]byte, 4))
	}))
	defer imageServer.Close()

	step := newLoopbackStep(t, map[string]any{"max_download_size": 1})
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{"type": "image_url", "image_url": map[string]any{"url": imageServer.URL + "/small1.png"}},
						map[string]any{"type": "image_url", "image_url": map[string]any{"url": imageServer.URL + "/big.png"}},
						map[string]any{"type": "image_url", "image_url": map[string]any{"url": imageServer.URL + "/small2.png"}},
					},
				},
			},
		},
	}

	err := step.Execute(context.Background(), reqCtx)
	if err == nil {
		t.Fatal("expected error when one of several entries is oversized")
	}
	if !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("expected ErrBadRequest, got %v", err)
	}
}

func TestReplaceMediaURLsStep_RejectsInvalidMaxDownloadSize(t *testing.T) {
	// Values that are zero, negative, or too large to convert to bytes without
	// overflowing int64 are rejected. Overflow would cause the io.LimitReader
	// sentinel (maxDownloadSize+1) to become negative, accepting oversized bodies.
	limit := (math.MaxInt - 1) / config.BytesPerMB
	for _, v := range []int{0, -1, limit + 1, math.MaxInt} {
		if _, err := NewReplaceMediaURLsStep(nil, map[string]any{"max_download_size": v}); err == nil {
			t.Fatalf("expected error for max_download_size=%d", v)
		}
	}
}

func TestReplaceMediaURLsStep_DownloadTruncatedBody(t *testing.T) {
	imageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hj, ok := w.(http.Hijacker)
		if !ok {
			return
		}
		conn, _, err := hj.Hijack()
		if err != nil {
			return
		}
		// Promise 100 bytes, send 5, then close: the client's io.ReadAll sees an
		// unexpected EOF.
		_, _ = conn.Write([]byte("HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nshort"))
		_ = conn.Close()
	}))
	defer imageServer.Close()

	rmu := newLoopbackStep(t, map[string]any{})

	_, _, err := rmu.download(context.Background(), imageServer.URL+"/truncated", ModalityImage)
	if err == nil {
		t.Fatal("expected error reading truncated response body")
	}
}

func TestAddressGuard_BlockedIP(t *testing.T) {
	tests := []struct {
		name string
		ip   string
		want bool
	}{
		{"metadata link-local", "169.254.169.254", true},
		{"loopback v4", "127.0.0.1", true},
		{"loopback v6", "::1", true},
		{"link-local v6", "fe80::1", true},
		{"unspecified v4", "0.0.0.0", true},
		{"unspecified v6", "::", true},
		{"cgnat", "100.64.1.1", true},
		{"private 10", "10.0.0.1", true},
		{"private 172", "172.16.0.1", true},
		{"private 192", "192.168.1.1", true},
		{"unique-local v6", "fc00::1", true},
		{"ipv4-mapped metadata", "::ffff:169.254.169.254", true},
		{"ipv4-mapped private", "::ffff:10.0.0.1", true},
		{"public v4", "8.8.8.8", false},
		{"public v6", "2001:4860:4860::8888", false},
	}
	guard := &addressGuard{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ip := net.ParseIP(tt.ip)
			if ip == nil {
				t.Fatalf("could not parse %q", tt.ip)
			}
			if got := guard.blockedIP(ip); got != tt.want {
				t.Fatalf("blockedIP(%s) = %v, want %v", tt.ip, got, tt.want)
			}
		})
	}
}

func TestAddressGuard_AllowPrivate(t *testing.T) {
	guard := &addressGuard{allowPrivate: true}
	// RFC1918 ranges are permitted when opted in...
	for _, ip := range []string{"10.0.0.1", "172.16.0.1", "192.168.1.1"} {
		if guard.blockedIP(net.ParseIP(ip)) {
			t.Errorf("blockedIP(%s) = true, want false with allowPrivate", ip)
		}
	}
	// ...but the metadata endpoint and other special ranges stay blocked.
	// allowPrivate is RFC1918-only: IPv6 unique-local (fc00::/7) must not leak
	// through, even though net.IP.IsPrivate treats it as private.
	for _, ip := range []string{"169.254.169.254", "127.0.0.1", "0.0.0.0", "100.64.1.1", "fc00::1"} {
		if !guard.blockedIP(net.ParseIP(ip)) {
			t.Errorf("blockedIP(%s) = false, want true even with allowPrivate", ip)
		}
	}
}

func TestAddressGuard_HostAllowed(t *testing.T) {
	open := &addressGuard{}
	if !open.hostAllowed("anything.example.com") {
		t.Fatal("empty allowlist must allow any host")
	}

	restricted := &addressGuard{allowedDomains: map[string]struct{}{"images.example.com": {}}}
	if !restricted.hostAllowed("images.example.com") {
		t.Fatal("listed host must be allowed")
	}
	if !restricted.hostAllowed("IMAGES.EXAMPLE.COM") {
		t.Fatal("host match must be case-insensitive")
	}
	if restricted.hostAllowed("evil.example.com") {
		t.Fatal("unlisted host must be rejected")
	}
}

// download rejects non-http(s) schemes before any network call.
func TestReplaceMediaURLsStep_RejectsScheme(t *testing.T) {
	rmu := newLoopbackStep(t, map[string]any{})
	for _, raw := range []string{"file:///etc/passwd", "gopher://host/1", "ftp://host/x"} {
		_, _, err := rmu.download(context.Background(), raw, ModalityImage)
		if err == nil {
			t.Fatalf("expected scheme %q to be rejected", raw)
		}
		if !errors.Is(err, pipeline.ErrBadRequest) {
			t.Fatalf("scheme rejection must be a bad request, got %v", err)
		}
	}
}

// A dial to a blocked range surfaces as a client error (ErrBadRequest), not a
// generic gateway fault, so the handler maps it to a 4xx.
func TestReplaceMediaURLsStep_BlocksMetadataIP(t *testing.T) {
	rmu := newLoopbackStep(t, map[string]any{"download_timeout": "2s"})
	_, _, err := rmu.download(context.Background(), "http://169.254.169.254/latest/meta-data/", ModalityImage)
	if err == nil {
		t.Fatal("expected metadata IP fetch to be blocked")
	}
	if !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("blocked address must classify as bad request, got %v", err)
	}
}

// An allowed public host that 302-redirects to a blocked address is rejected at
// the dial of the redirect hop.
func TestReplaceMediaURLsStep_BlocksRedirectToPrivate(t *testing.T) {
	redirector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "http://169.254.169.254/latest/meta-data/", http.StatusFound)
	}))
	defer redirector.Close()

	// Loopback allowed so the first hop (the httptest server) connects; the
	// metadata redirect target is link-local and stays blocked regardless.
	rmu := newLoopbackStep(t, map[string]any{"download_timeout": "2s"})
	_, _, err := rmu.download(context.Background(), redirector.URL+"/start", ModalityImage)
	if err == nil {
		t.Fatal("expected redirect to metadata IP to be blocked")
	}
	if !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("blocked redirect must classify as bad request, got %v", err)
	}
}

// A hostname that resolves to a blocked IP is caught at dial time, defeating a
// DNS-rebinding bypass. localhost resolves to loopback, blocked by default.
func TestReplaceMediaURLsStep_BlocksHostnameResolvingToPrivate(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("data"))
	}))
	defer server.Close()

	_, port, err := net.SplitHostPort(strings.TrimPrefix(server.URL, "http://"))
	if err != nil {
		t.Fatal(err)
	}

	// Default guard: loopback blocked. "localhost" resolves to 127.0.0.1/::1.
	built, err := NewReplaceMediaURLsStep(nil, map[string]any{"download_timeout": "2s"})
	if err != nil {
		t.Fatal(err)
	}
	step := built.(*ReplaceMediaURLsStep)
	_, _, err = step.download(context.Background(), "http://localhost:"+port+"/x", ModalityImage)
	if err == nil {
		t.Fatal("expected hostname resolving to loopback to be blocked")
	}
	if !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("blocked resolved host must classify as bad request, got %v", err)
	}
}

// With a domain allowlist set, only listed hosts are fetched; others are
// rejected before any connection.
func TestReplaceMediaURLsStep_DomainAllowlist(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", testImagePNGMIME)
		_, _ = w.Write([]byte("img"))
	}))
	defer server.Close()

	host := strings.TrimPrefix(server.URL, "http://")
	hostname, _, err := net.SplitHostPort(host)
	if err != nil {
		t.Fatal(err)
	}

	allowed := newLoopbackStep(t, map[string]any{"allowed_domains": []any{hostname}})
	if _, _, err := allowed.download(context.Background(), server.URL+"/ok.png", ModalityImage); err != nil {
		t.Fatalf("listed host must be fetchable: %v", err)
	}

	denied := newLoopbackStep(t, map[string]any{"allowed_domains": []any{"images.example.com"}})
	_, _, err = denied.download(context.Background(), server.URL+"/ok.png", ModalityImage)
	if err == nil {
		t.Fatal("unlisted host must be rejected")
	}
	if !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("allowlist rejection must classify as bad request, got %v", err)
	}
}

// allowed_domains entries must be strings.
func TestReplaceMediaURLsStep_RejectsNonStringAllowedDomain(t *testing.T) {
	_, err := NewReplaceMediaURLsStep(nil, map[string]any{"allowed_domains": []any{123}})
	if err == nil {
		t.Fatal("expected error for non-string allowed_domains entry")
	}
}

// A list arriving as []string (a programmatic caller, not the YAML path) must
// build the allowlist, not silently fall back to allow-all.
func TestReplaceMediaURLsStep_AllowedDomainsStringSlice(t *testing.T) {
	step, err := NewReplaceMediaURLsStep(nil, map[string]any{"allowed_domains": []string{"images.example.com"}})
	if err != nil {
		t.Fatal(err)
	}
	guard := step.(*ReplaceMediaURLsStep).guard
	if guard.hostAllowed("evil.example.com") {
		t.Fatal("allowlist must reject unlisted host")
	}
	if !guard.hostAllowed("images.example.com") {
		t.Fatal("allowlist must permit listed host")
	}
}

// An allowed_domains value of an unsupported type must error, not silently
// disable the allowlist.
func TestReplaceMediaURLsStep_RejectsNonListAllowedDomains(t *testing.T) {
	_, err := NewReplaceMediaURLsStep(nil, map[string]any{"allowed_domains": "images.example.com"})
	if err == nil {
		t.Fatal("expected error for non-list allowed_domains")
	}
}

func TestReplaceMediaURLsStep_RejectsNonImageDataURI(t *testing.T) {
	step, _ := NewReplaceMediaURLsStep(nil, map[string]any{})
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":      "image_url",
							"image_url": map[string]any{"url": "data:text/html;base64,PGgxPmhpPC9oMT4="},
						},
					},
				},
			},
		},
	}
	err := step.Execute(context.Background(), reqCtx)
	if err == nil {
		t.Fatal("expected error for non-image data URI content type")
	}
	if !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("expected ErrBadRequest, got %v", err)
	}
}

func TestReplaceMediaURLsStep_RejectsMissingMediaType(t *testing.T) {
	step, _ := NewReplaceMediaURLsStep(nil, map[string]any{})
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":      "image_url",
							"image_url": map[string]any{"url": "data:;base64,AAAA"},
						},
					},
				},
			},
		},
	}
	err := step.Execute(context.Background(), reqCtx)
	if err == nil {
		t.Fatal("expected error for data URI missing media type")
	}
	if !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("expected ErrBadRequest, got %v", err)
	}
}

func TestReplaceMediaURLsStep_CancelledContextSkipsDataURIParse(t *testing.T) {
	step, _ := NewReplaceMediaURLsStep(nil, map[string]any{})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":      "image_url",
							"image_url": map[string]any{"url": "data:image/jpeg,raw"},
						},
					},
				},
			},
		},
	}
	err := step.Execute(ctx, reqCtx)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
}

// ---- Audio / video ingestion -----------------------------------------------

// TestReplaceMediaURLsStep_AudioURL_Downloads asserts that an audio_url
// is fetched, size-capped, MIME-checked, inlined as a data URI, and
// added to MultimodalEntries as one audio entry.
func TestReplaceMediaURLsStep_AudioURL_Downloads(t *testing.T) {
	audioServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", testAudioWAVMIME)
		_, _ = w.Write([]byte("wav-bytes"))
	}))
	defer audioServer.Close()

	step := newLoopbackStep(t, map[string]any{"download_timeout": "5s"})
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":      "audio_url",
							"audio_url": map[string]any{"url": audioServer.URL + "/clip.wav"},
						},
					},
				},
			},
		},
	}

	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(reqCtx.MultimodalEntries) != 1 {
		t.Fatalf("expected 1 audio entry in MultimodalEntries, got %d", len(reqCtx.MultimodalEntries))
	}
	if reqCtx.MultimodalEntries[0].Modality != ModalityAudio {
		t.Fatalf("expected Modality=%q, got %q", ModalityAudio, reqCtx.MultimodalEntries[0].Modality)
	}
	msgs := reqCtx.Body["messages"].([]any)
	inner := msgs[0].(map[string]any)["content"].([]any)[0].(map[string]any)["audio_url"].(map[string]any)
	url := inner["url"].(string)
	if !strings.HasPrefix(url, "data:audio/wav;base64,") {
		t.Fatalf("expected inlined data URI, got %s", url)
	}
}

// TestReplaceMediaURLsStep_ImageURL_ParameterOnlyContentType covers a
// Content-Type that carries only parameters and no type (e.g.
// "; charset=utf-8"). The step should strip the parameters, notice
// the result is empty, and fall back to the default type so the
// rewritten URL stays well-formed. If the fallback ran before the
// strip, the URL would come out as data:;base64,... and be unusable.
func TestReplaceMediaURLsStep_ImageURL_ParameterOnlyContentType(t *testing.T) {
	oddServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "; charset=utf-8")
		_, _ = w.Write([]byte("some-bytes"))
	}))
	defer oddServer.Close()

	step := newLoopbackStep(t, map[string]any{"download_timeout": "5s"})
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":      "image_url",
							"image_url": map[string]any{"url": oddServer.URL + "/thing.jpg"},
						},
					},
				},
			},
		},
	}
	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("expected acceptance with fallback content type, got %v", err)
	}
	part := reqCtx.Body["messages"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)
	rewritten, _ := part["image_url"].(map[string]any)["url"].(string)
	if !strings.HasPrefix(rewritten, "data:"+defaultContentType+";base64,") {
		t.Errorf("rewritten url = %q, want prefix data:%s;base64,", rewritten, defaultContentType)
	}
}

// TestNormalizeMediaType covers what a Content-Type header or a data URI
// media type is reduced to before it reaches the allowlist or the emitted
// URI. The comma cases are the ones that matter for the emitted URI: RFC 2397
// ends the metadata at the first comma, so a type that kept one would move the
// payload boundary.
func TestNormalizeMediaType(t *testing.T) {
	tests := []struct {
		in   string
		want string
	}{
		{"image/png", "image/png"},
		{"IMAGE/PNG", "image/png"},
		{"  image/png  ", "image/png"},
		{"image/png; charset=utf-8", "image/png"},
		{`video/mp4; codecs="avc1.4D401E"`, "video/mp4"},
		{"image/png, image/png", "image/png"},
		{"image/png,image/jpeg", "image/png"},
		{`video/mp4; codecs="avc1.4D401E, mp4a.40.2"`, "video/mp4"},
		{"", ""},
		{"; charset=utf-8", ""},
		{", image/png", ""},
	}
	for _, tc := range tests {
		if got := normalizeMediaType(tc.in); got != tc.want {
			t.Errorf("normalizeMediaType(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

// TestReplaceMediaURLsStep_ImageURL_MultiValueContentType covers an origin
// whose Content-Type holds more than one value. The comma must not reach the
// rewritten data URI: a reader splitting on the first comma would take
// "image/png" as the whole metadata and " image/png;base64,..." as the
// payload, and the base64 decode would fail.
func TestReplaceMediaURLsStep_ImageURL_MultiValueContentType(t *testing.T) {
	data := []byte("png-bytes")
	oddServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", testImagePNGMIME+", "+testImagePNGMIME)
		_, _ = w.Write(data)
	}))
	defer oddServer.Close()

	step := newLoopbackStep(t, map[string]any{"download_timeout": "5s"})
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":      "image_url",
							"image_url": map[string]any{"url": oddServer.URL + "/thing.png"},
						},
					},
				},
			},
		},
	}
	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	part := reqCtx.Body["messages"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)
	rewritten, _ := part["image_url"].(map[string]any)["url"].(string)
	want := "data:" + testImagePNGMIME + ";base64," + base64.StdEncoding.EncodeToString(data)
	if rewritten != want {
		t.Fatalf("rewritten url = %q, want %q", rewritten, want)
	}
	// The URI must survive a round trip through the step's own reader.
	ct, b64, err := parseDataURI(rewritten)
	if err != nil {
		t.Fatalf("parseDataURI on the rewritten URI: %v", err)
	}
	if ct != testImagePNGMIME {
		t.Errorf("round-tripped content type = %q, want %q", ct, testImagePNGMIME)
	}
	decoded, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		t.Fatalf("decoding the round-tripped payload: %v", err)
	}
	if string(decoded) != string(data) {
		t.Errorf("round-tripped payload = %q, want %q", decoded, data)
	}
}

// TestReplaceMediaURLsStep_AudioVideo_AcceptsContentTypeWithParams asserts
// that a real audio_url / video_url whose origin returns a Content-Type
// with MIME parameters (";codecs=...", ";charset=..." and so on) is
// accepted, and that the parameters are stripped at the download
// boundary so the rewritten data URI carries a bare MIME. The codecs
// case embeds a comma inside a quoted parameter value, which is what
// breaks parseDataURI when the raw header value flows through.
func TestReplaceMediaURLsStep_AudioVideo_AcceptsContentTypeWithParams(t *testing.T) {
	for _, tc := range []struct {
		name          string
		partType      string
		urlKey        string
		serverHeader  string
		wantMediaType string
	}{
		{"audio with charset", "audio_url", "audio_url", "audio/wav; charset=US-ASCII", testAudioWAVMIME},
		{"video with codecs", "video_url", "video_url", `video/mp4; codecs="avc1.4D401E,mp4a.40.2"`, testVideoMP4MIME},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ct := tc.serverHeader
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", ct)
				_, _ = w.Write([]byte("bytes"))
			}))
			defer server.Close()

			step := newLoopbackStep(t, map[string]any{"download_timeout": "5s"})
			reqCtx := &pipeline.RequestContext{
				Body: map[string]any{
					"messages": []any{
						map[string]any{
							"role": "user",
							"content": []any{
								map[string]any{
									"type":    tc.partType,
									tc.urlKey: map[string]any{"url": server.URL + "/clip"},
								},
							},
						},
					},
				},
			}
			if err := step.Execute(context.Background(), reqCtx); err != nil {
				t.Fatalf("expected accepted Content-Type %q, got %v", tc.serverHeader, err)
			}
			if got := len(reqCtx.MultimodalEntries); got != 1 {
				t.Fatalf("expected 1 entry, got %d", got)
			}
			// The rewritten URL must round-trip through parseDataURI. When
			// the header carries a parameter value containing a comma, the
			// pre-normalization code produced a URL whose first comma was
			// inside the codecs list, so parseDataURI failed.
			part := reqCtx.Body["messages"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)
			inner := part[tc.urlKey].(map[string]any)
			rewritten, _ := inner["url"].(string)
			gotCT, _, err := parseDataURI(rewritten)
			if err != nil {
				t.Fatalf("rewritten URL not parsable as data URI: %v\nurl=%s", err, rewritten)
			}
			if gotCT != tc.wantMediaType {
				t.Errorf("parseDataURI(rewritten).contentType = %q, want %q", gotCT, tc.wantMediaType)
			}
		})
	}
}

// TestReplaceMediaURLsStep_AudioURL_RejectsUnexpectedContentType asserts an
// audio_url whose origin serves a non-audio Content-Type (e.g. text/html)
// is rejected as ErrBadRequest. This closes an SSRF-style widening where
// a caller could exploit an audio_url slot to smuggle text or HTML.
func TestReplaceMediaURLsStep_AudioURL_RejectsUnexpectedContentType(t *testing.T) {
	badServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		_, _ = w.Write([]byte("<html></html>"))
	}))
	defer badServer.Close()

	step := newLoopbackStep(t, map[string]any{"download_timeout": "5s"})
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":      "audio_url",
							"audio_url": map[string]any{"url": badServer.URL + "/clip.wav"},
						},
					},
				},
			},
		},
	}
	err := step.Execute(context.Background(), reqCtx)
	if err == nil {
		t.Fatal("expected error for audio_url served as text/html")
	}
	if !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("expected ErrBadRequest, got %v", err)
	}
}

// TestReplaceMediaURLsStep_VideoURL_RejectsUnexpectedContentType mirrors the
// audio case for video_url served with a non-video Content-Type.
func TestReplaceMediaURLsStep_VideoURL_RejectsUnexpectedContentType(t *testing.T) {
	badServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		_, _ = w.Write([]byte("<html></html>"))
	}))
	defer badServer.Close()

	step := newLoopbackStep(t, map[string]any{"download_timeout": "5s"})
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":      "video_url",
							"video_url": map[string]any{"url": badServer.URL + "/clip.mp4"},
						},
					},
				},
			},
		},
	}
	err := step.Execute(context.Background(), reqCtx)
	if err == nil {
		t.Fatal("expected error for video_url served as text/html")
	}
	if !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("expected ErrBadRequest, got %v", err)
	}
}

// readTrackingBody is a response body that records whether anything read it.
// Read returns EOF immediately, so a caller that does reach the body gets an
// empty payload rather than blocking.
type readTrackingBody struct {
	read atomic.Bool
}

func (b *readTrackingBody) Read([]byte) (int, error) {
	b.read.Store(true)
	return 0, io.EOF
}

func (b *readTrackingBody) Close() error { return nil }

// TestReplaceMediaURLsStep_Download_RejectsContentTypeBeforeReadingBody pins the
// order of the two download-path checks. An audio origin serving a type the
// allowlist rejects is turned away on its headers, with the body left unread.
// With the check after the read instead, the request is rejected just the same,
// but only once up to the modality's cap has crossed the network and been held
// in memory, which is the cost the cap exists to bound.
//
// ContentLength is -1, as it is for a chunked response, so the Content-Length
// guard does not fire and the ordering is what decides whether the body is read.
//
// The assertion is on whether the body was read at all. Elapsed time, or bytes
// counted inside a test server's handler, would both race with the client
// closing the connection.
func TestReplaceMediaURLsStep_Download_RejectsContentTypeBeforeReadingBody(t *testing.T) {
	body := &readTrackingBody{}
	step := newLoopbackStep(t, map[string]any{"download_timeout": "5s"})
	step.client = &http.Client{Transport: roundTripperFunc(func(_ *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode:    http.StatusOK,
			Header:        http.Header{"Content-Type": []string{"text/html"}},
			Body:          body,
			ContentLength: -1,
		}, nil
	})}

	_, _, err := step.download(context.Background(), "http://media.invalid/clip.wav", ModalityAudio)
	if err == nil {
		t.Fatal("expected error for audio download served as text/html")
	}
	if !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("expected ErrBadRequest, got %v", err)
	}
	if body.read.Load() {
		t.Error("response body was read before the Content-Type was rejected")
	}
}

// TestReplaceMediaURLsStep_ImageURL_PermissiveContentType documents that
// image_url downloads accept any Content-Type under the built-in default
// allowlist. Audio and video are stricter; the image default is not
// tightened to avoid breaking traffic that relies on this behavior. An
// operator who sets allowed_image_content_types explicitly does get
// enforcement here, covered by
// TestReplaceMediaURLsStep_ImageURL_ExplicitAllowlistAppliesToDownload.
func TestReplaceMediaURLsStep_ImageURL_PermissiveContentType(t *testing.T) {
	oddServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		_, _ = w.Write([]byte("not-really-an-image"))
	}))
	defer oddServer.Close()

	step := newLoopbackStep(t, map[string]any{"download_timeout": "5s"})
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":      "image_url",
							"image_url": map[string]any{"url": oddServer.URL + "/thing.jpg"},
						},
					},
				},
			},
		},
	}
	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("expected permissive behavior for image_url, got %v", err)
	}
}

// TestReplaceMediaURLsStep_ImageURL_ExplicitAllowlistAppliesToDownload
// covers the other half of the image content-type rule: once an operator
// sets allowed_image_content_types, the list is enforced on downloaded
// bytes too, not just on data URIs. Without this the origin's
// Content-Type would be inlined verbatim into the rewritten data URI and
// the operator's lockdown would be a no-op on the HTTP path.
func TestReplaceMediaURLsStep_ImageURL_ExplicitAllowlistAppliesToDownload(t *testing.T) {
	jpegServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", testImageJPEGMIME)
		_, _ = w.Write([]byte("jpeg-bytes"))
	}))
	defer jpegServer.Close()

	imageReq := func() *pipeline.RequestContext {
		return &pipeline.RequestContext{Body: map[string]any{
			"messages": []any{
				map[string]any{"role": "user", "content": []any{
					map[string]any{
						"type":      "image_url",
						"image_url": map[string]any{"url": jpegServer.URL + "/photo.jpg"},
					},
				}},
			},
		}}
	}

	// Narrowed to image/png only: the image/jpeg download is rejected.
	narrowed := newLoopbackStep(t, map[string]any{
		"allowed_image_content_types": []any{testImagePNGMIME},
	})
	err := narrowed.Execute(context.Background(), imageReq())
	if err == nil || !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("expected image/jpeg download rejected under image/png-only override, got %v", err)
	}

	// Explicitly unrestricted: the same download is accepted.
	unrestricted := newLoopbackStep(t, map[string]any{
		"allowed_image_content_types": []any{},
	})
	if err := unrestricted.Execute(context.Background(), imageReq()); err != nil {
		t.Fatalf("expected empty image allowlist to accept any download type, got %v", err)
	}
}

// TestReplaceMediaURLsStep_VideoURL_Downloads mirrors the audio case for
// video_url with a video/mp4 payload.
func TestReplaceMediaURLsStep_VideoURL_Downloads(t *testing.T) {
	videoServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", testVideoMP4MIME)
		_, _ = w.Write([]byte("mp4-bytes"))
	}))
	defer videoServer.Close()

	step := newLoopbackStep(t, map[string]any{"download_timeout": "5s"})
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":      "video_url",
							"video_url": map[string]any{"url": videoServer.URL + "/clip.mp4"},
						},
					},
				},
			},
		},
	}

	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(reqCtx.MultimodalEntries) != 1 {
		t.Fatalf("expected 1 video entry in MultimodalEntries, got %d", len(reqCtx.MultimodalEntries))
	}
	if reqCtx.MultimodalEntries[0].Modality != ModalityVideo {
		t.Fatalf("expected Modality=%q, got %q", ModalityVideo, reqCtx.MultimodalEntries[0].Modality)
	}
	msgs := reqCtx.Body["messages"].([]any)
	inner := msgs[0].(map[string]any)["content"].([]any)[0].(map[string]any)["video_url"].(map[string]any)
	url := inner["url"].(string)
	if !strings.HasPrefix(url, "data:video/mp4;base64,") {
		t.Fatalf("expected inlined data URI, got %s", url)
	}
}

// TestReplaceMediaURLsStep_AudioDataURI asserts a valid audio data URI
// under audio_url is accepted (kept in place) and added to
// MultimodalEntries as one audio entry.
func TestReplaceMediaURLsStep_AudioDataURI(t *testing.T) {
	step, _ := NewReplaceMediaURLsStep(nil, map[string]any{})
	const dataURI = "data:audio/wav;base64,UklGRg=="
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":      "audio_url",
							"audio_url": map[string]any{"url": dataURI},
						},
					},
				},
			},
		},
	}
	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(reqCtx.MultimodalEntries) != 1 {
		t.Fatalf("expected 1 audio entry in MultimodalEntries, got %d", len(reqCtx.MultimodalEntries))
	}
	if reqCtx.MultimodalEntries[0].Modality != ModalityAudio {
		t.Fatalf("expected Modality=%q, got %q", ModalityAudio, reqCtx.MultimodalEntries[0].Modality)
	}
	msgs := reqCtx.Body["messages"].([]any)
	inner := msgs[0].(map[string]any)["content"].([]any)[0].(map[string]any)["audio_url"].(map[string]any)
	if inner["url"].(string) != dataURI {
		t.Fatalf("expected data URI unchanged, got %v", inner["url"])
	}
}

// dataURIReqCtx builds a one-part request whose URL slot of the given part
// type carries a data URI of the given media type and base64 payload.
func dataURIReqCtx(partType, mediaType, b64 string) *pipeline.RequestContext {
	return &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":   partType,
							partType: map[string]any{"url": "data:" + mediaType + ";base64," + b64},
						},
					},
				},
			},
		},
	}
}

// TestReplaceMediaURLsStep_AudioVideoDataURI_RejectsOversized covers the
// per-modality cap applying to bytes that arrive inline in a URL slot, not
// only to bytes pulled over the network. The same payload sent as input_audio
// is rejected by validateInlineAudio, so accepting it here would let the two
// ways of sending one audio clip disagree.
func TestReplaceMediaURLsStep_AudioVideoDataURI_RejectsOversized(t *testing.T) {
	// 1 MB cap allows ~1_398_101 base64 chars; go well past it.
	oversized := strings.Repeat("A", 2*1024*1024)
	for _, tc := range []struct {
		name      string
		partType  string
		mediaType string
	}{
		{"audio_url", "audio_url", testAudioWAVMIME},
		{"video_url", "video_url", testVideoMP4MIME},
	} {
		t.Run(tc.name, func(t *testing.T) {
			step, err := NewReplaceMediaURLsStep(nil, map[string]any{"max_download_size": 1})
			if err != nil {
				t.Fatal(err)
			}
			reqCtx := dataURIReqCtx(tc.partType, tc.mediaType, oversized)
			err = step.Execute(context.Background(), reqCtx)
			if err == nil {
				t.Fatalf("expected error for oversized %s data URI", tc.partType)
			}
			if !errors.Is(err, pipeline.ErrBadRequest) {
				t.Fatalf("expected ErrBadRequest, got %v", err)
			}
			if len(reqCtx.MultimodalEntries) != 0 {
				t.Fatalf("rejected request must not seed entries, got %d", len(reqCtx.MultimodalEntries))
			}
		})
	}
}

// TestReplaceMediaURLsStep_AudioDataURI_UsesAudioCap checks the data URI bound
// reads the per-modality override rather than the global default: a payload
// over the 1 MB global cap is accepted once the audio cap is raised.
func TestReplaceMediaURLsStep_AudioDataURI_UsesAudioCap(t *testing.T) {
	payload := strings.Repeat("A", 2*1024*1024) // ~1.5 MB decoded
	step, err := NewReplaceMediaURLsStep(nil, map[string]any{
		"max_download_size":       1,
		"max_audio_download_size": 8,
	})
	if err != nil {
		t.Fatal(err)
	}
	reqCtx := dataURIReqCtx("audio_url", testAudioWAVMIME, payload)
	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("audio cap of 8 MB must accept a ~1.5 MB payload: %v", err)
	}
	if len(reqCtx.MultimodalEntries) != 1 {
		t.Fatalf("expected 1 entry, got %d", len(reqCtx.MultimodalEntries))
	}
}

// TestReplaceMediaURLsStep_ImageDataURI_ExemptFromSizeCap pins the deliberate
// exemption in enforceInlineSize: an image data URI is bounded by the server's
// max_request_body_size, not by the per-modality download cap.
func TestReplaceMediaURLsStep_ImageDataURI_ExemptFromSizeCap(t *testing.T) {
	oversized := strings.Repeat("A", 2*1024*1024)
	step, err := NewReplaceMediaURLsStep(nil, map[string]any{"max_download_size": 1})
	if err != nil {
		t.Fatal(err)
	}
	reqCtx := dataURIReqCtx("image_url", testImagePNGMIME, oversized)
	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("image data URI must stay exempt from the size cap, got %v", err)
	}
	if len(reqCtx.MultimodalEntries) != 1 {
		t.Fatalf("expected 1 entry, got %d", len(reqCtx.MultimodalEntries))
	}
}

// TestReplaceMediaURLsStep_VideoDataURI mirrors the audio data URI case.
func TestReplaceMediaURLsStep_VideoDataURI(t *testing.T) {
	step, _ := NewReplaceMediaURLsStep(nil, map[string]any{})
	const dataURI = "data:video/mp4;base64,AAAAHGZ0eXA="
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":      "video_url",
							"video_url": map[string]any{"url": dataURI},
						},
					},
				},
			},
		},
	}
	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(reqCtx.MultimodalEntries) != 1 {
		t.Fatalf("expected 1 video entry in MultimodalEntries, got %d", len(reqCtx.MultimodalEntries))
	}
	if reqCtx.MultimodalEntries[0].Modality != ModalityVideo {
		t.Fatalf("expected Modality=%q, got %q", ModalityVideo, reqCtx.MultimodalEntries[0].Modality)
	}
}

// TestReplaceMediaURLsStep_InputAudio_Valid asserts a well-formed input_audio
// part (base64 payload, known format) passes validation and leaves the body
// unchanged.
func TestReplaceMediaURLsStep_InputAudio_Valid(t *testing.T) {
	step, _ := NewReplaceMediaURLsStep(nil, map[string]any{})
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":        "input_audio",
							"input_audio": map[string]any{"data": "UklGRg==", "format": "wav"},
						},
					},
				},
			},
		},
	}
	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(reqCtx.MultimodalEntries) != 1 {
		t.Fatalf("expected 1 audio entry in MultimodalEntries, got %d", len(reqCtx.MultimodalEntries))
	}
	entry := reqCtx.MultimodalEntries[0]
	if entry.Modality != ModalityAudio {
		t.Fatalf("expected Modality=%q, got %q", ModalityAudio, entry.Modality)
	}
	msgs := reqCtx.Body["messages"].([]any)
	inner := msgs[0].(map[string]any)["content"].([]any)[0].(map[string]any)["input_audio"].(map[string]any)
	if inner["data"].(string) != "UklGRg==" || inner["format"].(string) != "wav" {
		t.Fatalf("expected input_audio body unchanged, got %+v", inner)
	}
}

// TestReplaceMediaURLsStep_RejectsAudioDataURIUnderImageURL asserts a
// data:audio/wav URI supplied in an image_url slot is rejected.
func TestReplaceMediaURLsStep_RejectsAudioDataURIUnderImageURL(t *testing.T) {
	step, _ := NewReplaceMediaURLsStep(nil, map[string]any{})
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":      "image_url",
							"image_url": map[string]any{"url": "data:audio/wav;base64,UklGRg=="},
						},
					},
				},
			},
		},
	}
	err := step.Execute(context.Background(), reqCtx)
	if err == nil {
		t.Fatal("expected error for audio data URI in image_url slot")
	}
	if !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("expected ErrBadRequest, got %v", err)
	}
}

// TestReplaceMediaURLsStep_RejectsImageDataURIUnderAudioURL asserts the
// symmetric case: an image data URI in an audio_url slot is rejected.
func TestReplaceMediaURLsStep_RejectsImageDataURIUnderAudioURL(t *testing.T) {
	step, _ := NewReplaceMediaURLsStep(nil, map[string]any{})
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":      "audio_url",
							"audio_url": map[string]any{"url": "data:image/jpeg;base64,/9j/4AAQ"},
						},
					},
				},
			},
		},
	}
	err := step.Execute(context.Background(), reqCtx)
	if err == nil {
		t.Fatal("expected error for image data URI in audio_url slot")
	}
	if !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("expected ErrBadRequest, got %v", err)
	}
}

// TestReplaceMediaURLsStep_InputAudio_UnknownFormat asserts an unknown
// format string (e.g. "aiff") is rejected before validation.
func TestReplaceMediaURLsStep_InputAudio_UnknownFormat(t *testing.T) {
	step, _ := NewReplaceMediaURLsStep(nil, map[string]any{})
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":        "input_audio",
							"input_audio": map[string]any{"data": "UklGRg==", "format": "aiff"},
						},
					},
				},
			},
		},
	}
	err := step.Execute(context.Background(), reqCtx)
	if err == nil {
		t.Fatal("expected error for unknown input_audio format")
	}
	if !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("expected ErrBadRequest, got %v", err)
	}
}

// TestReplaceMediaURLsStep_InputAudio_ExactlyAtCap encodes exactly cap bytes
// and asserts the step accepts the payload. The base64 length of cap bytes
// equals the size-check bound; a strictly-greater comparison must not
// reject the boundary.
func TestReplaceMediaURLsStep_InputAudio_ExactlyAtCap(t *testing.T) {
	const capMB = 1
	step, _ := NewReplaceMediaURLsStep(nil, map[string]any{"max_download_size": capMB})
	payload := base64.StdEncoding.EncodeToString(make([]byte, capMB*1024*1024))
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":        "input_audio",
							"input_audio": map[string]any{"data": payload, "format": "wav"},
						},
					},
				},
			},
		},
	}
	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("expected exactly-at-cap payload to be accepted, got %v", err)
	}
	if got := len(reqCtx.MultimodalEntries); got != 1 {
		t.Fatalf("expected 1 entry, got %d", got)
	}
}

// TestReplaceMediaURLsStep_InputAudio_OversizedPayload builds an input_audio
// item whose base64 payload alone exceeds 4/3 * max_download_size and asserts
// the step rejects it without attempting to decode.
func TestReplaceMediaURLsStep_InputAudio_OversizedPayload(t *testing.T) {
	// max_download_size in the constructor is given in megabytes.
	step, _ := NewReplaceMediaURLsStep(nil, map[string]any{"max_download_size": 1})
	// 1 MB * 4/3 ~= 1_398_101 base64 chars. Build a slightly larger string.
	oversized := strings.Repeat("A", 2*1024*1024)
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":        "input_audio",
							"input_audio": map[string]any{"data": oversized, "format": "wav"},
						},
					},
				},
			},
		},
	}
	err := step.Execute(context.Background(), reqCtx)
	if err == nil {
		t.Fatal("expected error for oversized input_audio payload")
	}
	if !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("expected ErrBadRequest, got %v", err)
	}
}

// TestReplaceMediaURLsStep_InputAudio_AllowlistUsesCanonicalMIME pins the
// vocabulary an operator has to write. An input_audio part names a format and
// the allowlist names MIME types, so "mp3" is checked as audio/mpeg and an
// allowlist of audio/mp3 alone does not admit it. The rejection has to name
// the format, or the operator cannot connect it back to the config.
func TestReplaceMediaURLsStep_InputAudio_AllowlistUsesCanonicalMIME(t *testing.T) {
	mp3Part := func() map[string]any {
		return map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":        "input_audio",
							"input_audio": map[string]any{"data": "AAAA", "format": "mp3"},
						},
					},
				},
			},
		}
	}

	t.Run("alias_alone_rejects", func(t *testing.T) {
		step := newLoopbackStep(t, map[string]any{
			"download_timeout":            "5s",
			"allowed_audio_content_types": []any{"audio/mp3"},
		})
		err := step.Execute(context.Background(), &pipeline.RequestContext{Body: mp3Part()})
		if err == nil {
			t.Fatal("expected mp3 to be rejected when only audio/mp3 is allowed")
		}
		if !errors.Is(err, pipeline.ErrBadRequest) {
			t.Fatalf("expected ErrBadRequest, got %v", err)
		}
		if !strings.Contains(err.Error(), `"mp3"`) {
			t.Errorf("error should name the format, got %v", err)
		}
		if !strings.Contains(err.Error(), "audio/mpeg") {
			t.Errorf("error should name the MIME the format maps to, got %v", err)
		}
	})

	t.Run("canonical_type_accepts", func(t *testing.T) {
		step := newLoopbackStep(t, map[string]any{
			"download_timeout":            "5s",
			"allowed_audio_content_types": []any{"audio/mpeg"},
		})
		if err := step.Execute(context.Background(), &pipeline.RequestContext{Body: mp3Part()}); err != nil {
			t.Fatalf("expected mp3 accepted when audio/mpeg is allowed, got %v", err)
		}
	})
}

// TestReplaceMediaURLsStep_InputAudio_RejectedBeforeDownloads puts a bad
// input_audio LAST in walker order, behind an audio_url that would
// otherwise be downloaded. The inline checks are local, so they must all
// run before any download starts; otherwise a request that is going to be
// rejected anyway still pays for the bytes. The server counter is the
// assertion: it must never be hit.
func TestReplaceMediaURLsStep_InputAudio_RejectedBeforeDownloads(t *testing.T) {
	var hits atomic.Int32
	audioServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		w.Header().Set("Content-Type", testAudioWAVMIME)
		_, _ = w.Write([]byte("wav-bytes"))
	}))
	defer audioServer.Close()

	step := newLoopbackStep(t, map[string]any{"download_timeout": "5s"})
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":      "audio_url",
							"audio_url": map[string]any{"url": audioServer.URL + "/clip.wav"},
						},
						map[string]any{
							"type":        "input_audio",
							"input_audio": map[string]any{"data": "AAAA", "format": "aiff"},
						},
					},
				},
			},
		},
	}

	err := step.Execute(context.Background(), reqCtx)
	if err == nil || !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("expected unsupported input_audio format rejected, got %v", err)
	}
	if got := hits.Load(); got != 0 {
		t.Fatalf("expected no download before inline validation rejected the request, got %d request(s)", got)
	}
}

// TestReplaceMediaURLsStep_MixedImageAudioVideo runs one request with
// one image URL, one audio URL, and one video URL. All three are
// inlined as data URIs and added to MultimodalEntries in walker order,
// and all three count against max_multimodal_entries.
func TestReplaceMediaURLsStep_MixedImageAudioVideo(t *testing.T) {
	mediaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, ".jpg"):
			w.Header().Set("Content-Type", testImageJPEGMIME)
			_, _ = w.Write([]byte("jpg-bytes"))
		case strings.HasSuffix(r.URL.Path, ".wav"):
			w.Header().Set("Content-Type", testAudioWAVMIME)
			_, _ = w.Write([]byte("wav-bytes"))
		case strings.HasSuffix(r.URL.Path, ".mp4"):
			w.Header().Set("Content-Type", testVideoMP4MIME)
			_, _ = w.Write([]byte("mp4-bytes"))
		default:
			http.NotFound(w, r)
		}
	}))
	defer mediaServer.Close()

	step := newLoopbackStep(t, map[string]any{
		"download_timeout":       "5s",
		"max_multimodal_entries": 3,
	})
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{"type": "image_url", "image_url": map[string]any{"url": mediaServer.URL + "/photo.jpg"}},
						map[string]any{"type": "audio_url", "audio_url": map[string]any{"url": mediaServer.URL + "/clip.wav"}},
						map[string]any{"type": "video_url", "video_url": map[string]any{"url": mediaServer.URL + "/clip.mp4"}},
					},
				},
			},
		},
	}

	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// All three media parts feed into MultimodalEntries. Entries appear
	// in walker order: URL refs first (image, audio, video), then any
	// inline refs (none here).
	if len(reqCtx.MultimodalEntries) != 3 {
		t.Fatalf("expected 3 entries (1 image + 1 audio + 1 video), got %d", len(reqCtx.MultimodalEntries))
	}
	wantModalities := []string{ModalityImage, ModalityAudio, ModalityVideo}
	for i, want := range wantModalities {
		if got := reqCtx.MultimodalEntries[i].Modality; got != want {
			t.Errorf("MultimodalEntries[%d].Modality = %q, want %q", i, got, want)
		}
	}
	// Verify audio and video URLs were rewritten in place.
	content := reqCtx.Body["messages"].([]any)[0].(map[string]any)["content"].([]any)
	audioURL := content[1].(map[string]any)["audio_url"].(map[string]any)["url"].(string)
	videoURL := content[2].(map[string]any)["video_url"].(map[string]any)["url"].(string)
	if !strings.HasPrefix(audioURL, "data:audio/wav;base64,") {
		t.Errorf("audio not inlined: %s", audioURL)
	}
	if !strings.HasPrefix(videoURL, "data:video/mp4;base64,") {
		t.Errorf("video not inlined: %s", videoURL)
	}
}

// TestReplaceMediaURLsStep_MixedAudio_WalkerOrder locks in the walker-order
// invariant for the audio modality, which is the only modality that
// carries both a URL-based variant (audio_url) and an inline variant
// (input_audio). The request has input_audio A first and audio_url B
// second. MultimodalEntries must reflect that order: entries[0] carries
// A's inline payload and entries[1] carries B's downloaded payload. A
// split append (URLs first, inline second) would swap them, and the
// encode / decode steps -- which walk parts in request order -- would
// then pair each audio entry with the wrong content part.
func TestReplaceMediaURLsStep_MixedAudio_WalkerOrder(t *testing.T) {
	audioServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", testAudioWAVMIME)
		_, _ = w.Write([]byte("bytes-of-B"))
	}))
	defer audioServer.Close()

	const inlineData = "SU5MSU5FLUE=" // "INLINE-A" base64
	step := newLoopbackStep(t, map[string]any{"download_timeout": "5s"})
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":        "input_audio",
							"input_audio": map[string]any{"data": inlineData, "format": "wav"},
						},
						map[string]any{
							"type":      "audio_url",
							"audio_url": map[string]any{"url": audioServer.URL + "/clip.wav"},
						},
					},
				},
			},
		},
	}
	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := len(reqCtx.MultimodalEntries); got != 2 {
		t.Fatalf("expected 2 audio entries, got %d", got)
	}
	// The inline part's input_audio.data must be untouched, and the URL
	// slot must have been overwritten in place with the downloaded data
	// URI. Together with the collectMediaParts check below, this
	// verifies that entry 0 corresponds to the inline part and entry 1
	// to the URL part.
	msgs := reqCtx.Body["messages"].([]any)
	inlinePart := msgs[0].(map[string]any)["content"].([]any)[0].(map[string]any)["input_audio"].(map[string]any)
	if got, _ := inlinePart["data"].(string); got != inlineData {
		t.Errorf("input_audio data changed: got %q, want %q", got, inlineData)
	}
	urlPart := msgs[0].(map[string]any)["content"].([]any)[1].(map[string]any)["audio_url"].(map[string]any)
	if got, _ := urlPart["url"].(string); !strings.HasPrefix(got, "data:audio/wav;base64,") {
		t.Errorf("audio_url url = %q, want inlined data URI", got)
	}

	// Downstream alignment: collectMediaParts walks the SAME body in
	// request order. entries[0] (inline) must pair with partsByMod[audio][0]
	// (the input_audio part) and entries[1] (URL) with partsByMod[audio][1]
	// (the audio_url part). Any regression that swaps entries here would
	// break this pairing silently.
	partsByMod := collectMediaParts(reqCtx.Body)
	audioParts := partsByMod[ModalityAudio]
	if got := len(audioParts); got != 2 {
		t.Fatalf("partsByMod[audio] len = %d, want 2", got)
	}
	if _, ok := audioParts[0]["input_audio"].(map[string]any); !ok {
		t.Errorf("audio parts[0] = %+v, want the input_audio part first", audioParts[0])
	}
	if _, ok := audioParts[1]["audio_url"].(map[string]any); !ok {
		t.Errorf("audio parts[1] = %+v, want the audio_url part second", audioParts[1])
	}
}

// TestReplaceMediaURLsStep_MaxEntriesCountsAllModalities pushes max entries
// past the configured cap by combining one image, one audio, and one video.
// three total against a cap of 2. Expected: rejected.
func TestReplaceMediaURLsStep_MaxEntriesCountsAllModalities(t *testing.T) {
	step, _ := NewReplaceMediaURLsStep(nil, map[string]any{"max_multimodal_entries": 2})
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:image/jpeg;base64,/9j/4AAQ"}},
						map[string]any{"type": "audio_url", "audio_url": map[string]any{"url": "data:audio/wav;base64,UklGRg=="}},
						map[string]any{"type": "video_url", "video_url": map[string]any{"url": "data:video/mp4;base64,AAAA"}},
					},
				},
			},
		},
	}
	err := step.Execute(context.Background(), reqCtx)
	if err == nil {
		t.Fatal("expected error when total media parts exceed max_multimodal_entries")
	}
	if !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("expected ErrBadRequest, got %v", err)
	}
}

// ---- Per-modality caps and allowlists --------------------------------------

// TestReplaceMediaURLsStep_MaxVideoDownloadSize_OverridesGlobal shows a
// video payload accepted with max_video_download_size high enough while the
// same body under max_download_size alone is rejected, exercising the
// per-modality override path.
func TestReplaceMediaURLsStep_MaxVideoDownloadSize_OverridesGlobal(t *testing.T) {
	// 2 MB video payload.
	payload := make([]byte, 2*1024*1024)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", testVideoMP4MIME)
		_, _ = w.Write(payload)
	}))
	defer server.Close()

	body := func() *pipeline.RequestContext {
		return &pipeline.RequestContext{Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{"type": "video_url", "video_url": map[string]any{"url": server.URL + "/clip.mp4"}},
					},
				},
			},
		}}
	}

	// Rejected under a 1 MB global cap.
	tight := newLoopbackStep(t, map[string]any{"max_download_size": 1})
	err := tight.Execute(context.Background(), body())
	if err == nil || !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("expected ErrBadRequest under 1 MB global cap, got %v", err)
	}

	// Accepted when max_video_download_size raises the video-only cap.
	loose := newLoopbackStep(t, map[string]any{
		"max_download_size":       1,
		"max_video_download_size": 5,
	})
	if err := loose.Execute(context.Background(), body()); err != nil {
		t.Fatalf("expected acceptance under 5 MB video-specific cap, got %v", err)
	}
}

// TestReplaceMediaURLsStep_MaxAudioDownloadSize_FallsBackToGlobal confirms
// that when no per-modality override is set, audio downloads honor the
// global max_download_size.
func TestReplaceMediaURLsStep_MaxAudioDownloadSize_FallsBackToGlobal(t *testing.T) {
	payload := make([]byte, 2*1024*1024)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", testAudioWAVMIME)
		_, _ = w.Write(payload)
	}))
	defer server.Close()

	step := newLoopbackStep(t, map[string]any{"max_download_size": 1})
	reqCtx := &pipeline.RequestContext{Body: map[string]any{
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{"type": "audio_url", "audio_url": map[string]any{"url": server.URL + "/clip.wav"}},
				},
			},
		},
	}}
	err := step.Execute(context.Background(), reqCtx)
	if err == nil || !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("expected ErrBadRequest when audio has no override and exceeds global cap, got %v", err)
	}
}

// TestReplaceMediaURLsStep_InputAudio_UsesAudioCap asserts inline input_audio
// size validation respects max_audio_download_size, not the global cap.
func TestReplaceMediaURLsStep_InputAudio_UsesAudioCap(t *testing.T) {
	step, _ := NewReplaceMediaURLsStep(nil, map[string]any{
		"max_download_size":       10, // global 10 MB (would allow)
		"max_audio_download_size": 1,  // audio 1 MB (rejects)
	})
	// Base64 length > (4/3) * 1 MB triggers rejection.
	oversized := strings.Repeat("A", 2*1024*1024)
	reqCtx := &pipeline.RequestContext{Body: map[string]any{
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": oversized, "format": "wav"}},
				},
			},
		},
	}}
	err := step.Execute(context.Background(), reqCtx)
	if err == nil || !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("expected ErrBadRequest from audio-cap check, got %v", err)
	}
}

// TestReplaceMediaURLsStep_RejectsInvalidPerModalityCap covers the three
// per-modality caps' validation: non-positive and megabyte-overflow values
// must fail step construction rather than silently disabling the cap.
func TestReplaceMediaURLsStep_RejectsInvalidPerModalityCap(t *testing.T) {
	tooLarge := (math.MaxInt-1)/config.BytesPerMB + 1
	for _, tc := range []struct {
		name  string
		param map[string]any
	}{
		{"max_image_download_size zero", map[string]any{"max_image_download_size": 0}},
		{"max_audio_download_size negative", map[string]any{"max_audio_download_size": -1}},
		{"max_video_download_size overflow", map[string]any{"max_video_download_size": tooLarge}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := NewReplaceMediaURLsStep(nil, tc.param); err == nil {
				t.Fatalf("expected construction error for %s", tc.name)
			}
		})
	}
}

// TestReplaceMediaURLsStep_AllowedAudioContentTypes_Overrides swaps in a
// narrower audio allowlist ({audio/wav}) and asserts (a) audio/wav passes and
// (b) audio/mpeg, which the default set allows, is rejected.
func TestReplaceMediaURLsStep_AllowedAudioContentTypes_Overrides(t *testing.T) {
	step, _ := NewReplaceMediaURLsStep(nil, map[string]any{
		"allowed_audio_content_types": []any{testAudioWAVMIME},
	})

	accept := &pipeline.RequestContext{Body: map[string]any{
		"messages": []any{
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "audio_url", "audio_url": map[string]any{"url": "data:audio/wav;base64,UklGRg=="}},
			}},
		},
	}}
	if err := step.Execute(context.Background(), accept); err != nil {
		t.Fatalf("expected audio/wav accepted under override, got %v", err)
	}

	reject := &pipeline.RequestContext{Body: map[string]any{
		"messages": []any{
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "audio_url", "audio_url": map[string]any{"url": "data:audio/mpeg;base64,AAAA"}},
			}},
		},
	}}
	err := step.Execute(context.Background(), reject)
	if err == nil || !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("expected audio/mpeg rejected under audio/wav-only override, got %v", err)
	}
}

// TestReplaceMediaURLsStep_AllowedVideoContentTypes_Overrides mirrors the
// image and audio cases for the video allowlist: narrowing to
// {video/mp4} rejects the default-allowed video/webm.
func TestReplaceMediaURLsStep_AllowedVideoContentTypes_Overrides(t *testing.T) {
	step, _ := NewReplaceMediaURLsStep(nil, map[string]any{
		"allowed_video_content_types": []any{testVideoMP4MIME},
	})
	reject := &pipeline.RequestContext{Body: map[string]any{
		"messages": []any{
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "video_url", "video_url": map[string]any{"url": "data:video/webm;base64,GkXf"}},
			}},
		},
	}}
	err := step.Execute(context.Background(), reject)
	if err == nil || !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("expected video/webm rejected under video/mp4-only override, got %v", err)
	}
}

// TestReplaceMediaURLsStep_MaxImageDownloadSize_OverridesGlobal mirrors the
// video case for max_image_download_size: a payload rejected under a
// 1 MB global cap is accepted when the image-only cap is raised.
func TestReplaceMediaURLsStep_MaxImageDownloadSize_OverridesGlobal(t *testing.T) {
	// 2 MB image payload.
	payload := make([]byte, 2*1024*1024)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", testImagePNGMIME)
		_, _ = w.Write(payload)
	}))
	defer server.Close()

	body := func() *pipeline.RequestContext {
		return &pipeline.RequestContext{Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{"type": "image_url", "image_url": map[string]any{"url": server.URL + "/photo.png"}},
					},
				},
			},
		}}
	}

	// Rejected under a 1 MB global cap.
	tight := newLoopbackStep(t, map[string]any{"max_download_size": 1})
	err := tight.Execute(context.Background(), body())
	if err == nil || !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("expected ErrBadRequest under 1 MB global cap, got %v", err)
	}

	// Accepted when max_image_download_size raises the image-only cap.
	loose := newLoopbackStep(t, map[string]any{
		"max_download_size":       1,
		"max_image_download_size": 5,
	})
	if err := loose.Execute(context.Background(), body()); err != nil {
		t.Fatalf("expected acceptance under 5 MB image-specific cap, got %v", err)
	}
}

// TestReplaceMediaURLsStep_AllowedImageContentTypes_Overrides mirrors the
// audio case for the image allowlist: narrowing to {image/png} rejects the
// default-allowed image/jpeg.
func TestReplaceMediaURLsStep_AllowedImageContentTypes_Overrides(t *testing.T) {
	step, _ := NewReplaceMediaURLsStep(nil, map[string]any{
		"allowed_image_content_types": []any{testImagePNGMIME},
	})
	reject := &pipeline.RequestContext{Body: map[string]any{
		"messages": []any{
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:image/jpeg;base64,/9j/4AAQ"}},
			}},
		},
	}}
	err := step.Execute(context.Background(), reject)
	if err == nil || !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("expected image/jpeg rejected under image/png-only override, got %v", err)
	}
}

// TestReplaceMediaURLsStep_AllowedContentTypes_DefaultsWhenUnset confirms
// that when no per-modality allowlist is configured, the built-in defaults
// apply, audio/mpeg (a default entry) is accepted.
func TestReplaceMediaURLsStep_AllowedContentTypes_DefaultsWhenUnset(t *testing.T) {
	step, _ := NewReplaceMediaURLsStep(nil, map[string]any{})
	reqCtx := &pipeline.RequestContext{Body: map[string]any{
		"messages": []any{
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "audio_url", "audio_url": map[string]any{"url": "data:audio/mpeg;base64,AAAA"}},
			}},
		},
	}}
	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("expected default audio allowlist to accept audio/mpeg, got %v", err)
	}
}

// TestReplaceMediaURLsStep_RejectsNonStringAllowedContentType fails
// construction when a per-modality allowlist entry is not a string. The
// alternative (silently dropping the bad entry) would be a security
// downgrade, an operator's intent to lock down the allowlist is lost.
func TestReplaceMediaURLsStep_RejectsNonStringAllowedContentType(t *testing.T) {
	_, err := NewReplaceMediaURLsStep(nil, map[string]any{
		"allowed_audio_content_types": []any{testAudioWAVMIME, 42},
	})
	if err == nil {
		t.Fatal("expected construction error for non-string allowlist entry")
	}
}

// TestReplaceMediaURLsStep_RejectsNonListAllowedContentTypes fails
// construction when a per-modality allowlist is set to a non-list value.
func TestReplaceMediaURLsStep_RejectsNonListAllowedContentTypes(t *testing.T) {
	_, err := NewReplaceMediaURLsStep(nil, map[string]any{
		"allowed_video_content_types": testVideoMP4MIME,
	})
	if err == nil {
		t.Fatal("expected construction error for non-list allowlist value")
	}
}

// TestReplaceMediaURLsStep_RejectsNullAllowedContentTypes fails construction
// when a per-modality allowlist key is present with a null value, the shape a
// template produces when its variable is unset. Silently treating it as absent
// is worse than an empty list: for image it also leaves the download-path
// Content-Type check off, so the operator gets no enforcement from a line they
// wrote to add some. allowed_domains rejects a null value the same way, and the
// image case is asserted below because it is the one with two behaviors
// riding on the param.
func TestReplaceMediaURLsStep_RejectsNullAllowedContentTypes(t *testing.T) {
	for _, key := range []string{
		"allowed_image_content_types",
		"allowed_audio_content_types",
		"allowed_video_content_types",
	} {
		t.Run(key, func(t *testing.T) {
			if _, err := NewReplaceMediaURLsStep(nil, map[string]any{key: nil}); err == nil {
				t.Fatalf("expected construction error for %s with a null value", key)
			}
		})
	}
}

// TestReplaceMediaURLsStep_ExplicitImageAllowlistEnablesDownloadCheck pins the
// pairing the null case above protects: setting the image allowlist at all,
// empty list included, turns on the download-path Content-Type check, while
// leaving it unset keeps the permissive path.
func TestReplaceMediaURLsStep_ExplicitImageAllowlistEnablesDownloadCheck(t *testing.T) {
	for _, tc := range []struct {
		name   string
		params map[string]any
		want   bool
	}{
		{"unset", map[string]any{}, false},
		{"empty list", map[string]any{"allowed_image_content_types": []any{}}, true},
		{"explicit list", map[string]any{"allowed_image_content_types": []any{testImagePNGMIME}}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			step, err := NewReplaceMediaURLsStep(nil, tc.params)
			if err != nil {
				t.Fatal(err)
			}
			got := step.(*ReplaceMediaURLsStep).enforceDownloadContentType(ModalityImage)
			if got != tc.want {
				t.Fatalf("enforceDownloadContentType(image) = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestReplaceMediaURLsStep_AllowedContentTypes_EmptyMeansUnrestricted
// asserts that setting allowed_<modality>_content_types to an empty list
// disables the per-modality allowlist for that modality, so any type is
// accepted. This mirrors the "empty means unrestricted" convention
// allowed_domains uses.
func TestReplaceMediaURLsStep_AllowedContentTypes_EmptyMeansUnrestricted(t *testing.T) {
	step, err := NewReplaceMediaURLsStep(nil, map[string]any{
		"allowed_video_content_types": []any{},
	})
	if err != nil {
		t.Fatal(err)
	}
	// A type outside the built-in video allowlist (application/octet-stream)
	// would normally be rejected. Under the unrestricted override it must
	// be accepted.
	reqCtx := &pipeline.RequestContext{Body: map[string]any{
		"messages": []any{
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "video_url", "video_url": map[string]any{"url": "data:application/octet-stream;base64,AAAA"}},
			}},
		},
	}}
	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("expected acceptance under empty video allowlist, got %v", err)
	}
}

// TestParsePerModalityContentTypes_DoesNotAliasDefaults confirms that a
// caller mutating the returned per-modality set cannot reach into the
// package-level defaultAllowedContentTypesByModality: adding an entry to
// the returned image set must not appear in a fresh call. This protects
// every subsequent ReplaceMediaURLsStep from picking up leaked overrides.
func TestParsePerModalityContentTypes_DoesNotAliasDefaults(t *testing.T) {
	first, _, err := parsePerModalityContentTypes(nil)
	if err != nil {
		t.Fatalf("parsePerModalityContentTypes returned error: %v", err)
	}
	const poison = "application/x-poison"
	first[ModalityImage][poison] = struct{}{}

	second, _, err := parsePerModalityContentTypes(nil)
	if err != nil {
		t.Fatalf("parsePerModalityContentTypes returned error: %v", err)
	}
	if _, leaked := second[ModalityImage][poison]; leaked {
		t.Fatal("mutation of first result reached defaults and leaked into second result")
	}
	if _, leaked := defaultAllowedContentTypesByModality[ModalityImage][poison]; leaked {
		t.Fatal("mutation of first result reached package-level defaults")
	}
}
