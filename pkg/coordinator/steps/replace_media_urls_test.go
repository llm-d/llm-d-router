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
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"

	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// MIME literals reused across tests. Extracted so goconst does not flag
// their repetition and so a rename or typo lands in one place.
// image/jpeg is not declared here: conditional_decode_test.go already names
// that value testImageJPEGContentType, and this package shares one test scope.
const (
	testAudioWAVMIME = "audio/wav"
	testImagePNGMIME = "image/png"
	testVideoMP4MIME = "video/mp4"
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
		w.Header().Set("Content-Type", testImageJPEGContentType)
		_, _ = w.Write([]byte("jpeg-bytes"))
	}))
	defer imageServer.Close()

	step := newLoopbackStep(t, map[string]any{"download_timeout": "5s"})

	reqCtx := &pipeline.RequestContext{
		OriginalPath: reqcommon.PathChatCompletions,
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
	if !strings.HasPrefix(url, "data:image/jpeg;base64,") {
		t.Fatalf("expected data URI, got %s", url)
	}
}

func TestReplaceMediaURLsStep_Responses_DownloadsAndInlines(t *testing.T) {
	imageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", testImageJPEGContentType)
		_, _ = w.Write([]byte("jpeg-bytes"))
	}))
	defer imageServer.Close()

	step := newLoopbackStep(t, map[string]any{"download_timeout": "5s"})

	reqCtx := &pipeline.RequestContext{
		OriginalPath: reqcommon.PathResponses,
		Body: map[string]any{
			"input": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{"type": "input_text", "text": "describe this"},
						map[string]any{
							"type":      "input_image",
							"image_url": imageServer.URL + "/photo.jpg",
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
	// The downloaded bytes stay in the body as a data URI (asserted below)
	// rather than on the entry, so the modality is all the entry carries here.
	if got := reqCtx.MultimodalEntries[0].Modality; got != reqcommon.ModalityImage {
		t.Fatalf("entry modality = %q, want %q", got, reqcommon.ModalityImage)
	}

	input := reqCtx.Body["input"].([]any)
	content := input[0].(map[string]any)["content"].([]any)
	url := content[1].(map[string]any)["image_url"].(string)
	if !strings.HasPrefix(url, "data:image/jpeg;base64,") {
		t.Fatalf("expected data URI, got %s", url)
	}
}

// See collectMediaRefs' doc comment for why a file_id-referenced image is
// rejected rather than skipped.
func TestReplaceMediaURLsStep_Responses_RejectsFileIDImage(t *testing.T) {
	step, _ := NewReplaceMediaURLsStep(nil, map[string]any{})

	reqCtx := &pipeline.RequestContext{
		OriginalPath: reqcommon.PathResponses,
		Body: map[string]any{
			"input": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{"type": "input_text", "text": "describe this"},
						map[string]any{"type": "input_image", "file_id": "file-abc123"},
					},
				},
			},
		},
	}

	err := step.Execute(context.Background(), reqCtx)
	if err == nil {
		t.Fatal("expected error for input_image part with no image_url string")
	}
	if !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("expected ErrBadRequest, got %v", err)
	}
	if len(reqCtx.MultimodalEntries) != 0 {
		t.Fatalf("expected no entries populated on rejection, got %d", len(reqCtx.MultimodalEntries))
	}
}

// A chat-completions request carrying a stray top-level "input" array must not
// have that field's image processed.
func TestReplaceMediaURLsStep_IgnoresStrayInputOnChatCompletions(t *testing.T) {
	var hits atomic.Int32
	imageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		w.Header().Set("Content-Type", "image/jpeg")
		_, _ = w.Write([]byte("jpeg-bytes"))
	}))
	defer imageServer.Close()

	step := newLoopbackStep(t, map[string]any{})

	strayImageURL := imageServer.URL + "/stray.jpg"
	reqCtx := &pipeline.RequestContext{
		OriginalPath: reqcommon.PathChatCompletions,
		Body: map[string]any{
			"messages": []any{
				map[string]any{"role": "user", "content": "just text"},
			},
			"input": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{"type": "input_image", "image_url": strayImageURL},
					},
				},
			},
		},
	}

	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if hits.Load() != 0 {
		t.Fatalf("expected the stray input array's image to never be fetched, got %d hits", hits.Load())
	}
	if len(reqCtx.MultimodalEntries) != 0 {
		t.Fatalf("expected 0 multimodal entries, got %d", len(reqCtx.MultimodalEntries))
	}
	input := reqCtx.Body["input"].([]any)
	part := input[0].(map[string]any)["content"].([]any)[0].(map[string]any)
	if part["image_url"] != strayImageURL {
		t.Fatalf("expected stray input's image_url left untouched, got %v", part["image_url"])
	}
}

func TestReplaceMediaURLsStep_NoImages(t *testing.T) {
	step, _ := NewReplaceMediaURLsStep(nil, map[string]any{})

	reqCtx := &pipeline.RequestContext{
		OriginalPath: reqcommon.PathChatCompletions,
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
		OriginalPath: reqcommon.PathChatCompletions,
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
		OriginalPath: reqcommon.PathChatCompletions,
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

// An uppercase data: scheme, which RFC 3986 allows, must be recognized as
// inline data and left alone; a case-sensitive prefix check would send it down
// the download path, where the scheme guard rejects it for not being http(s).
// The url is asserted unchanged, casing included: the step does not rewrite an
// inline payload, and vLLM lowercases the scheme through urlparse, so the
// client's casing reaches the backend and still parses.
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
// whether the part came from a download or an inline data: URI. The encode
// fanout and decode.injectUUIDs pair the Nth entry of a modality with its Nth
// part (see collectMediaParts), so drift here attaches the wrong bytes to
// the wrong entry. Asserted in both source orderings.
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
				OriginalPath: reqcommon.PathChatCompletions,
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
				if got := reqCtx.MultimodalEntries[i].Modality; got != reqcommon.ModalityImage {
					t.Errorf("entry[%d].Modality = %q, want %q", i, got, reqcommon.ModalityImage)
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

// Execute and collectMediaParts must agree on exactly which parts count.
// Execute fixes the entry order the encode fanout and decode.injectUUIDs index
// into, and those two walk with collectMediaParts, so a part counted by one and
// not the other shifts the pairing. Asserted end to end over every recognized
// part type rather than by trusting the shared walk.
func TestReplaceMediaURLsStep_EntriesMatchTheMediaPartWalk(t *testing.T) {
	parts := []any{
		map[string]any{"type": "text", "text": "hi"},
		map[string]any{"type": reqcommon.PartTypeImageURL, reqcommon.PartTypeImageURL: map[string]any{"url": "data:image/png;base64,aGk="}},
		map[string]any{"type": reqcommon.PartTypeAudioURL, reqcommon.PartTypeAudioURL: map[string]any{"url": "data:audio/wav;base64,aGk="}},
		// input_audio is inline, carrying its payload under data, not url.
		map[string]any{"type": reqcommon.PartTypeInputAudio, reqcommon.PartTypeInputAudio: map[string]any{"data": "aGk=", "format": "wav"}},
		map[string]any{"type": reqcommon.PartTypeVideoURL, reqcommon.PartTypeVideoURL: map[string]any{"url": "data:video/mp4;base64,aGk="}},
		// A chat request names an image either way (see reqcommon.PartModality).
		map[string]any{"type": reqcommon.PartTypeInputImage, reqcommon.FieldImageURL: "data:image/png;base64,aGk="},
		// Recognized by neither: passed through untouched.
		map[string]any{"type": "image_embeds", "image_embeds": map[string]any{}},
	}

	items := []any{map[string]any{"role": "user", "content": parts}}

	// What the shared walk says the answer is, over the same body Execute reads.
	walked := collectMediaParts(items, reqcommon.APITypeChatCompletions)
	wantModalities := make([]reqcommon.Modality, 0, len(walked))
	for _, media := range walked {
		wantModalities = append(wantModalities, media.modality)
	}
	// Guard the guard: a fixture typo that made every part unrecognized would
	// otherwise let this test pass with both sides at zero.
	if len(wantModalities) != 5 {
		t.Fatalf("fixture drift: the walk collected %d parts, want 5 (%v)", len(wantModalities), wantModalities)
	}

	step, _ := NewReplaceMediaURLsStep(nil, map[string]any{})
	reqCtx := &pipeline.RequestContext{
		OriginalPath: reqcommon.PathChatCompletions,
		Body:         map[string]any{"messages": items},
	}
	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	gotModalities := make([]reqcommon.Modality, 0, len(reqCtx.MultimodalEntries))
	for _, e := range reqCtx.MultimodalEntries {
		gotModalities = append(gotModalities, e.Modality)
	}
	if !slices.Equal(gotModalities, wantModalities) {
		t.Errorf("Execute produced modalities %v, the walk expects %v", gotModalities, wantModalities)
	}
}

// A media part the step cannot resolve fails the request, for every recognized
// part type and every way of being unusable. Skipping one instead would shift
// every later part of its modality onto another part's hash downstream, which
// is why collectMediaRefs rejects rather than skips. Each case puts a
// well-formed part of the same modality after the broken one, the shape that
// would be mispaired.
func TestReplaceMediaURLsStep_RejectsUnusableMediaPart(t *testing.T) {
	for _, tc := range []struct {
		name   string
		broken map[string]any
		good   map[string]any
	}{
		{
			name:   "image_url inner missing",
			broken: map[string]any{"type": reqcommon.PartTypeImageURL},
			good:   map[string]any{"type": reqcommon.PartTypeImageURL, reqcommon.PartTypeImageURL: map[string]any{"url": "data:image/png;base64,aGk="}},
		},
		{
			name:   "image_url inner not an object",
			broken: map[string]any{"type": reqcommon.PartTypeImageURL, reqcommon.PartTypeImageURL: "not-an-object"},
			good:   map[string]any{"type": reqcommon.PartTypeImageURL, reqcommon.PartTypeImageURL: map[string]any{"url": "data:image/png;base64,aGk="}},
		},
		{
			name:   "image_url url absent",
			broken: map[string]any{"type": reqcommon.PartTypeImageURL, reqcommon.PartTypeImageURL: map[string]any{}},
			good:   map[string]any{"type": reqcommon.PartTypeImageURL, reqcommon.PartTypeImageURL: map[string]any{"url": "data:image/png;base64,aGk="}},
		},
		{
			name:   "image_url url not a string",
			broken: map[string]any{"type": reqcommon.PartTypeImageURL, reqcommon.PartTypeImageURL: map[string]any{"url": 123}},
			good:   map[string]any{"type": reqcommon.PartTypeImageURL, reqcommon.PartTypeImageURL: map[string]any{"url": "data:image/png;base64,aGk="}},
		},
		{
			name:   "image_url url empty",
			broken: map[string]any{"type": reqcommon.PartTypeImageURL, reqcommon.PartTypeImageURL: map[string]any{"url": ""}},
			good:   map[string]any{"type": reqcommon.PartTypeImageURL, reqcommon.PartTypeImageURL: map[string]any{"url": "data:image/png;base64,aGk="}},
		},
		{
			name:   "input_image url not a string",
			broken: map[string]any{"type": reqcommon.PartTypeInputImage, reqcommon.FieldImageURL: map[string]any{"url": "x"}},
			good:   map[string]any{"type": reqcommon.PartTypeInputImage, reqcommon.FieldImageURL: "data:image/png;base64,aGk="},
		},
		{
			name:   "audio_url inner not an object",
			broken: map[string]any{"type": reqcommon.PartTypeAudioURL, reqcommon.PartTypeAudioURL: "not-an-object"},
			good:   map[string]any{"type": reqcommon.PartTypeAudioURL, reqcommon.PartTypeAudioURL: map[string]any{"url": "data:audio/wav;base64,aGk="}},
		},
		{
			name:   "video_url url absent",
			broken: map[string]any{"type": reqcommon.PartTypeVideoURL, reqcommon.PartTypeVideoURL: map[string]any{}},
			good:   map[string]any{"type": reqcommon.PartTypeVideoURL, reqcommon.PartTypeVideoURL: map[string]any{"url": "data:video/mp4;base64,aGk="}},
		},
		{
			name:   "input_audio inner missing",
			broken: map[string]any{"type": reqcommon.PartTypeInputAudio},
			good:   map[string]any{"type": reqcommon.PartTypeInputAudio, reqcommon.PartTypeInputAudio: map[string]any{"data": "aGk=", "format": "wav"}},
		},
		{
			name:   "input_audio data empty",
			broken: map[string]any{"type": reqcommon.PartTypeInputAudio, reqcommon.PartTypeInputAudio: map[string]any{"data": "", "format": "wav"}},
			good:   map[string]any{"type": reqcommon.PartTypeInputAudio, reqcommon.PartTypeInputAudio: map[string]any{"data": "aGk=", "format": "wav"}},
		},
		{
			name:   "input_audio data absent",
			broken: map[string]any{"type": reqcommon.PartTypeInputAudio, reqcommon.PartTypeInputAudio: map[string]any{"format": "wav"}},
			good:   map[string]any{"type": reqcommon.PartTypeInputAudio, reqcommon.PartTypeInputAudio: map[string]any{"data": "aGk=", "format": "wav"}},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			step, _ := NewReplaceMediaURLsStep(nil, map[string]any{})
			reqCtx := &pipeline.RequestContext{
				OriginalPath: reqcommon.PathChatCompletions,
				Body: map[string]any{
					"messages": []any{
						map[string]any{"role": "user", "content": []any{tc.broken, tc.good}},
					},
				},
			}

			err := step.Execute(context.Background(), reqCtx)
			if err == nil {
				t.Fatal("expected the unusable media part to fail the request")
			}
			if !errors.Is(err, pipeline.ErrBadRequest) {
				t.Fatalf("expected ErrBadRequest, got %v", err)
			}
			if len(reqCtx.MultimodalEntries) != 0 {
				t.Fatalf("expected no entries on rejection, got %d", len(reqCtx.MultimodalEntries))
			}
		})
	}
}

// The inline size check measures the payload, so it is exact at the cap for
// every size, including the non-multiples of 3 where padding decides the
// answer. This is the property the old encoded-length bound could not hold:
// it compared against 4*ceil(cap/3), which for a cap that is not a multiple
// of 3 sits up to 2 bytes above the cap.
func TestInlineSizeExceeded_ExactAtEveryBoundary(t *testing.T) {
	for _, sizeCap := range []int64{1, 2, 3, 4, 5, 6, 100, 1023, 1024, 1048576} {
		step := &ReplaceMediaURLsStep{maxDownloadSize: sizeCap}
		for _, payloadBytes := range []int64{sizeCap - 1, sizeCap, sizeCap + 1, sizeCap + 2} {
			if payloadBytes < 0 {
				continue
			}
			b64 := base64.StdEncoding.EncodeToString(make([]byte, payloadBytes))
			want := payloadBytes > sizeCap
			if got := step.inlineSizeExceeded(b64, reqcommon.ModalityAudio); got != want {
				t.Errorf("cap %d, payload %d bytes: exceeded = %v, want %v",
					sizeCap, payloadBytes, got, want)
			}
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
							"type":                       reqcommon.PartTypeInputAudio,
							reqcommon.PartTypeInputAudio: map[string]any{"data": "aGk=", "format": "wav"},
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

// encodeDataURI streams into a strings.Builder instead of encoding then
// concatenating, so it must still agree with the obvious implementation byte
// for byte. Lengths 0-4 cover every padding case, where a missed Close() flush
// would show up.
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
			wantType:    testImageJPEGContentType,
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
		OriginalPath: reqcommon.PathChatCompletions,
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
		OriginalPath: reqcommon.PathChatCompletions,
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
		OriginalPath: reqcommon.PathChatCompletions,
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
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			step, _ := NewReplaceMediaURLsStep(nil, map[string]any{})
			reqCtx := &pipeline.RequestContext{OriginalPath: reqcommon.PathChatCompletions, Body: tt.body}
			if err := step.Execute(context.Background(), reqCtx); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(reqCtx.MultimodalEntries) != 0 {
				t.Fatalf("expected 0 multimodal entries, got %d", len(reqCtx.MultimodalEntries))
			}
		})
	}
}

// TestReplaceMediaURLsStep_RejectsMalformedImageURLPart locks in that
// collectMediaRefs rejects a malformed image_url part rather than silently
// skipping it; see its doc comment for why.
func TestReplaceMediaURLsStep_RejectsMalformedImageURLPart(t *testing.T) {
	tests := []struct {
		name string
		body map[string]any
	}{
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
			reqCtx := &pipeline.RequestContext{OriginalPath: reqcommon.PathChatCompletions, Body: tt.body}
			err := step.Execute(context.Background(), reqCtx)
			if err == nil {
				t.Fatal("expected error for malformed image_url part")
			}
			if !errors.Is(err, pipeline.ErrBadRequest) {
				t.Fatalf("expected ErrBadRequest, got %v", err)
			}
			if len(reqCtx.MultimodalEntries) != 0 {
				t.Fatalf("expected no entries populated on rejection, got %d", len(reqCtx.MultimodalEntries))
			}
		})
	}
}

// TestReplaceMediaURLsStep_RejectsMixedMalformedAndValidImageParts covers the
// concrete failure collectMediaRefs's doc comment describes:
// skipping the malformed part instead of rejecting it would leave the valid
// image's hash misassigned to the malformed part.
func TestReplaceMediaURLsStep_RejectsMixedMalformedAndValidImageParts(t *testing.T) {
	step, _ := NewReplaceMediaURLsStep(nil, map[string]any{})
	reqCtx := &pipeline.RequestContext{
		OriginalPath: reqcommon.PathChatCompletions,
		Body: map[string]any{
			"messages": []any{
				map[string]any{"role": "user", "content": []any{
					map[string]any{"type": "image_url", "image_url": "http://bad/a.png"},
					map[string]any{"type": "image_url", "image_url": map[string]any{"url": "http://good/b.png"}},
				}},
			},
		},
	}

	err := step.Execute(context.Background(), reqCtx)
	if err == nil {
		t.Fatal("expected error for the malformed part")
	}
	if !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("expected ErrBadRequest, got %v", err)
	}
	if len(reqCtx.MultimodalEntries) != 0 {
		t.Fatalf("expected no entries populated on rejection, got %d", len(reqCtx.MultimodalEntries))
	}
}

func TestReplaceMediaURLsStep_InvalidDataURI(t *testing.T) {
	step, _ := NewReplaceMediaURLsStep(nil, map[string]any{})
	reqCtx := &pipeline.RequestContext{
		OriginalPath: reqcommon.PathChatCompletions,
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
		OriginalPath: reqcommon.PathChatCompletions,
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
		OriginalPath: reqcommon.PathChatCompletions,
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
	_, _, err := rmu.download(context.Background(), "http://\x7f/control-char", reqcommon.ModalityImage)
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
		OriginalPath: reqcommon.PathChatCompletions,
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

	_, _, err := rmu.download(context.Background(), imageServer.URL+"/big.png", reqcommon.ModalityImage)
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
		OriginalPath: reqcommon.PathChatCompletions,
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
		OriginalPath: reqcommon.PathChatCompletions,
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

	_, _, err := rmu.download(context.Background(), imageServer.URL+"/truncated", reqcommon.ModalityImage)
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
		_, _, err := rmu.download(context.Background(), raw, reqcommon.ModalityImage)
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
	_, _, err := rmu.download(context.Background(), "http://169.254.169.254/latest/meta-data/", reqcommon.ModalityImage)
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
	_, _, err := rmu.download(context.Background(), redirector.URL+"/start", reqcommon.ModalityImage)
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
	_, _, err = step.download(context.Background(), "http://localhost:"+port+"/x", reqcommon.ModalityImage)
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
	if _, _, err := allowed.download(context.Background(), server.URL+"/ok.png", reqcommon.ModalityImage); err != nil {
		t.Fatalf("listed host must be fetchable: %v", err)
	}

	denied := newLoopbackStep(t, map[string]any{"allowed_domains": []any{"images.example.com"}})
	_, _, err = denied.download(context.Background(), server.URL+"/ok.png", reqcommon.ModalityImage)
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
		OriginalPath: reqcommon.PathChatCompletions,
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
		OriginalPath: reqcommon.PathChatCompletions,
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
		OriginalPath: reqcommon.PathChatCompletions,
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

// An audio_url is fetched, size-capped, MIME-checked, inlined as a data URI,
// and added to MultimodalEntries as one audio entry.
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
	if reqCtx.MultimodalEntries[0].Modality != reqcommon.ModalityAudio {
		t.Fatalf("expected Modality=%q, got %q", reqcommon.ModalityAudio, reqCtx.MultimodalEntries[0].Modality)
	}
	msgs := reqCtx.Body["messages"].([]any)
	inner := msgs[0].(map[string]any)["content"].([]any)[0].(map[string]any)["audio_url"].(map[string]any)
	url := inner["url"].(string)
	if !strings.HasPrefix(url, "data:audio/wav;base64,") {
		t.Fatalf("expected inlined data URI, got %s", url)
	}
}

// A Content-Type carrying only parameters and no type ("; charset=utf-8") must
// be stripped, noticed as empty, and replaced with the default type so the
// rewritten URL stays well-formed. With the fallback before the strip, the URL
// would come out as data:;base64,... and be unusable.
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

// What a Content-Type header or data URI media type is reduced to before it
// reaches the allowlist or the emitted URI. The comma cases are the ones that
// matter for that URI: RFC 2397 ends the metadata at the first comma, so a type
// keeping one would move the payload boundary.
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

// An origin whose Content-Type holds more than one value: the comma must not
// reach the rewritten data URI, since a reader splitting on the first comma
// would take "image/png" as the whole metadata and " image/png;base64,..." as
// the payload, failing the base64 decode.
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

// An audio_url / video_url whose origin returns a Content-Type with MIME
// parameters (";codecs=...", ";charset=...") is accepted, and the parameters
// are stripped at the download boundary so the rewritten data URI carries a
// bare MIME. The codecs case embeds a comma inside a quoted parameter value,
// which is what breaks parseDataURI when the raw header value flows through.
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
			// The rewritten URL must round-trip through parseDataURI: with a
			// comma inside a parameter value, the pre-normalization code put
			// the first comma inside the codecs list and parseDataURI failed.
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

// An audio_url whose origin serves a non-audio Content-Type (text/html) is
// rejected as ErrBadRequest, closing an SSRF-style widening where a caller
// uses an audio_url slot to smuggle text or HTML.
// An object store that serves its objects unlabeled must not fail the download
// check. S3, GCS and presigned URLs return application/octet-stream for
// anything whose type was not set at upload, and an absent header lands on the
// same value, so the strict form rejected ordinary audio and video hosting. It
// bought nothing: an origin that is lying sends audio/wav just as easily, so
// the only thing refusing octet-stream stopped was honest hosting -- and both
// documented escapes turn the check off wholesale, losing the text/html case
// below with it.
func TestReplaceMediaURLsStep_Download_AcceptsUnlabeledOrigin(t *testing.T) {
	for _, tc := range []struct {
		name        string
		contentType string // "" sends no header at all
	}{
		{"declared octet-stream", defaultContentType},
		{"no Content-Type header", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, media := range []struct {
				partType string
				file     string
			}{
				{reqcommon.PartTypeAudioURL, "/clip.wav"},
				{reqcommon.PartTypeVideoURL, "/clip.mp4"},
			} {
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					if tc.contentType != "" {
						w.Header().Set("Content-Type", tc.contentType)
					} else {
						// Go sniffs a type for an unset header; an empty value
						// is how a handler sends none.
						w.Header()["Content-Type"] = nil
					}
					_, _ = w.Write([]byte("payload"))
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
										"type":         media.partType,
										media.partType: map[string]any{"url": server.URL + media.file},
									},
								},
							},
						},
					},
				}
				if err := step.Execute(context.Background(), reqCtx); err != nil {
					t.Errorf("%s: expected an unlabeled origin to be accepted, got %v", media.partType, err)
				}
			}
		})
	}
}

// The allowance is for the built-in list only. An operator who writes the param
// has said exactly what to accept, and coordinator.yaml tells them to add
// application/octet-stream when they want unlabeled origins too -- so honoring
// the list literally is what keeps that line meaningful.
func TestReplaceMediaURLsStep_Download_ExplicitListExcludesUnlabeled(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", defaultContentType)
		_, _ = w.Write([]byte("payload"))
	}))
	defer server.Close()

	for _, tc := range []struct {
		name      string
		allowed   []any
		wantError bool
	}{
		{"locked down to wav", []any{testAudioWAVMIME}, true},
		{"octet-stream added back", []any{testAudioWAVMIME, defaultContentType}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			step := newLoopbackStep(t, map[string]any{
				"download_timeout":            "5s",
				"allowed_audio_content_types": tc.allowed,
			})
			reqCtx := &pipeline.RequestContext{
				Body: map[string]any{
					"messages": []any{
						map[string]any{
							"role": "user",
							"content": []any{
								map[string]any{
									"type":      reqcommon.PartTypeAudioURL,
									"audio_url": map[string]any{"url": server.URL + "/clip.wav"},
								},
							},
						},
					},
				},
			}
			err := step.Execute(context.Background(), reqCtx)
			if tc.wantError && err == nil {
				t.Error("expected an explicit allowlist to exclude an unlabeled origin")
			}
			if !tc.wantError && err != nil {
				t.Errorf("expected octet-stream to be accepted once listed, got %v", err)
			}
		})
	}
}

// A data URI gets no such allowance: there the client wrote the media type, so
// an unlabeled one is a request to fix rather than an origin to tolerate.
func TestReplaceMediaURLsStep_DataURI_StillRejectsUnlabeled(t *testing.T) {
	step, err := NewReplaceMediaURLsStep(nil, map[string]any{})
	if err != nil {
		t.Fatal(err)
	}
	reqCtx := &pipeline.RequestContext{
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{
							"type":      reqcommon.PartTypeAudioURL,
							"audio_url": map[string]any{"url": "data:" + defaultContentType + ";base64,aGk="},
						},
					},
				},
			},
		},
	}
	if err := step.Execute(context.Background(), reqCtx); err == nil {
		t.Fatal("expected an unlabeled audio data URI to stay rejected")
	}
}

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

// The audio case above, for video_url served with a non-video Content-Type.
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
// Read returns EOF immediately, so a caller reaching the body gets an empty
// payload rather than blocking.
type readTrackingBody struct {
	read atomic.Bool
}

func (b *readTrackingBody) Read([]byte) (int, error) {
	b.read.Store(true)
	return 0, io.EOF
}

func (b *readTrackingBody) Close() error { return nil }

// Pins the order of the two download-path checks: an audio origin serving a
// type the allowlist rejects is turned away on its headers, body left unread.
// Checking after the read rejects the same request, but only once up to the
// modality's cap has crossed the network and been held in memory, the cost the
// cap exists to bound.
//
// ContentLength is -1, as for a chunked response, so the Content-Length guard
// does not fire and only the ordering decides whether the body is read. The
// assertion is on whether it was read at all; timing or handler-side byte
// counts would race with the client closing the connection.
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

	_, _, err := step.download(context.Background(), "http://media.invalid/clip.wav", reqcommon.ModalityAudio)
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

// image_url downloads accept any Content-Type under the built-in default
// allowlist. Audio and video are stricter; the image default stays permissive
// so traffic relying on it keeps working. Setting allowed_image_content_types
// explicitly does enforce here, covered by
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

// The other half of the image content-type rule: once an operator sets
// allowed_image_content_types, the list is enforced on downloaded bytes too,
// not just data URIs. Otherwise the origin's Content-Type is inlined verbatim
// into the rewritten data URI and the lockdown is a no-op on the HTTP path.
func TestReplaceMediaURLsStep_ImageURL_ExplicitAllowlistAppliesToDownload(t *testing.T) {
	jpegServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", testImageJPEGContentType)
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

// The audio download case, for video_url with a video/mp4 payload.
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
	if reqCtx.MultimodalEntries[0].Modality != reqcommon.ModalityVideo {
		t.Fatalf("expected Modality=%q, got %q", reqcommon.ModalityVideo, reqCtx.MultimodalEntries[0].Modality)
	}
	msgs := reqCtx.Body["messages"].([]any)
	inner := msgs[0].(map[string]any)["content"].([]any)[0].(map[string]any)["video_url"].(map[string]any)
	url := inner["url"].(string)
	if !strings.HasPrefix(url, "data:video/mp4;base64,") {
		t.Fatalf("expected inlined data URI, got %s", url)
	}
}

// A valid audio data URI under audio_url is accepted, kept in place, and added
// to MultimodalEntries as one audio entry.
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
	if reqCtx.MultimodalEntries[0].Modality != reqcommon.ModalityAudio {
		t.Fatalf("expected Modality=%q, got %q", reqcommon.ModalityAudio, reqCtx.MultimodalEntries[0].Modality)
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

// The per-modality cap applies to bytes arriving inline in a URL slot, not only
// to bytes pulled over the network. validateInlineAudio rejects the same
// payload sent as input_audio, so accepting it here would let the two ways of
// sending one audio clip disagree.
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

// The data URI bound reads the per-modality override, not the global default: a
// payload over the 1 MB global cap is accepted once the audio cap is raised.
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

// Pins the default in enforceInlineSize: with no max_image_download_size set,
// an image data URI is bounded by the server's max_request_body_size, not by
// max_download_size. Every deployment setting max_download_size relies on that
// pre-existing behavior, so the fallback must not reach a data URI.
func TestReplaceMediaURLsStep_ImageDataURI_ExemptFromGlobalCap(t *testing.T) {
	oversized := strings.Repeat("A", 2*1024*1024)
	step, err := NewReplaceMediaURLsStep(nil, map[string]any{"max_download_size": 1})
	if err != nil {
		t.Fatal(err)
	}
	reqCtx := dataURIReqCtx("image_url", testImagePNGMIME, oversized)
	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("image data URI must stay exempt from the global cap, got %v", err)
	}
	if len(reqCtx.MultimodalEntries) != 1 {
		t.Fatalf("expected 1 entry, got %d", len(reqCtx.MultimodalEntries))
	}
}

// The other half of the pair above: max_image_download_size is an explicit
// request to bound image payloads, so it reaches a data URI too, not only the
// download path, as an operator setting it to bound memory would expect.
func TestReplaceMediaURLsStep_ImageDataURI_HonorsExplicitImageCap(t *testing.T) {
	step, err := NewReplaceMediaURLsStep(nil, map[string]any{"max_image_download_size": 1})
	if err != nil {
		t.Fatal(err)
	}

	// 2 MB of base64 against a 1 MB image cap.
	oversized := dataURIReqCtx("image_url", testImagePNGMIME, strings.Repeat("A", 2*1024*1024))
	err = step.Execute(context.Background(), oversized)
	if err == nil || !errors.Is(err, pipeline.ErrBadRequest) {
		t.Fatalf("expected ErrBadRequest for an image data URI over an explicit 1 MB cap, got %v", err)
	}

	// A payload inside the same cap still goes through.
	small := dataURIReqCtx("image_url", testImagePNGMIME, strings.Repeat("A", 1024))
	if err := step.Execute(context.Background(), small); err != nil {
		t.Fatalf("payload within the explicit cap must be accepted, got %v", err)
	}
	if len(small.MultimodalEntries) != 1 {
		t.Fatalf("expected 1 entry, got %d", len(small.MultimodalEntries))
	}
}

// The audio data URI case, for video.
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
	if reqCtx.MultimodalEntries[0].Modality != reqcommon.ModalityVideo {
		t.Fatalf("expected Modality=%q, got %q", reqcommon.ModalityVideo, reqCtx.MultimodalEntries[0].Modality)
	}
}

// A well-formed input_audio part (base64 payload, known format) passes
// validation and leaves the body unchanged.
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
	if entry.Modality != reqcommon.ModalityAudio {
		t.Fatalf("expected Modality=%q, got %q", reqcommon.ModalityAudio, entry.Modality)
	}
	msgs := reqCtx.Body["messages"].([]any)
	inner := msgs[0].(map[string]any)["content"].([]any)[0].(map[string]any)["input_audio"].(map[string]any)
	if inner["data"].(string) != "UklGRg==" || inner["format"].(string) != "wav" {
		t.Fatalf("expected input_audio body unchanged, got %+v", inner)
	}
}

// A data:audio/wav URI supplied in an image_url slot is rejected.
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

// The symmetric case: an image data URI in an audio_url slot is rejected.
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

// An unknown format string ("aiff") is rejected before validation.
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

// Exactly cap bytes must be accepted: the base64 length of cap bytes equals the
// size-check bound, so a strictly-greater comparison must not reject it.
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

// An input_audio item whose base64 payload alone exceeds 4/3 *
// max_download_size is rejected without being decoded.
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

// Pins the vocabulary an operator has to write. An input_audio part names a
// format and the allowlist names MIME types, so "mp3" is checked as audio/mpeg
// and an allowlist of audio/mp3 alone does not admit it. The rejection must
// name the format, or the operator cannot connect it back to the config.
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

// The companion to the test above: that one pins which MIME a format maps to,
// this one pins the shape it has to be written in. validateInlineAudio hands
// audioFormatMIME's value to allowedContentTypeForModality without
// normalizing it, and that match is by equality, so a value carrying a MIME
// parameter or any upper case would miss every allowlist entry and refuse its
// format however the operator wrote the config. Adding such an entry fails
// here rather than on a request.
func TestAudioFormatMIMEValuesAreNormalized(t *testing.T) {
	for format, mime := range audioFormatMIME {
		if got := normalizeMediaType(mime); got != mime {
			t.Errorf("audioFormatMIME[%q] = %q, which normalizes to %q; values must be bare lowercase types",
				format, mime, got)
		}
	}
}

// A bad input_audio LAST in walker order, behind an audio_url that would
// otherwise be downloaded. The inline checks are local, so they must all run
// before any download starts, or a request that will be rejected anyway still
// pays for the bytes. The assertion is the server counter: never hit.
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

// One request with one image URL, one audio URL, and one video URL: all three
// are inlined as data URIs, added to MultimodalEntries in walker order, and
// counted against max_multimodal_entries.
func TestReplaceMediaURLsStep_MixedImageAudioVideo(t *testing.T) {
	mediaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, ".jpg"):
			w.Header().Set("Content-Type", testImageJPEGContentType)
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
	// All three parts feed into MultimodalEntries, in walker order: URL refs
	// first (image, audio, video), then any inline refs (none here).
	if len(reqCtx.MultimodalEntries) != 3 {
		t.Fatalf("expected 3 entries (1 image + 1 audio + 1 video), got %d", len(reqCtx.MultimodalEntries))
	}
	wantModalities := []reqcommon.Modality{reqcommon.ModalityImage, reqcommon.ModalityAudio, reqcommon.ModalityVideo}
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

// Locks in the walker-order invariant for audio, the only modality carrying
// both a URL-based variant (audio_url) and an inline one (input_audio). The
// request has input_audio A first and audio_url B second, so entries[0] must
// carry A's inline payload and entries[1] B's downloaded payload. A split
// append (URLs first, inline second) would swap them, and encode and decode,
// which walk parts in request order, would pair each entry with the wrong part.
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
	// The inline part's input_audio.data must be untouched and the URL slot
	// overwritten in place with the downloaded data URI. With the
	// collectMediaParts check below, this verifies entry 0 is the inline part
	// and entry 1 the URL part.
	msgs := reqCtx.Body["messages"].([]any)
	inlinePart := msgs[0].(map[string]any)["content"].([]any)[0].(map[string]any)["input_audio"].(map[string]any)
	if got, _ := inlinePart["data"].(string); got != inlineData {
		t.Errorf("input_audio data changed: got %q, want %q", got, inlineData)
	}
	urlPart := msgs[0].(map[string]any)["content"].([]any)[1].(map[string]any)["audio_url"].(map[string]any)
	if got, _ := urlPart["url"].(string); !strings.HasPrefix(got, "data:audio/wav;base64,") {
		t.Errorf("audio_url url = %q, want inlined data URI", got)
	}

	// Downstream alignment: collectMediaParts walks the SAME body in request
	// order, so entries[0] (inline) must pair with partsByMod[audio][0] and
	// entries[1] (URL) with partsByMod[audio][1]. A regression that swaps the
	// entries would break this pairing silently.
	items, _ := promptItems(reqCtx.Body, reqcommon.APITypeChatCompletions)
	partsByMod := groupMediaPartsByModality(collectMediaParts(items, reqcommon.APITypeChatCompletions))
	audioParts := partsByMod[reqcommon.ModalityAudio]
	if got := len(audioParts); got != 2 {
		t.Fatalf("partsByMod[audio] len = %d, want 2", got)
	}
	if _, ok := audioParts[0].part["input_audio"].(map[string]any); !ok {
		t.Errorf("audio parts[0] = %+v, want the input_audio part first", audioParts[0].part)
	}
	if _, ok := audioParts[1].part["audio_url"].(map[string]any); !ok {
		t.Errorf("audio parts[1] = %+v, want the audio_url part second", audioParts[1].part)
	}
}

// One image, one audio, and one video against a cap of 2: three media parts of
// any mix count toward max_multimodal_entries, so the request is rejected.
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

// A video payload rejected under max_download_size alone is accepted once
// max_video_download_size raises the video-only cap.
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

// With no per-modality override set, audio downloads honor the global
// max_download_size.
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

// Inline input_audio size validation respects max_audio_download_size, not the
// global cap.
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

// For all three per-modality caps, non-positive and megabyte-overflow values
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

// A narrower audio allowlist ({audio/wav}): audio/wav passes and audio/mpeg,
// which the default set allows, is rejected.
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

// The image and audio cases, for video: narrowing to {video/mp4} rejects the
// default-allowed video/webm.
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

// The video case, for max_image_download_size: a payload rejected under a 1 MB
// global cap is accepted when the image-only cap is raised.
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

// The audio case, for the image allowlist: narrowing to {image/png} rejects the
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

// With no per-modality allowlist configured, the built-in defaults apply:
// audio/mpeg, a default entry, is accepted.
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

// Construction fails when a per-modality allowlist entry is not a string.
// Silently dropping the bad entry would be a security downgrade: the operator's
// intent to lock down the allowlist is lost.
func TestReplaceMediaURLsStep_RejectsNonStringAllowedContentType(t *testing.T) {
	_, err := NewReplaceMediaURLsStep(nil, map[string]any{
		"allowed_audio_content_types": []any{testAudioWAVMIME, 42},
	})
	if err == nil {
		t.Fatal("expected construction error for non-string allowlist entry")
	}
}

// Construction fails when a per-modality allowlist is set to a non-list value.
func TestReplaceMediaURLsStep_RejectsNonListAllowedContentTypes(t *testing.T) {
	_, err := NewReplaceMediaURLsStep(nil, map[string]any{
		"allowed_video_content_types": testVideoMP4MIME,
	})
	if err == nil {
		t.Fatal("expected construction error for non-list allowlist value")
	}
}

// Construction fails when a per-modality allowlist key is present with a null
// value, the shape a template produces when its variable is unset. Treating it
// as absent is worse than an empty list: for image it also leaves the
// download-path Content-Type check off, so a line written to add enforcement
// adds none. allowed_domains rejects null the same way, and the image case is
// asserted below as the one with two behaviors riding on the param.
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

// Construction fails on an allowlist entry that normalizes to nothing, the
// same class of templating slip as the null value above and a worse outcome:
// such an entry keys the set at "", which no normalized Content-Type equals,
// so the modality would reject every request while an empty list accepts
// anything. A typo would invert the control rather than relax it.
func TestReplaceMediaURLsStep_RejectsEmptyAllowedContentTypeEntry(t *testing.T) {
	for _, tc := range []struct {
		name  string
		entry string
	}{
		{"empty", ""},
		{"spaces", "   "},
		{"tab and newline", "\t\n"},
		{"parameters only", "; charset=utf-8"},
		{"leading comma", ",audio/wav"}, // normalization cuts at the comma
	} {
		t.Run(tc.name, func(t *testing.T) {
			entry := tc.entry
			for _, key := range []string{
				"allowed_image_content_types",
				"allowed_audio_content_types",
				"allowed_video_content_types",
			} {
				params := map[string]any{key: []any{testAudioWAVMIME, entry}}
				if _, err := NewReplaceMediaURLsStep(nil, params); err == nil {
					t.Errorf("%s: expected construction error for entry %q", key, entry)
				}
			}
		})
	}
}

// allowed_domains inverts the same way, which is why it shares the parser: an
// empty set means unrestricted in hostAllowed, so [""] is a live allowlist
// whose only key no lowercased host equals, and every download would be
// refused by a line written to permit one. Whitespace-only entries are not
// covered here: this guard folds case and nothing else, since trimming would
// widen what it matches.
func TestReplaceMediaURLsStep_RejectsEmptyAllowedDomainEntry(t *testing.T) {
	if _, err := NewReplaceMediaURLsStep(nil, map[string]any{
		"allowed_domains": []any{"example.com", ""},
	}); err == nil {
		t.Error("expected construction error for an empty allowed_domains entry")
	}
	// A list that is empty outright keeps its documented meaning.
	if _, err := NewReplaceMediaURLsStep(nil, map[string]any{
		"allowed_domains": []any{},
	}); err != nil {
		t.Errorf("empty allowed_domains must stay unrestricted, got %v", err)
	}
}

// Pins the pairing the null case above protects: setting the image allowlist at
// all, empty list included, turns on the download-path Content-Type check,
// while leaving it unset keeps the permissive path.
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
			got := step.(*ReplaceMediaURLsStep).enforceDownloadContentType(reqcommon.ModalityImage)
			if got != tc.want {
				t.Fatalf("enforceDownloadContentType(image) = %v, want %v", got, tc.want)
			}
		})
	}
}

// An empty allowed_<modality>_content_types list disables that modality's
// allowlist, so any type is accepted, mirroring the "empty means unrestricted"
// convention allowed_domains uses.
func TestReplaceMediaURLsStep_AllowedContentTypes_EmptyMeansUnrestricted(t *testing.T) {
	step, err := NewReplaceMediaURLsStep(nil, map[string]any{
		"allowed_video_content_types": []any{},
	})
	if err != nil {
		t.Fatal(err)
	}
	// application/octet-stream is outside the built-in video allowlist and
	// would normally be rejected; the unrestricted override must accept it.
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

// A caller mutating the returned per-modality set must not reach the
// package-level defaultAllowedContentTypesByModality: an entry added to the
// returned image set must not appear in a fresh call. This keeps every
// subsequent ReplaceMediaURLsStep from picking up leaked overrides.
func TestParsePerModalityContentTypes_DoesNotAliasDefaults(t *testing.T) {
	first, _, err := parsePerModalityContentTypes(nil)
	if err != nil {
		t.Fatalf("parsePerModalityContentTypes returned error: %v", err)
	}
	const poison = "application/x-poison"
	first[reqcommon.ModalityImage][poison] = struct{}{}

	second, _, err := parsePerModalityContentTypes(nil)
	if err != nil {
		t.Fatalf("parsePerModalityContentTypes returned error: %v", err)
	}
	if _, leaked := second[reqcommon.ModalityImage][poison]; leaked {
		t.Fatal("mutation of first result reached defaults and leaked into second result")
	}
	if _, leaked := defaultAllowedContentTypesByModality[reqcommon.ModalityImage][poison]; leaked {
		t.Fatal("mutation of first result reached package-level defaults")
	}
}

// TestReplaceMediaURLsStep_ChatCompletionsInputImage covers an input_image part
// sent on a chat-completions request. vLLM's chat parser primes input_image and
// image_url through the same content part map, so such a part reaches the model
// and has to be fetched under this step's address guard and size limit rather
// than left for the model server to fetch itself. The sidecar's encoder fan-out
// collects it for the same reason.
//
// The URL sits where a Responses input_image keeps it, a bare string on the
// part, even though the request is chat completions, so this also pins that the
// rewritten data URI goes back in that shape.
func TestReplaceMediaURLsStep_ChatCompletionsInputImage(t *testing.T) {
	imageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", testImageJPEGContentType)
		_, _ = w.Write([]byte("jpeg-bytes"))
	}))
	defer imageServer.Close()

	step := newLoopbackStep(t, map[string]any{"download_timeout": "5s"})

	reqCtx := &pipeline.RequestContext{
		OriginalPath: reqcommon.PathChatCompletions,
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{"type": "text", "text": "describe this"},
						map[string]any{
							"type":      reqcommon.PartTypeInputImage,
							"image_url": imageServer.URL + "/photo.jpg",
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
		t.Fatalf("expected the input_image part to produce 1 multimodal entry, got %d", len(reqCtx.MultimodalEntries))
	}
	if got := reqCtx.MultimodalEntries[0].Modality; got != reqcommon.ModalityImage {
		t.Errorf("entry modality = %q, want %q", got, reqcommon.ModalityImage)
	}

	msgs := reqCtx.Body["messages"].([]any)
	content := msgs[0].(map[string]any)["content"].([]any)
	url, ok := content[1].(map[string]any)[reqcommon.FieldImageURL].(string)
	if !ok {
		t.Fatalf("expected image_url to stay a bare string, got %T", content[1].(map[string]any)[reqcommon.FieldImageURL])
	}
	if !strings.HasPrefix(url, "data:image/jpeg;base64,") {
		t.Errorf("expected the URL rewritten as a data URI, got %q", url)
	}
}

// A chat-completions message defines no output array, so an image_url part
// under one names content the client never sent and is left alone.
func TestReplaceMediaURLsStep_IgnoresOutputOnChatCompletions(t *testing.T) {
	var hits atomic.Int32
	imageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		w.Header().Set("Content-Type", testImageJPEGContentType)
		_, _ = w.Write([]byte("jpeg-bytes"))
	}))
	defer imageServer.Close()

	step := newLoopbackStep(t, map[string]any{"download_timeout": "5s"})

	reqCtx := &pipeline.RequestContext{
		OriginalPath: reqcommon.PathChatCompletions,
		Body: map[string]any{
			"messages": []any{
				map[string]any{
					"role": "user",
					"output": []any{
						map[string]any{
							"type":      reqcommon.PartTypeImageURL,
							"image_url": map[string]any{"url": imageServer.URL + "/stray.jpg"},
						},
					},
				},
			},
		},
	}

	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := hits.Load(); got != 0 {
		t.Fatalf("expected no download for an output array on a chat request, got %d", got)
	}
	if len(reqCtx.MultimodalEntries) != 0 {
		t.Fatalf("expected no multimodal entries, got %d", len(reqCtx.MultimodalEntries))
	}
}

// TestReplaceMediaURLsStep_Responses_InlinesFunctionCallOutputImage covers a
// Responses function_call_output, which carries its parts under output rather
// than content. vLLM forwards that array as a tool message's content, so media
// in it reaches the model like any other part and has to be fetched under this
// step's address guard and size limit rather than left for the model server to
// fetch itself.
func TestReplaceMediaURLsStep_Responses_InlinesFunctionCallOutputImage(t *testing.T) {
	imageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", testImageJPEGContentType)
		_, _ = w.Write([]byte("jpeg-bytes"))
	}))
	defer imageServer.Close()

	step := newLoopbackStep(t, map[string]any{"download_timeout": "5s"})

	reqCtx := &pipeline.RequestContext{
		OriginalPath: reqcommon.PathResponses,
		Body: map[string]any{
			"input": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{"type": "input_text", "text": "describe this"},
						map[string]any{"type": reqcommon.PartTypeInputImage, "image_url": imageServer.URL + "/content.jpg"},
					},
				},
				map[string]any{
					"type":    "function_call_output",
					"call_id": "call-1",
					"output": []any{
						map[string]any{"type": reqcommon.PartTypeInputImage, "image_url": imageServer.URL + "/output.jpg"},
					},
				},
			},
		},
	}

	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(reqCtx.MultimodalEntries) != 2 {
		t.Fatalf("expected 2 multimodal entries, got %d", len(reqCtx.MultimodalEntries))
	}

	input := reqCtx.Body["input"].([]any)
	contentURL := input[0].(map[string]any)["content"].([]any)[1].(map[string]any)["image_url"].(string)
	if !strings.HasPrefix(contentURL, "data:image/jpeg;base64,") {
		t.Fatalf("expected the content image inlined, got %s", contentURL)
	}
	outputURL := input[1].(map[string]any)["output"].([]any)[0].(map[string]any)["image_url"].(string)
	if !strings.HasPrefix(outputURL, "data:image/jpeg;base64,") {
		t.Fatalf("expected the output image inlined, got %s", outputURL)
	}
}

// TestReplaceMediaURLsStep_ResponsesIgnoresChatImagePart is the other half of
// reqcommon.PartModality's rule. The Responses input union does not define image_url, so a
// request carrying one fails the model server's input validation and no worker
// sees it. Collecting it here would download an image the request never uses
// and leave an entry the render service reports no hash for, failing the
// request on the feature count instead.
func TestReplaceMediaURLsStep_ResponsesIgnoresChatImagePart(t *testing.T) {
	var hits atomic.Int32
	imageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		w.Header().Set("Content-Type", testImageJPEGContentType)
		_, _ = w.Write([]byte("jpeg-bytes"))
	}))
	defer imageServer.Close()

	step := newLoopbackStep(t, map[string]any{"download_timeout": "5s"})

	reqCtx := &pipeline.RequestContext{
		OriginalPath: reqcommon.PathResponses,
		Body: map[string]any{
			"input": []any{
				map[string]any{
					"role": "user",
					"content": []any{
						map[string]any{"type": "input_text", "text": "describe this"},
						map[string]any{
							"type":      reqcommon.PartTypeImageURL,
							"image_url": map[string]any{"url": imageServer.URL + "/photo.jpg"},
						},
					},
				},
			},
		},
	}

	if err := step.Execute(context.Background(), reqCtx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(reqCtx.MultimodalEntries) != 0 {
		t.Errorf("expected no multimodal entry for a chat image part on a Responses request, got %d", len(reqCtx.MultimodalEntries))
	}
	if n := hits.Load(); n != 0 {
		t.Errorf("expected no download, got %d", n)
	}
}
