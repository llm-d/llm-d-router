package singlepool

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

func newOpenAIClient() *openai.Client {
	c := openai.NewClient(option.WithBaseURL(fmt.Sprintf("http://localhost:%s/v1", port)))
	return &c
}

func extractInferenceHeaders(httpResp *http.Response) (string, string, string) {
	return httpResp.Header.Get("x-inference-namespace"),
		httpResp.Header.Get("x-inference-pod"),
		httpResp.Header.Get("x-inference-port")
}

// doPost sends a POST request with a JSON body to the given path, asserts HTTP 200,
// and returns the x-inference-namespace, x-inference-pod headers and the response body.
func doPost(path, body string, extraHeaders map[string]string) (string, string, []byte) {
	req, err := http.NewRequest(http.MethodPost, fmt.Sprintf("http://localhost:%s%s", port, path), strings.NewReader(body))
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	req.Header.Set("Content-Type", "application/json")
	for k, v := range extraHeaders {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	defer func() {
		gomega.Expect(resp.Body.Close()).ToNot(gomega.HaveOccurred())
	}()
	gomega.Expect(resp.StatusCode).Should(gomega.Equal(http.StatusOK))

	respBody, err := io.ReadAll(resp.Body)
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())

	return resp.Header.Get("x-inference-namespace"), resp.Header.Get("x-inference-pod"), respBody
}

func runCompletion(prompt string, theModel openai.CompletionNewParamsModel) (string, string, string) {
	var httpResp *http.Response

	completionParams := openai.CompletionNewParams{
		Prompt: openai.CompletionNewParamsPromptUnion{
			OfString: openai.String(prompt),
		},
		Model: theModel,
	}

	ginkgo.By(fmt.Sprintf("Sending Completion Request: (port %s) %#v", port, completionParams))

	resp, err := newOpenAIClient().Completions.New(testConfig.Context, completionParams, option.WithResponseInto(&httpResp), option.WithRequestTimeout(readyTimeout))

	ginkgo.By(fmt.Sprintf("Verifying Completion Response: %#v", resp))

	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	gomega.Expect(resp.Choices).Should(gomega.HaveLen(1))
	gomega.Expect(resp.Choices[0].FinishReason).Should(gomega.Equal(openai.CompletionChoiceFinishReasonStop))
	gomega.Expect(resp.Choices[0].Text).Should(gomega.Equal(prompt))

	return extractInferenceHeaders(httpResp)
}

// tryCompletion is like runCompletion but returns an error instead of asserting,
// intended for use inside Eventually blocks where transient failures are acceptable.
func tryCompletion(prompt string, theModel openai.CompletionNewParamsModel) (string, string, string, error) {
	var httpResp *http.Response
	completionParams := openai.CompletionNewParams{
		Prompt: openai.CompletionNewParamsPromptUnion{OfString: openai.String(prompt)},
		Model:  theModel,
	}
	resp, err := newOpenAIClient().Completions.New(
		testConfig.Context,
		completionParams,
		option.WithResponseInto(&httpResp),
		option.WithRequestTimeout(readyTimeout),
	)
	if err != nil {
		return "", "", "", err
	}
	if httpResp == nil {
		return "", "", "", errors.New("missing http response")
	}
	if len(resp.Choices) != 1 {
		return "", "", "", fmt.Errorf("expected 1 choice, got %d", len(resp.Choices))
	}
	ns, pod, p := extractInferenceHeaders(httpResp)
	return ns, pod, p, nil
}

func runChatCompletion(prompt, modelName string) (string, string, string) {
	var httpResp *http.Response

	params := openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage(prompt),
		},
		Model: modelName,
	}
	resp, err := newOpenAIClient().Chat.Completions.New(testConfig.Context, params, option.WithResponseInto(&httpResp))
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	gomega.Expect(resp.Choices).Should(gomega.HaveLen(1))
	gomega.Expect(resp.Choices[0].FinishReason).Should(gomega.Equal("stop"))
	gomega.Expect(resp.Choices[0].Message.Content).Should(gomega.Equal(prompt))

	return extractInferenceHeaders(httpResp)
}

// runRawChatCompletion POSTs the given JSON body to /v1/chat/completions and returns
// the x-inference-namespace and x-inference-pod response headers.
func runRawChatCompletion(body string) (string, string) {
	ns, pod, _ := doPost("/v1/chat/completions", body, nil)
	return ns, pod
}

// runChatCompletionWithImages sends a multimodal chat completion request with one or more image_url
// content blocks. When called with no arguments it defaults to testImageURL (single image).
// Each image is assigned a uuid derived from its index.
// Returns the namespace and pod name from the response headers.
func runChatCompletionWithImages(imageURLs ...string) (string, string) {
	if len(imageURLs) == 0 {
		imageURLs = []string{testImageURL}
	}
	ginkgo.By(fmt.Sprintf("Sending Multimodal Chat Completion Request with %d images", len(imageURLs)))
	var sb strings.Builder
	for i, url := range imageURLs {
		sb.WriteString(fmt.Sprintf(`{"type":"image_url","image_url":{"url":%q},"uuid":"image-%d"},`, url, i))
	}
	body := fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":[%s{"type":"text","text":"Describe what you see."}]}],"max_tokens":150}`,
		simModelName, sb.String())
	return runRawChatCompletion(body)
}

// runChatCompletionWithVideo sends a multimodal chat completion request with a video_url content block.
// Returns the namespace and pod name from the response headers.
func runChatCompletionWithVideo() (string, string) {
	ginkgo.By("Sending Multimodal Chat Completion Request with video: " + testVideoURL)
	body := fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":[{"type":"text","text":"What is happening in this video?"},{"type":"video_url","video_url":{"url":%q}}]}]}`,
		simModelName, testVideoURL)
	return runRawChatCompletion(body)
}

// runChatCompletionWithImageEmbeds sends a chat completion request with an image_embeds content block
// carrying a pre-encoded tensor. image_embeds is not a recognised multimodal type for encode
// disaggregation, so the request routes like a text request (decode-only or prefill-decode).
// Returns the namespace and pod name from the response headers.
func runChatCompletionWithImageEmbeds() (string, string) {
	ginkgo.By("Sending Chat Completion Request with image_embeds")
	body := fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":[{"type":"text","text":"Describe this embedded image:"},{"type":"image_embeds","image_embeds":%q,"uuid":"embedded-image-1"}]}]}`,
		simModelName, testImageEmbeds)
	return runRawChatCompletion(body)
}

// runChatCompletionWithAudio sends a chat completion request with an input_audio content block.
// input_audio is a recognised multimodal type so it triggers the encode stage.
// Returns the namespace and pod name from the response headers.
func runChatCompletionWithAudio() (string, string) {
	ginkgo.By("Sending Chat Completion Request with input_audio")
	body := fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":[{"type":"text","text":"What is being said in this audio clip?"},{"type":"input_audio","input_audio":{"data":%q,"format":"wav"}}]}],"max_tokens":100}`,
		simModelName, testAudioData)
	return runRawChatCompletion(body)
}

func runStreamingCompletion(prompt string, theModel openai.CompletionNewParamsModel) (string, string) {
	ginkgo.By(fmt.Sprintf("Sending Streaming Completion Request: (port %s) model=%s", port, theModel))
	body := fmt.Sprintf(`{"model":"%s","prompt":"%s","max_tokens":50,"stream":true}`, theModel, prompt)
	ns, pod, respBody := doPost("/v1/completions", body, nil)
	ginkgo.By(fmt.Sprintf("Streaming Completion received response length: %d bytes", len(respBody)))
	return ns, pod
}

func runStreamingChatCompletion(prompt string) (string, string) {
	ginkgo.By(fmt.Sprintf("Sending Streaming Chat Completion Request: (port %s)", port))
	body := fmt.Sprintf(`{"model":"%s","messages":[{"role":"user","content":"%s"}],"stream":true}`, simModelName, prompt)
	ns, pod, respBody := doPost("/v1/chat/completions", body, nil)
	ginkgo.By(fmt.Sprintf("Streaming Chat Completion received response length: %d bytes", len(respBody)))
	return ns, pod
}

// runCompletionWithCacheThreshold sends a completion request with cache_hit_threshold parameter.
// This triggers the decode-first optimization in the shared-storage connector.
// Returns namespace header, pod header, and the finish reason from the response.
func runCompletionWithCacheThreshold(prompt string, cacheHitThreshold float64, forceCacheThresholdFinishReason bool) (string, string, string) {
	ginkgo.By(fmt.Sprintf("Sending Completion Request with cache_hit_threshold=%v, forceCacheThreshold=%v", cacheHitThreshold, forceCacheThresholdFinishReason))
	body := fmt.Sprintf(`{"model":"%s","prompt":"%s","max_tokens":10,"cache_hit_threshold":%v}`, simModelName, prompt, cacheHitThreshold)
	extraHeaders := cacheThresholdHeaders(forceCacheThresholdFinishReason)
	ns, pod, respBody := doPost("/v1/completions", body, extraHeaders)
	finishReason := extractFinishReason(string(respBody))
	ginkgo.By(fmt.Sprintf("Completion Response: ns=%s, pod=%s, finish_reason=%s", ns, pod, finishReason))
	return ns, pod, finishReason
}

// runStreamingCompletionWithCacheThreshold sends a streaming completion request with cache_hit_threshold.
func runStreamingCompletionWithCacheThreshold(prompt string, cacheHitThreshold float64, forceCacheThresholdFinishReason bool) (string, string, string) {
	ginkgo.By(fmt.Sprintf("Sending Streaming Completion Request with cache_hit_threshold=%v, forceCacheThreshold=%v", cacheHitThreshold, forceCacheThresholdFinishReason))
	body := fmt.Sprintf(`{"model":"%s","prompt":"%s","max_tokens":10,"stream":true,"cache_hit_threshold":%v}`, simModelName, prompt, cacheHitThreshold)
	extraHeaders := cacheThresholdHeaders(forceCacheThresholdFinishReason)
	ns, pod, respBody := doPost("/v1/completions", body, extraHeaders)
	finishReason := extractFinishReasonFromStreaming(string(respBody))
	ginkgo.By(fmt.Sprintf("Streaming Completion Response: ns=%s, pod=%s, finish_reason=%s", ns, pod, finishReason))
	return ns, pod, finishReason
}

func cacheThresholdHeaders(force bool) map[string]string {
	if force {
		// Forces the simulator to return cache_threshold as the finish_reason.
		return map[string]string{"X-Cache-Threshold-Finish-Reason": "true"}
	}
	return nil
}
