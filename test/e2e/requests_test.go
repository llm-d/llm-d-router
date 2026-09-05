package e2e

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"

	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

func newOpenAIClient() *openai.Client {
	c := openai.NewClient(option.WithBaseURL(fmt.Sprintf("http://localhost:%d/v1", getPort())))
	return &c
}

func extractInferenceHeaders(httpResp *http.Response) (string, string, string) {
	return httpResp.Header.Get("x-inference-namespace"),
		httpResp.Header.Get("x-inference-pod"),
		httpResp.Header.Get("x-inference-port")
}

func generateAndCheckLoad(count int) {
	nsName := getNamespace()
	for range count {
		prefillPods, decodePods := getModelServerPods(podSelector, prefillSelector, decodeSelector, nsName)
		gomega.Expect(prefillPods).Should(gomega.BeEmpty())
		gomega.Expect(decodePods).Should(gomega.HaveLen(1))

		nsHdr, podHdr, _ := runCompletion(simplePrompt, simModelName)
		gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
		gomega.Expect(podHdr).Should(gomega.Equal(decodePods[0]))

		nsHdr, podHdr, _ = runChatCompletion(simplePrompt, simModelName)
		gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
		gomega.Expect(podHdr).Should(gomega.Equal(decodePods[0]))

		nsHdr, podHdr, _ = runEmbeddings(singleEmbedding, simModelName)
		gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
		gomega.Expect(podHdr).Should(gomega.Equal(decodePods[0]))

		nsHdr, podHdr, _ = runEmbeddings(doubleEmbedding, simModelName)
		gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
		gomega.Expect(podHdr).Should(gomega.Equal(decodePods[0]))
	}
}

// doPost sends a POST request with a JSON body to the given path, asserts HTTP 200,
// and returns the x-inference-namespace, x-inference-pod headers and the response body.
func doPost(path, body string, extraHeaders map[string]string) (string, string, []byte) {
	req, err := http.NewRequest(http.MethodPost, fmt.Sprintf("http://localhost:%d%s", getPort(), path), strings.NewReader(body))
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

// doPostWithError sends a POST request with a JSON body to the given path
// and returns the status code and the response body.
func doPostWithError(path, body string, extraHeaders map[string]string) (int, []byte) {
	req, err := http.NewRequest(http.MethodPost, fmt.Sprintf("http://localhost:%d%s", getPort(), path), strings.NewReader(body))
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

	respBody, err := io.ReadAll(resp.Body)
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())

	return resp.StatusCode, respBody
}

func runCompletion(prompt string, theModel openai.CompletionNewParamsModel) (string, string, string) {
	var httpResp *http.Response

	completionParams := openai.CompletionNewParams{
		Prompt: openai.CompletionNewParamsPromptUnion{
			OfString: openai.String(prompt),
		},
		Model: theModel,
	}

	ginkgo.By(fmt.Sprintf("Sending Completion Request: (port %d) %#v", getPort(), completionParams))

	resp, err := newOpenAIClient().Completions.New(testConfig.Context, completionParams, option.WithResponseInto(&httpResp), option.WithRequestTimeout(readyTimeout))

	ginkgo.By(fmt.Sprintf("Verifying Completion Response: %#v", resp))

	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	gomega.Expect(resp.Choices).Should(gomega.HaveLen(1))
	gomega.Expect(resp.Choices[0].FinishReason).Should(gomega.Equal(openai.CompletionChoiceFinishReasonStop))
	gomega.Expect(resp.Choices[0].Text).Should(gomega.Equal(prompt))

	return extractInferenceHeaders(httpResp)
}

// runCompletionWithModelRewrite POSTs a raw completion request carrying incomingModel in the
// body. When targetModel is non-empty it is sent as the x-llm-d-model-name-rewrite header,
// forcing EPP to rewrite the body's model field before forwarding. incomingModel is not a
// name the model server recognizes on its own, so a 200 response with the rewritten model
// echoed back proves EPP actually rewrote the body rather than passing it through.
func runCompletionWithModelRewrite(prompt, incomingModel, targetModel string) string {
	body := fmt.Sprintf(`{"model":%q,"prompt":%q,"max_tokens":10}`, incomingModel, prompt)
	headers := map[string]string{}
	if targetModel != "" {
		headers[metadata.ModelNameRewriteKey] = targetModel
	}
	_, _, respBody := doPost("/v1/completions", body, headers)

	var resp struct {
		Model string `json:"model"`
	}
	gomega.Expect(json.Unmarshal(respBody, &resp)).ShouldNot(gomega.HaveOccurred())

	return resp.Model
}

// tryCompletion is like runCompletion but returns an error instead of asserting,
// intended for use inside Eventually blocks where transient failures are acceptable.
func tryCompletion(prompt string, theModel openai.CompletionNewParamsModel) (string, string, error) {
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
		return "", "", err
	}
	if httpResp == nil {
		return "", "", errors.New("missing http response")
	}
	if len(resp.Choices) != 1 {
		return "", "", fmt.Errorf("expected 1 choice, got %d", len(resp.Choices))
	}
	if resp.Choices[0].FinishReason != openai.CompletionChoiceFinishReasonStop {
		return "", "", fmt.Errorf("expected finish reason %q, got %q",
			openai.CompletionChoiceFinishReasonStop, resp.Choices[0].FinishReason)
	}
	if resp.Choices[0].Text != prompt {
		return "", "", fmt.Errorf("expected echoed prompt, got %q", resp.Choices[0].Text)
	}
	ns, pod, _ := extractInferenceHeaders(httpResp)
	return ns, pod, nil
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

func runEmbeddings(embeddings []string, modelName string) (string, string, string) {
	var httpResp *http.Response

	input := openai.EmbeddingNewParamsInputUnion{}
	if len(embeddings) == 1 {
		input.OfString = param.NewOpt(embeddings[0])
	} else {
		input.OfArrayOfStrings = embeddings
	}

	params := openai.EmbeddingNewParams{
		Input: input,
		Model: modelName,
	}
	resp, err := newOpenAIClient().Embeddings.New(testConfig.Context, params, option.WithResponseInto(&httpResp))
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	gomega.Expect(resp.Data).Should(gomega.HaveLen(len(embeddings)))

	return extractInferenceHeaders(httpResp)
}

// runRawChatCompletion POSTs the given JSON body to /v1/chat/completions and returns
// the x-inference-namespace and x-inference-pod response headers.
func runRawChatCompletion(body string) (string, string) {
	ns, pod, _ := doPost("/v1/chat/completions", body, nil)
	return ns, pod
}

// imageItemJSON builds the JSON fragment for one image_url content block.
func imageItemJSON(url string, i int) string {
	return fmt.Sprintf(`{"type":"image_url","image_url":{"url":%q},"uuid":"image-%d"}`, url, i)
}

// videoItemJSON builds the JSON fragment for one video_url content block.
func videoItemJSON(url string) string {
	return fmt.Sprintf(`{"type":"video_url","video_url":{"url":%q}}`, url)
}

// inlineAudioItemJSON builds the JSON fragment for one input_audio content block.
func inlineAudioItemJSON(data string) string {
	return fmt.Sprintf(`{"type":"input_audio","input_audio":{"data":%q,"format":"wav"}}`, data)
}

// imageEmbedsItemJSON builds the JSON fragment for one image_embeds content block.
func imageEmbedsItemJSON(embeds string, i int) string {
	return fmt.Sprintf(`{"type":"image_embeds","image_embeds":%q,"uuid":"embedded-image-%d"}`, embeds, i)
}

// chatCompletionBody assembles a chat-completions request body from pre-built
// content-item JSON fragments plus a trailing text prompt. maxTokens is
// omitted when zero.
func chatCompletionBody(prompt string, maxTokens int, items []string) string {
	prefix := strings.Join(items, ",")
	if len(items) > 0 {
		prefix += ","
	}
	body := fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":[%s{"type":"text","text":%q}]}]`,
		simModelName, prefix, prompt)
	if maxTokens > 0 {
		body += fmt.Sprintf(`,"max_tokens":%d`, maxTokens)
	}
	return body + "}"
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
	items := make([]string, len(imageURLs))
	for i, u := range imageURLs {
		items[i] = imageItemJSON(u, i)
	}
	return runRawChatCompletion(chatCompletionBody("Describe what you see.", 150, items))
}

// runChatCompletionWithImageEmbeds sends a chat completion request with an image_embeds content block
// carrying a pre-encoded tensor. image_embeds is not a recognised multimodal type for encode
// disaggregation, so the request routes like a text request (decode-only or prefill-decode).
// Returns the namespace and pod name from the response headers.
func runChatCompletionWithImageEmbeds() (string, string) {
	ginkgo.By("Sending Chat Completion Request with image_embeds")
	return runRawChatCompletion(chatCompletionBody("Describe this embedded image:", 0, []string{imageEmbedsItemJSON(testImageEmbeds, 1)}))
}

// runChatCompletionWithVideos sends a multimodal chat completion request with one or more
// video_url content blocks. When called with no arguments it defaults to testVideoURL.
// Returns the namespace and pod name from the response headers.
func runChatCompletionWithVideos(videoURLs ...string) (string, string) {
	if len(videoURLs) == 0 {
		videoURLs = []string{testVideoURL}
	}
	ginkgo.By(fmt.Sprintf("Sending Multimodal Chat Completion Request with %d videos", len(videoURLs)))
	items := make([]string, len(videoURLs))
	for i, u := range videoURLs {
		items[i] = videoItemJSON(u)
	}
	return runRawChatCompletion(chatCompletionBody("What is happening in these videos?", 0, items))
}

// runChatCompletionWithAudios sends a multimodal chat completion request with one or more
// input_audio content blocks. When called with no arguments it defaults to testAudioData.
// Returns the namespace and pod name from the response headers.
func runChatCompletionWithAudios(audioClips ...string) (string, string) {
	if len(audioClips) == 0 {
		audioClips = []string{testAudioData}
	}
	ginkgo.By(fmt.Sprintf("Sending Multimodal Chat Completion Request with %d audio clips", len(audioClips)))
	items := make([]string, len(audioClips))
	for i, d := range audioClips {
		items[i] = inlineAudioItemJSON(d)
	}
	return runRawChatCompletion(chatCompletionBody("What is being said in these audio clips?", 100, items))
}

// runChatCompletionWithMixedMedia sends a multimodal chat completion request combining
// image_url, input_audio, and video_url content blocks in a single request.
// Returns the namespace and pod name from the response headers.
func runChatCompletionWithMixedMedia(imageURLs, audioClips, videoURLs []string) (string, string) {
	ginkgo.By(fmt.Sprintf("Sending Mixed-Media Chat Completion Request: %d images + %d audio + %d video",
		len(imageURLs), len(audioClips), len(videoURLs)))
	items := make([]string, 0, len(imageURLs)+len(audioClips)+len(videoURLs))
	for i, u := range imageURLs {
		items = append(items, imageItemJSON(u, i))
	}
	for _, d := range audioClips {
		items = append(items, inlineAudioItemJSON(d))
	}
	for _, u := range videoURLs {
		items = append(items, videoItemJSON(u))
	}
	return runRawChatCompletion(chatCompletionBody("Describe everything you see and hear.", 150, items))
}

func runStreamingCompletion(prompt string, theModel openai.CompletionNewParamsModel) (string, string) {
	ginkgo.By(fmt.Sprintf("Sending Streaming Completion Request: (port %d) model=%s", getPort(), theModel))
	body := fmt.Sprintf(`{"model":"%s","prompt":"%s","max_tokens":50,"stream":true}`, theModel, prompt)
	ns, pod, respBody := doPost("/v1/completions", body, nil)
	ginkgo.By(fmt.Sprintf("Streaming Completion received response length: %d bytes", len(respBody)))
	return ns, pod
}

func runStreamingChatCompletion(prompt string) (string, string) {
	ginkgo.By(fmt.Sprintf("Sending Streaming Chat Completion Request: (port %d)", getPort()))
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

func verifyMetrics(infPoolName string, numTargetPorts int) {

	generateAndCheckLoad(10)

	// Send a few errors
	for range 10 {
		doPostWithError("/v1/chat/completions", "an invalid body", nil)
	}

	metricsURL := fmt.Sprintf("http://localhost:%d/metrics", getMetricsPort())

	startEPPMetricsPortForward()

	theMetrics := getMetrics(metricsURL)
	gomega.Expect(theMetrics).ShouldNot(gomega.BeEmpty())
	metricsAsString := strings.Join(theMetrics, "\n")

	_, decodePods := getModelServerPods(podSelector, prefillSelector, decodeSelector, getNamespace())

	// Define the metrics we expect to see
	preset := []string{ //nolint:prealloc
		"inference_objective_request_total",
		"inference_objective_request_error_total",
		"inference_objective_request_duration_seconds",
		"inference_objective_normalized_time_per_output_token_seconds",
		"inference_objective_request_sizes",
		"inference_objective_response_sizes",
		"inference_objective_input_tokens",
		"inference_objective_output_tokens",
		"inference_pool_average_kv_cache_utilization",
		"inference_pool_average_queue_size",
		"inference_pool_per_pod_queue_size",
		"inference_objective_running_requests",
		"inference_pool_ready_pods",
		"inference_extension_info",

		// llm_d metrics
		"llm_d_epp_request_total",
		"llm_d_epp_request_error_total",
		"llm_d_epp_request_duration_seconds",
		"llm_d_epp_request_ntpot_seconds",
		"llm_d_epp_request_size_bytes",
		"llm_d_epp_response_size_bytes",
		"llm_d_epp_request_input_tokens",
		"llm_d_epp_request_output_tokens",
		"llm_d_epp_average_kv_cache_utilization",
		"llm_d_epp_average_queue_size",
		"llm_d_epp_per_endpoint_queue_size",
		"llm_d_epp_request_running",
		"llm_d_epp_ready_endpoints",
		"llm_d_epp_info",
		"llm_d_epp_request_processing_duration_seconds",
		"llm_d_epp_response_processing_duration_seconds",
	}
	expectedMetrics := make([]string, 0, len(preset)+len(decodePods)*numTargetPorts*2)
	expectedMetrics = append(expectedMetrics, preset...)

	for _, modelServerPodName := range decodePods {
		for rank := range numTargetPorts {
			metricQueueSize := fmt.Sprintf(
				"inference_pool_per_pod_queue_size{model_server_pod=\"%s-rank-%d\",name=\"%s\"}",
				modelServerPodName,
				rank,
				infPoolName)
			expectedMetrics = append(expectedMetrics, metricQueueSize)

			metricQueueSizeNew := fmt.Sprintf(
				"llm_d_epp_per_endpoint_queue_size{model_server_endpoint=\"%s-rank-%d\",name=\"%s\"}",
				modelServerPodName,
				rank,
				infPoolName,
			)
			expectedMetrics = append(expectedMetrics, metricQueueSizeNew)
		}
	}

	// Check if all expected metrics are present in the metrics output.
	for _, metric := range expectedMetrics {
		gomega.Expect(metricsAsString).Should(gomega.ContainSubstring(metric))
	}
}
