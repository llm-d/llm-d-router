package e2e

import (
	"fmt"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	"github.com/llm-d/llm-d-inference-scheduler/pkg/metrics"
	testutils "github.com/llm-d/llm-d-inference-scheduler/test/utils"
)

const (
	// epdDeploymentDir references the Kustomize directory for the non-disaggregated
	// EPD scenario — single deployment, no routing sidecar, vLLM on port 8000
	epdDeploymentDir = "../../deploy/environments/dev/epd"
	// pdDisaggDir references the Kustomize directory for the deployment
	// running vLLM with P/D (connector type is configurable via ${CONNECTOR_TYPE})
	pdDisaggDir = "../../deploy/environments/dev/p-d"
	// ePdDisaggDir references the Kustomize directory for the deployment
	// running vLLM with E/PD (Encode/Prefill-Decode)
	ePdDisaggDir = "../../deploy/environments/dev/e-pd"
	// ePDDisaggDir references the Kustomize directory for the deployment
	// running vLLM with E/P/D (Encode/Prefill/Decode)
	ePDDisaggDir = "../../deploy/environments/dev/e-p-d"

	simplePrompt = "Hello my name is Andrew, I have a doctorate in Rocket Science, and I like interplanetary space exploration"
	extraPrompt  = "Why is the sky sometimes blue and sometimes red close to sunset?"

	// testImageURL and testImageURL2 are architecture diagrams stored in docs/images/ and served via GitHub raw content.
	testImageURL  = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/cat_snow.jpg"
	testImageURL2 = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/flycatcher.jpeg"
	// testVideoURL is a publicly accessible video used in multimodal e2e tests.
	testVideoURL = "https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4"
	// testImageEmbeds is a small dummy base64-encoded tensor used to test image_embeds requests.
	// The actual bytes are not processed by the simulator; only routing behaviour is validated.
	testImageEmbeds = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=="
	// testAudioData is a minimal base64-encoded WAV clip (44-byte header, no samples).
	// The actual bytes are not processed by the simulator; only routing behaviour is validated.
	testAudioData = "UklGRiQAAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YQAAAAA="
)

var (
	poolName              = simModelName + "-inference-pool"
	podSelector           = map[string]string{"app": poolName}
	prefillSelector       = map[string]string{"llm-d.ai/role": "prefill"}
	decodeSelector        = map[string]string{"llm-d.ai/role": "decode"}
	prefillDecodeSelector = map[string]string{"llm-d.ai/role": "prefill-decode"}
	encodeSelector        = map[string]string{"llm-d.ai/role": "encode"}
	epdSingleSelector     = map[string]string{"llm-d.ai/role": "encode-prefill-decode"}
)

var _ = ginkgo.Describe("Run end to end tests", ginkgo.Ordered, func() {
	ginkgo.When("Running simple non-PD configuration", func() {
		ginkgo.It("should run successfully", func() {
			infPoolObjects = createInferencePool(1, true)

			modelServers := createModelServersDecode(1)

			epp := createEndPointPicker(simpleConfig)

			prefillPods, decodePods := getModelServerPods(podSelector, prefillSelector, decodeSelector)
			gomega.Expect(prefillPods).Should(gomega.BeEmpty())
			gomega.Expect(decodePods).Should(gomega.HaveLen(1))

			nsHdr, podHdr, _ := runCompletion(simplePrompt, simModelName)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.Equal(decodePods[0]))

			nsHdr, podHdr, _ = runChatCompletion(simplePrompt, simModelName)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.Equal(decodePods[0]))

			testutils.DeleteObjects(testConfig, epp)
			testutils.DeleteObjects(testConfig, modelServers)
		})
	})

	ginkgo.When("Running a PD configuration (deprecated pd-profile-handler)", func() {
		ginkgo.It("should run successfully", func() {
			infPoolObjects = createInferencePool(1, true)

			prefillReplicas := 1
			decodeReplicas := 4
			modelServers := createModelServersPDNixl(prefillReplicas, decodeReplicas)

			epp := createEndPointPicker(deprecatedPdConfig)

			metricsURL := fmt.Sprintf("http://localhost:%s/metrics", metricsPort)

			if k8sContext != "" {
				// Use port-forward to access the EPP pod's metrics endpoint.
				startEPPMetricsPortForward()
			}

			prefillPods, decodePods := getModelServerPods(podSelector, prefillSelector, decodeSelector)
			gomega.Expect(prefillPods).Should(gomega.HaveLen(prefillReplicas))
			gomega.Expect(decodePods).Should(gomega.HaveLen(decodeReplicas))

			nsHdr, podHdrCompletion, _ := runCompletion(simplePrompt, simModelName)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdrCompletion).Should(gomega.BeElementOf(decodePods))

			nsHdr, podHdrChat, _ := runChatCompletion(simplePrompt, simModelName)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdrChat).Should(gomega.BeElementOf(decodePods))

			// Do an extra completion call with a different prompt
			nsHdr, podHdr, _ := runCompletion(extraPrompt, simModelName)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.BeElementOf(decodePods))

			// Run completion with the original prompt
			nsHdr, podHdr, _ = runCompletion(simplePrompt, simModelName)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.BeElementOf(decodePods))
			gomega.Expect(podHdr).Should(gomega.Equal(podHdrCompletion))

			// Do an extra chat completion call with a different prompt
			nsHdr, podHdr, _ = runChatCompletion(extraPrompt, simModelName)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.BeElementOf(decodePods))

			// Run chat completion with the original prompt
			nsHdr, podHdr, _ = runChatCompletion(simplePrompt, simModelName)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.BeElementOf(decodePods))
			gomega.Expect(podHdr).Should(gomega.Equal(podHdrChat))

			// Metrics Validation
			labelFilter := fmt.Sprintf(`decision_type=%q,model_name="%s"`, metrics.DecisionTypePrefillDecode, simModelName)
			prefillDecodeCount := getCounterMetric(metricsURL, "llm_d_inference_scheduler_pd_decision_total", labelFilter)
			prefillDecodeCountllm_d_router_epp := getCounterMetric(metricsURL, "llm_d_router_epp_pd_decision_total", labelFilter)

			labelFilter2 := fmt.Sprintf(`decision_type=%q,model_name="%s"`, metrics.DecisionTypeDecodeOnly, simModelName)
			decodeOnlyCount := getCounterMetric(metricsURL, "llm_d_inference_scheduler_pd_decision_total", labelFilter2)
			decodeOnlyCountllm_d_router_epp := getCounterMetric(metricsURL, "llm_d_router_epp_pd_decision_total", labelFilter2)

			gomega.Expect(prefillDecodeCount).Should(gomega.Equal(4))
			gomega.Expect(prefillDecodeCountllm_d_router_epp).Should(gomega.Equal(4))
			gomega.Expect(decodeOnlyCount).Should(gomega.Equal(2))
			gomega.Expect(decodeOnlyCountllm_d_router_epp).Should(gomega.Equal(2))

			testutils.DeleteObjects(testConfig, epp)
			testutils.DeleteObjects(testConfig, modelServers)
		})
	})

	for _, tc := range []struct {
		name   string
		config string
	}{
		{"deprecated pd-profile-handler", deprecatedPdConfig},
		{"disagg-profile-handler", pdConfig},
	} {
		config := tc.config // capture for closure
		ginkgo.When("Running a PD configuration with shared-storage connector using "+tc.name, func() {
			ginkgo.It("should run regular (non-streaming) requests successfully", func() {
				infPoolObjects = createInferencePool(1, true)

				prefillReplicas := 1
				decodeReplicas := 2
				modelServers := createModelServersPDSharedStorage(decodeReplicas)

				epp := createEndPointPicker(config)

				prefillPods, decodePods := getModelServerPods(podSelector, prefillSelector, decodeSelector)
				gomega.Expect(prefillPods).Should(gomega.HaveLen(prefillReplicas))
				gomega.Expect(decodePods).Should(gomega.HaveLen(decodeReplicas))

				// Test regular completion request
				nsHdr, podHdrCompletion, _ := runCompletion(simplePrompt, simModelName)
				gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
				gomega.Expect(podHdrCompletion).Should(gomega.BeElementOf(decodePods))

				// Test regular chat completion request
				nsHdr, podHdrChat, _ := runChatCompletion(simplePrompt, simModelName)
				gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
				gomega.Expect(podHdrChat).Should(gomega.BeElementOf(decodePods))

				// Run completion with a different prompt
				nsHdr, podHdr, _ := runCompletion(extraPrompt, simModelName)
				gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
				gomega.Expect(podHdr).Should(gomega.BeElementOf(decodePods))

				// Run completion with original prompt (should go to same pod due to prefix cache)
				nsHdr, podHdr, _ = runCompletion(simplePrompt, simModelName)
				gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
				gomega.Expect(podHdr).Should(gomega.BeElementOf(decodePods))
				gomega.Expect(podHdr).Should(gomega.Equal(podHdrCompletion))

				testutils.DeleteObjects(testConfig, epp)
				testutils.DeleteObjects(testConfig, modelServers)
			})

			ginkgo.It("should run streaming requests successfully", func() {
				infPoolObjects = createInferencePool(1, true)

				prefillReplicas := 1
				decodeReplicas := 2
				modelServers := createModelServersPDSharedStorage(decodeReplicas)

				epp := createEndPointPicker(config)

				prefillPods, decodePods := getModelServerPods(podSelector, prefillSelector, decodeSelector)
				gomega.Expect(prefillPods).Should(gomega.HaveLen(prefillReplicas))
				gomega.Expect(decodePods).Should(gomega.HaveLen(decodeReplicas))

				// Test streaming completion request
				nsHdr, podHdr := runStreamingCompletion(simplePrompt, simModelName)
				gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
				gomega.Expect(podHdr).Should(gomega.BeElementOf(decodePods))

				// Test streaming chat completion request
				nsHdr, podHdr = runStreamingChatCompletion(simplePrompt)
				gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
				gomega.Expect(podHdr).Should(gomega.BeElementOf(decodePods))

				// Run streaming completion with a different prompt
				nsHdr, podHdr = runStreamingCompletion(extraPrompt, simModelName)
				gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
				gomega.Expect(podHdr).Should(gomega.BeElementOf(decodePods))

				testutils.DeleteObjects(testConfig, epp)
				testutils.DeleteObjects(testConfig, modelServers)
			})

			ginkgo.It("should handle decode-first success scenario with cache_hit_threshold", func() {
				// This test verifies the decode-first optimization:
				// When cache_hit_threshold is set and the decode succeeds (cache hit),
				// the request should complete without falling back to P/D.
				// IMPORTANT: The prefill pod should NOT process any requests in this scenario.
				infPoolObjects = createInferencePool(1, true)

				prefillReplicas := 1
				decodeReplicas := 2
				modelServers := createModelServersPDSharedStorage(decodeReplicas)

				epp := createEndPointPicker(config)

				prefillPods, decodePods := getModelServerPods(podSelector, prefillSelector, decodeSelector)
				gomega.Expect(prefillPods).Should(gomega.HaveLen(prefillReplicas))
				gomega.Expect(decodePods).Should(gomega.HaveLen(decodeReplicas))

				// Get prefill request count BEFORE the test
				prefillCountBefore := getPodRequestCount(prefillPods[0])
				ginkgo.By(fmt.Sprintf("Prefill request count before decode-first test: %d", prefillCountBefore))

				// Test decode-first success: cache_hit_threshold is set, but simulator returns "stop"
				// (without X-Cache-Threshold header), meaning decode succeeded without prefill
				nsHdr, podHdr, finishReason := runCompletionWithCacheThreshold(simplePrompt, 0.5, false)
				gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
				gomega.Expect(podHdr).Should(gomega.BeElementOf(decodePods))
				gomega.Expect(finishReason).ShouldNot(gomega.Equal("cache_threshold"))

				// Test streaming decode-first success
				nsHdr, podHdr, finishReason = runStreamingCompletionWithCacheThreshold(simplePrompt, 0.5, false)
				gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
				gomega.Expect(podHdr).Should(gomega.BeElementOf(decodePods))
				gomega.Expect(finishReason).ShouldNot(gomega.Equal("cache_threshold"))

				// Get prefill request count AFTER the test
				prefillCountAfter := getPodRequestCount(prefillPods[0])
				ginkgo.By(fmt.Sprintf("Prefill request count after decode-first test: %d", prefillCountAfter))

				// VERIFY: Prefill pod should NOT have processed any new requests
				// (decode-first succeeded, so no P/D fallback occurred)
				gomega.Expect(prefillCountAfter).Should(gomega.Equal(prefillCountBefore),
					"Prefill pod should NOT process requests when cache threshold is met (decode-first success)")

				testutils.DeleteObjects(testConfig, epp)
				testutils.DeleteObjects(testConfig, modelServers)
			})

			ginkgo.It("should handle decode-first fallback to P/D when cache threshold not met", func() {
				// This test verifies the decode-first fallback scenario:
				// When cache_hit_threshold is set and the decode returns cache_threshold finish_reason,
				// the sidecar should fall back to P/D disaggregation.
				// IMPORTANT: The prefill pod SHOULD process requests in this scenario.
				infPoolObjects = createInferencePool(1, true)

				prefillReplicas := 1
				decodeReplicas := 2
				modelServers := createModelServersPDSharedStorage(decodeReplicas)

				epp := createEndPointPicker(config)

				prefillPods, decodePods := getModelServerPods(podSelector, prefillSelector, decodeSelector)
				gomega.Expect(prefillPods).Should(gomega.HaveLen(prefillReplicas))
				gomega.Expect(decodePods).Should(gomega.HaveLen(decodeReplicas))

				// Get prefill request count BEFORE the test
				prefillCountBefore := getPodRequestCount(prefillPods[0])
				ginkgo.By(fmt.Sprintf("Prefill request count before P/D fallback test: %d", prefillCountBefore))

				// Test decode-first fallback: cache_hit_threshold is set AND X-Cache-Threshold header
				// forces simulator to return "cache_threshold" finish_reason, triggering P/D fallback
				nsHdr, podHdr, finishReason := runCompletionWithCacheThreshold(simplePrompt, 0.5, true)
				gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
				gomega.Expect(podHdr).Should(gomega.BeElementOf(decodePods))
				// The sidecar completes the P/D flow but returns cache_threshold as the finish_reason
				// from the initial decode attempt (which triggered the fallback)
				gomega.Expect(finishReason).Should(gomega.Equal("cache_threshold"))

				// Test streaming decode-first fallback
				nsHdr, podHdr, finishReason = runStreamingCompletionWithCacheThreshold(extraPrompt, 0.5, true)
				gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
				gomega.Expect(podHdr).Should(gomega.BeElementOf(decodePods))
				gomega.Expect(finishReason).Should(gomega.Equal("cache_threshold"))

				// Get prefill request count AFTER the test
				prefillCountAfter := getPodRequestCount(prefillPods[0])
				ginkgo.By(fmt.Sprintf("Prefill request count after P/D fallback test: %d", prefillCountAfter))

				// VERIFY: Prefill pod SHOULD have processed 2 new requests (1 regular + 1 streaming)
				// (decode-first failed, so P/D fallback occurred and prefill was invoked)
				gomega.Expect(prefillCountAfter).Should(gomega.BeNumerically(">", prefillCountBefore),
					"Prefill pod SHOULD process requests when cache threshold is NOT met (P/D fallback)")
				gomega.Expect(prefillCountAfter-prefillCountBefore).Should(gomega.Equal(2),
					"Prefill pod should have processed exactly 2 requests (1 regular + 1 streaming)")

				testutils.DeleteObjects(testConfig, epp)
				testutils.DeleteObjects(testConfig, modelServers)
			})
		})
	}

	ginkgo.When("Running a PD configuration with disagg-profile-handler and metrics validation", func() {
		ginkgo.It("should run successfully", func() {
			infPoolObjects = createInferencePool(1, true)

			prefillReplicas := 1
			decodeReplicas := 4
			modelServers := createModelServersPDSharedStorage(decodeReplicas)

			epp := createEndPointPicker(pdConfig)

			metricsURL := fmt.Sprintf("http://localhost:%s/metrics", metricsPort)

			if k8sContext != "" {
				// Use port-forward to access the EPP pod's metrics endpoint.
				startEPPMetricsPortForward()
			}

			prefillPods, decodePods := getModelServerPods(podSelector, prefillSelector, decodeSelector)
			gomega.Expect(prefillPods).Should(gomega.HaveLen(prefillReplicas))
			gomega.Expect(decodePods).Should(gomega.HaveLen(decodeReplicas))

			nsHdr, podHdrCompletion, _ := runCompletion(simplePrompt, simModelName)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdrCompletion).Should(gomega.BeElementOf(decodePods))

			nsHdr, podHdrChat, _ := runChatCompletion(simplePrompt, simModelName)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdrChat).Should(gomega.BeElementOf(decodePods))

			// Do an extra completion call with a different prompt
			nsHdr, podHdr, _ := runCompletion(extraPrompt, simModelName)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.BeElementOf(decodePods))

			// Run completion with the original prompt
			nsHdr, podHdr, _ = runCompletion(simplePrompt, simModelName)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.BeElementOf(decodePods))
			gomega.Expect(podHdr).Should(gomega.Equal(podHdrCompletion))

			// Do an extra chat completion call with a different prompt
			nsHdr, podHdr, _ = runChatCompletion(extraPrompt, simModelName)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.BeElementOf(decodePods))

			// Run chat completion with the original prompt
			nsHdr, podHdr, _ = runChatCompletion(simplePrompt, simModelName)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.BeElementOf(decodePods))
			gomega.Expect(podHdr).Should(gomega.Equal(podHdrChat))

			// Metrics Validation
			labelFilter := fmt.Sprintf(`decision_type=%q,model_name="%s"`, metrics.DecisionTypePrefillDecode, simModelName)
			prefillDecodeCount := getCounterMetric(metricsURL, "llm_d_inference_scheduler_disagg_decision_total", labelFilter)
			prefillDecodeCountllm_d_router_epp := getCounterMetric(metricsURL, "llm_d_router_epp_disagg_decision_total", labelFilter)

			labelFilter2 := fmt.Sprintf(`decision_type=%q,model_name="%s"`, metrics.DecisionTypeDecodeOnly, simModelName)
			decodeOnlyCount := getCounterMetric(metricsURL, "llm_d_inference_scheduler_disagg_decision_total", labelFilter2)
			decodeOnlyCountllm_d_router_epp := getCounterMetric(metricsURL, "llm_d_router_epp_disagg_decision_total", labelFilter2)

			gomega.Expect(prefillDecodeCount).Should(gomega.Equal(4))
			gomega.Expect(prefillDecodeCountllm_d_router_epp).Should(gomega.Equal(4))
			gomega.Expect(decodeOnlyCount).Should(gomega.Equal(2))
			gomega.Expect(decodeOnlyCountllm_d_router_epp).Should(gomega.Equal(2))

			testutils.DeleteObjects(testConfig, epp)
			testutils.DeleteObjects(testConfig, modelServers)
		})
	})

	ginkgo.When("Running simple non-PD configuration with disagg-profile-handler", func() {
		ginkgo.It("should run successfully", func() {
			infPoolObjects = createInferencePool(1, true)

			modelServers := createModelServersDecode(1)

			epp := createEndPointPicker(decodeOnlyConfig)

			prefillPods, decodePods := getModelServerPods(podSelector, prefillSelector, decodeSelector)
			gomega.Expect(prefillPods).Should(gomega.BeEmpty())
			gomega.Expect(decodePods).Should(gomega.HaveLen(1))

			nsHdr, podHdr, _ := runCompletion(simplePrompt, simModelName)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.Equal(decodePods[0]))

			nsHdr, podHdr, _ = runChatCompletion(simplePrompt, simModelName)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.Equal(decodePods[0]))

			testutils.DeleteObjects(testConfig, epp)
			testutils.DeleteObjects(testConfig, modelServers)
		})
	})

	ginkgo.When("Running an E/PD (Encode/Prefill-Decode) configuration", func() {
		ginkgo.It("should route multimodal requests through encode and decode pods", func() {
			infPoolObjects = createInferencePool(1, true)

			encodeReplicas := 2
			decodeReplicas := 1
			modelServers := createModelServersEpDDisagg(encodeReplicas, decodeReplicas)

			epp := createEndPointPicker(epdEncodeDecodeConfig)

			metricsURL := fmt.Sprintf("http://localhost:%s/metrics", metricsPort)
			if k8sContext != "" {
				startEPPMetricsPortForward()
			}

			encodePods := getPodNames(encodeSelector)
			prefillDecodePods := getPodNames(prefillDecodeSelector)
			gomega.Expect(encodePods).Should(gomega.HaveLen(encodeReplicas))
			gomega.Expect(prefillDecodePods).Should(gomega.HaveLen(decodeReplicas))

			// Text request: encode stage skipped, routed directly to a prefill-decode pod
			nsHdr, podHdr, _ := runCompletion(simplePrompt, simModelName)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.BeElementOf(prefillDecodePods))

			// Multimodal request: triggers encode stage, decode handled by prefill-decode pod
			nsHdr, podHdr = runChatCompletionWithImages(testImageURL)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.BeElementOf(prefillDecodePods))

			nsHdr, podHdr = runChatCompletionWithImages(testImageURL)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.BeElementOf(prefillDecodePods))

			// Multi-image request: two images in one request, triggers encode stage
			nsHdr, podHdr = runChatCompletionWithImages(testImageURL, testImageURL2)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.BeElementOf(prefillDecodePods))

			// Video request: video_url triggers encode stage, decode handled by prefill-decode pod
			nsHdr, podHdr = runChatCompletionWithVideo()
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.BeElementOf(prefillDecodePods))

			// Audio request: input_audio triggers encode stage, decode handled by prefill-decode pod
			nsHdr, podHdr = runChatCompletionWithAudio()
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.BeElementOf(prefillDecodePods))

			// image_embeds request: pre-encoded tensor, encode stage skipped, routes to prefill-decode pod
			nsHdr, podHdr = runChatCompletionWithImageEmbeds()
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.BeElementOf(prefillDecodePods))

			// Metrics: text + image_embeds requests recorded as decode-only (encode skipped)
			decodeOnlyFilter := fmt.Sprintf(`decision_type=%q,model_name="%s"`, metrics.DecisionTypeDecodeOnly, simModelName)
			decodeOnlyCount := getCounterMetric(metricsURL, "llm_d_inference_scheduler_disagg_decision_total", decodeOnlyFilter)
			decodeOnlyCountllm_d_router_epp := getCounterMetric(metricsURL, "llm_d_router_epp_disagg_decision_total", decodeOnlyFilter)
			gomega.Expect(decodeOnlyCount).Should(gomega.Equal(2))
			gomega.Expect(decodeOnlyCountllm_d_router_epp).Should(gomega.Equal(2))

			// Metrics: encode-decode decisions recorded (2 single-image + 1 multi-image + 1 video + 1 audio)
			labelFilter := fmt.Sprintf(`decision_type=%q,model_name="%s"`, metrics.DecisionTypeEncodeDecode, simModelName)
			encodeDecodeCount := getCounterMetric(metricsURL, "llm_d_inference_scheduler_disagg_decision_total", labelFilter)
			encodeDecodeCountllm_d_router_epp := getCounterMetric(metricsURL, "llm_d_router_epp_disagg_decision_total", labelFilter)
			gomega.Expect(encodeDecodeCount).Should(gomega.Equal(5))
			gomega.Expect(encodeDecodeCountllm_d_router_epp).Should(gomega.Equal(5))

			testutils.DeleteObjects(testConfig, epp)
			testutils.DeleteObjects(testConfig, modelServers)
		})
	})

	ginkgo.When("Running an E/P/D (encode/prefill/decode) configuration", func() {
		ginkgo.It("should route multimodal requests through encode, prefill, and decode pods", func() {
			infPoolObjects = createInferencePool(1, true)

			encodeReplicas := 2
			prefillReplicas := 1
			decodeReplicas := 1
			modelServers := createModelServersEPDDisagg(encodeReplicas, prefillReplicas, decodeReplicas)

			epp := createEndPointPicker(epdConfig)

			metricsURL := fmt.Sprintf("http://localhost:%s/metrics", metricsPort)
			if k8sContext != "" {
				startEPPMetricsPortForward()
			}

			encodePods := getPodNames(encodeSelector)
			prefillPods := getPodNames(prefillSelector)
			decodePods := getPodNames(decodeSelector)
			gomega.Expect(encodePods).Should(gomega.HaveLen(encodeReplicas))
			gomega.Expect(prefillPods).Should(gomega.HaveLen(prefillReplicas))
			gomega.Expect(decodePods).Should(gomega.HaveLen(decodeReplicas))

			// Text request: encode stage skipped, prefill triggered by prefix-based-pd-decider
			nsHdr, podHdr, _ := runCompletion(simplePrompt, simModelName)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.BeElementOf(decodePods))

			// First multimodal request: encode + prefill + decode
			nsHdr, podHdr = runChatCompletionWithImages(testImageURL)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.BeElementOf(decodePods))

			// Second multimodal request with same image (prefix cache may skip prefill)
			nsHdr, podHdr = runChatCompletionWithImages(testImageURL)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.BeElementOf(decodePods))

			// Multi-image request: two images in one request, encode + prefill + decode
			nsHdr, podHdr = runChatCompletionWithImages(testImageURL, testImageURL2)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.BeElementOf(decodePods))

			// Video request: video_url triggers encode stage, decode handled by decode pod
			nsHdr, podHdr = runChatCompletionWithVideo()
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.BeElementOf(decodePods))

			// image_embeds request: pre-encoded tensor, encode stage skipped, routes to decode pod
			nsHdr, podHdr = runChatCompletionWithImageEmbeds()
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.BeElementOf(decodePods))

			// Metrics: text + image_embeds requests recorded as decode-only or prefill-decode (encode skipped)
			pdLabelFilter := fmt.Sprintf(`decision_type=%q,model_name="%s"`, metrics.DecisionTypePrefillDecode, simModelName)
			doLabelFilter := fmt.Sprintf(`decision_type=%q,model_name="%s"`, metrics.DecisionTypeDecodeOnly, simModelName)
			pdCount := getCounterMetric(metricsURL, "llm_d_inference_scheduler_disagg_decision_total", pdLabelFilter)
			pdCountllm_d_router_epp := getCounterMetric(metricsURL, "llm_d_router_epp_disagg_decision_total", pdLabelFilter)
			doCount := getCounterMetric(metricsURL, "llm_d_inference_scheduler_disagg_decision_total", doLabelFilter)
			doCountllm_d_router_epp := getCounterMetric(metricsURL, "llm_d_router_epp_disagg_decision_total", doLabelFilter)
			gomega.Expect(pdCount + doCount).Should(gomega.Equal(2))
			gomega.Expect(pdCountllm_d_router_epp + doCountllm_d_router_epp).Should(gomega.Equal(2))

			// Metrics: 4 multimodal requests each produce either encode-prefill-decode or encode-decode
			// (encode-decode occurs if the prefix cache hits on the second same-image request).
			// The 3 requests with unique content (1st image, multi-image, video) always produce encode-prefill-decode.
			epdLabelFilter := fmt.Sprintf(`decision_type=%q,model_name="%s"`, metrics.DecisionTypeEncodePrefillDecode, simModelName)
			edLabelFilter := fmt.Sprintf(`decision_type=%q,model_name="%s"`, metrics.DecisionTypeEncodeDecode, simModelName)
			epdCount := getCounterMetric(metricsURL, "llm_d_inference_scheduler_disagg_decision_total", epdLabelFilter)
			epdCountllm_d_router_epp := getCounterMetric(metricsURL, "llm_d_router_epp_disagg_decision_total", epdLabelFilter)
			edCount := getCounterMetric(metricsURL, "llm_d_inference_scheduler_disagg_decision_total", edLabelFilter)
			edCountllm_d_router_epp := getCounterMetric(metricsURL, "llm_d_router_epp_disagg_decision_total", edLabelFilter)
			gomega.Expect(epdCount).Should(gomega.BeNumerically(">=", 3))
			gomega.Expect(epdCountllm_d_router_epp).Should(gomega.BeNumerically(">=", 3))
			gomega.Expect(epdCount + edCount).Should(gomega.Equal(4))
			gomega.Expect(epdCountllm_d_router_epp + edCountllm_d_router_epp).Should(gomega.Equal(4))

			testutils.DeleteObjects(testConfig, epp)
			testutils.DeleteObjects(testConfig, modelServers)
		})
	})

	ginkgo.When("Running an EPD (no disaggregation) configuration", func() {
		ginkgo.It("should route text and multimodal requests to the single deployment", func() {
			infPoolObjects = createInferencePool(1, true)

			// Single deployment labeled encode-prefill-decode: matches encode-filter, prefill-filter,
			// and decode-filter, so all EPD stages are handled by the same deployment.
			replicas := 1
			modelServers := createModelServersEPDUnified(replicas)

			// Using epdConfig instead of decodeOnlyConfig to validate the EPD logic path within
			// a single pod; multimodal stages will resolve to this same deployment.
			epp := createEndPointPicker(epdConfig)

			metricsURL := fmt.Sprintf("http://localhost:%s/metrics", metricsPort)
			if k8sContext != "" {
				startEPPMetricsPortForward()
			}

			epdPods := getPodNames(epdSingleSelector)
			gomega.Expect(epdPods).Should(gomega.HaveLen(replicas))

			// Text completion: encode skipped, routes to decode profile -> single deployment
			nsHdr, podHdr, _ := runCompletion(simplePrompt, simModelName)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.Equal(epdPods[0]))

			// Text chat completion: same routing as above
			nsHdr, podHdr, _ = runChatCompletion(simplePrompt, simModelName)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.Equal(epdPods[0]))

			// Metrics: text requests recorded as decode-only or prefill-decode (encode skipped)
			pdLabelFilter := fmt.Sprintf(`decision_type=%q,model_name="%s"`, metrics.DecisionTypePrefillDecode, simModelName)
			doLabelFilter := fmt.Sprintf(`decision_type=%q,model_name="%s"`, metrics.DecisionTypeDecodeOnly, simModelName)
			pdCount := getCounterMetric(metricsURL, "llm_d_inference_scheduler_disagg_decision_total", pdLabelFilter)
			pdCountllm_d_router_epp := getCounterMetric(metricsURL, "llm_d_router_epp_disagg_decision_total", pdLabelFilter)
			doCount := getCounterMetric(metricsURL, "llm_d_inference_scheduler_disagg_decision_total", doLabelFilter)
			doCountllm_d_router_epp := getCounterMetric(metricsURL, "llm_d_router_epp_disagg_decision_total", doLabelFilter)
			gomega.Expect(pdCount + doCount).Should(gomega.Equal(2))
			gomega.Expect(pdCountllm_d_router_epp + doCountllm_d_router_epp).Should(gomega.Equal(2))

			// Multimodal request: encode and decode profiles both resolve to the same single deployment
			nsHdr, podHdr = runChatCompletionWithImages(testImageURL)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.Equal(epdPods[0]))

			// Multi-image request: all stages handled by single deployment
			nsHdr, podHdr = runChatCompletionWithImages(testImageURL, testImageURL2)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.Equal(epdPods[0]))

			// Video request: all stages handled by single deployment
			nsHdr, podHdr = runChatCompletionWithVideo()
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.Equal(epdPods[0]))

			// image_embeds request: encode skipped, routes to single deployment
			nsHdr, podHdr = runChatCompletionWithImageEmbeds()
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.Equal(epdPods[0]))

			testutils.DeleteObjects(testConfig, epp)
			testutils.DeleteObjects(testConfig, modelServers)
		})
	})

	ginkgo.When("Running simple non-PD KV enabled configuration", func() {
		ginkgo.It("should run successfully", func() {
			infPoolObjects = createInferencePool(1, true)

			epp := createEndPointPicker(kvConfig)

			modelServers := createModelServersDecodeKV(1)
			time.Sleep(5 * time.Second) // wait for model server(s) to become ready

			prefillPods, decodePods := getModelServerPods(podSelector, prefillSelector, decodeSelector)
			gomega.Expect(prefillPods).Should(gomega.BeEmpty())
			gomega.Expect(decodePods).Should(gomega.HaveLen(1))

			for range 5 {
				nsHdr, podHdr, _ := runCompletion(simplePrompt, kvModelName)
				gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
				gomega.Expect(podHdr).Should(gomega.Equal(decodePods[0]))
			}

			testutils.DeleteObjects(testConfig, epp)
			testutils.DeleteObjects(testConfig, modelServers)
		})
	})

	ginkgo.When("Running KV configuration with external tokenizer DataProducer plugin", func() {
		ginkgo.It("should run successfully", func() {
			infPoolObjects = createInferencePool(1, true)

			epp := createEndPointPicker(kvExternalTokenizerConfig)

			modelServers := createModelServersDecodeKV(1)
			time.Sleep(5 * time.Second) // wait for model server(s) to become ready

			prefillPods, decodePods := getModelServerPods(podSelector, prefillSelector, decodeSelector)
			gomega.Expect(prefillPods).Should(gomega.BeEmpty())
			gomega.Expect(decodePods).Should(gomega.HaveLen(1))

			// Test completions
			nsHdr, podHdr, _ := runCompletion(simplePrompt, kvModelName)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.Equal(decodePods[0]))

			// Test chat completions
			nsHdr, podHdr, _ = runChatCompletion(simplePrompt, kvModelName)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.Equal(decodePods[0]))

			// Repeat to verify prefix cache affinity with pre-tokenized prompts
			for range 3 {
				nsHdr, podHdr, _ = runCompletion(simplePrompt, kvModelName)
				gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
				gomega.Expect(podHdr).Should(gomega.Equal(decodePods[0]))
			}

			testutils.DeleteObjects(testConfig, epp)
			testutils.DeleteObjects(testConfig, modelServers)
		})
	})

	ginkgo.When("Scaling up and down the model servers", func() {
		ginkgo.It("should distribute inference requests across all model servers", func() {
			infPoolObjects = createInferencePool(1, true)

			modelServers := createModelServersDecode(1)

			epp := createEndPointPicker(scaleConfig)

			prefillPods, decodePods := getModelServerPods(podSelector, prefillSelector, decodeSelector)
			gomega.Expect(prefillPods).Should(gomega.BeEmpty())
			gomega.Expect(decodePods).Should(gomega.HaveLen(1))

			var nsHdr, podHdr string
			for range 5 {
				nsHdr, podHdr, _ = runCompletion(simplePrompt, simModelName)
				gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
				gomega.Expect(podHdr).Should(gomega.Equal(decodePods[0]))
			}

			scaleDeployment(modelServers, 1)

			scaledUpPrefillPods, scaledUpDecodePods := getModelServerPods(podSelector, prefillSelector, decodeSelector)
			gomega.Expect(scaledUpPrefillPods).Should(gomega.BeEmpty())
			gomega.Expect(scaledUpDecodePods).Should(gomega.HaveLen(2))

			var scaledNsHdr, scaledPodHdr string
			// Run inference multiple times until one is scheduled on the new pod
			for range 30 {
				scaledNsHdr, scaledPodHdr, _ = runCompletion(extraPrompt, simModelName)
				gomega.Expect(scaledNsHdr).Should(gomega.Equal(nsName))
				gomega.Expect(scaledPodHdr).Should(gomega.BeElementOf(scaledUpDecodePods))
				if scaledPodHdr != podHdr {
					break
				}
			}
			gomega.Expect(scaledPodHdr).ShouldNot(gomega.Equal(podHdr))

			scaleDeployment(modelServers, -1)

			scaledDownPrefillPods, scaledDownDecodePods := getModelServerPods(podSelector, prefillSelector, decodeSelector)
			gomega.Expect(scaledDownPrefillPods).Should(gomega.BeEmpty())
			gomega.Expect(scaledDownDecodePods).Should(gomega.HaveLen(1))
			gomega.Expect(scaledDownDecodePods[0]).Should(gomega.BeElementOf(scaledUpDecodePods))

			// Run multiple times and insure that they are scheduled on the remaining pod
			for range 5 {
				nsHdr, podHdr, _ = runCompletion(simplePrompt, simModelName)
				gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
				gomega.Expect(podHdr).Should(gomega.Equal(scaledDownDecodePods[0]))
			}

			testutils.DeleteObjects(testConfig, epp)
			testutils.DeleteObjects(testConfig, modelServers)
		})
	})

	ginkgo.When("Running a vLLM Data Parallel configuration", func() {
		ginkgo.It("should schedule inference on all ranks", func() {
			infPoolObjects = createInferencePool(2, true)

			modelServers := createModelServersDecodeDP(1)

			epp := createEndPointPicker(dataParallelConfig)

			prefillPods, decodePods := getModelServerPods(podSelector, prefillSelector, decodeSelector)
			gomega.Expect(prefillPods).Should(gomega.BeEmpty())
			gomega.Expect(decodePods).Should(gomega.HaveLen(1))

			nsHdr, podHdr, portHdr := runCompletion(simplePrompt, simModelName)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.Equal(decodePods[0]))

			var parallelNsHdr, parallelPodHdr, parallelPortHdr string

			// Run inference multiple times until one is scheduled on the other port
			for range 30 {
				parallelNsHdr, parallelPodHdr, parallelPortHdr = runCompletion(extraPrompt, simModelName)
				gomega.Expect(parallelNsHdr).Should(gomega.Equal(nsName))
				gomega.Expect(parallelPodHdr).Should(gomega.Equal(decodePods[0]))
				if parallelPortHdr != portHdr {
					break
				}
			}
			gomega.Expect(parallelPortHdr).ShouldNot(gomega.Equal(portHdr))

			nsHdr, podHdr, portHdr = runChatCompletion(simplePrompt, simModelName)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.Equal(decodePods[0]))

			// Run inference multiple times until one is scheduled on the other port
			for range 30 {
				parallelNsHdr, parallelPodHdr, parallelPortHdr = runChatCompletion(extraPrompt, simModelName)
				gomega.Expect(parallelNsHdr).Should(gomega.Equal(nsName))
				gomega.Expect(parallelPodHdr).Should(gomega.Equal(decodePods[0]))
				if parallelPortHdr != portHdr {
					break
				}
			}
			gomega.Expect(parallelPortHdr).ShouldNot(gomega.Equal(portHdr))

			testutils.DeleteObjects(testConfig, epp)
			testutils.DeleteObjects(testConfig, modelServers)
		})
	})
})
