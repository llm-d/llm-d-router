package e2e

import (
	"fmt"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/disagg"
	"github.com/llm-d/llm-d-router/test/e2e/utils"
	"github.com/llm-d/llm-d-router/test/e2e/utils/standalone"
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

	// encodeOnlyDir is the single-component kustomize path for encode-only pods.
	encodeOnlyDir = "../../deploy/components/vllm-encode"
	// prefillOnlyDir is the single-component kustomize path for prefill-only pods.
	prefillOnlyDir = "../../deploy/components/vllm-prefill"

	simplePrompt = "Hello my name is Andrew, I have a doctorate in Rocket Science, and I like interplanetary space exploration"
	extraPrompt  = "Why is the sky sometimes blue and sometimes red close to sunset?"

	// testImageURL and testImageURL2 are architecture diagrams stored in docs/images/ and served via GitHub raw content.
	testImageURL  = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/cat_snow.jpg"
	testImageURL2 = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/flycatcher.jpeg"
	// testVideoURL is a publicly accessible video used in multimodal e2e tests.
	testVideoURL = "https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4"
	// testVideoURL2 is a second distinct URL string used to exercise multi-video
	// fan-out. The router deduplicates by exact URL string, so a query-string
	// variant is sufficient; the simulator never fetches the URL, so
	// reachability of the query variant does not matter.
	testVideoURL2 = testVideoURL + "?v=2"
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

	singleEmbedding = []string{"The food was delicious and the service was great."}
	doubleEmbedding = []string{"First sentence to embed.", "Second sentence to embed."}
)

var _ = ginkgo.Describe("Run end to end tests", func() {
	ginkgo.When("Running simple non-PD configuration", ginkgo.Ordered, testWrapper(func() {
		ginkgo.It("should run successfully", func() {
			createModelServersDecode(1)

			standalone.Create(standaloneConfig(), simpleConfig, 1, 8000)

			generateAndCheckLoad(5)
		})

		ginkgo.It("should report metrics", func() {
			numTargetPorts := 1

			createModelServersDecode(1)

			router := standalone.Create(standaloneConfig(), simpleConfig, 1, 8000)

			verifyMetrics(router.PoolName, numTargetPorts)
		})

		ginkgo.It("should rewrite the body model name when x-llm-d-model-name-rewrite is set", func() {
			createModelServersDecode(1)

			standalone.Create(standaloneConfig(), simpleConfig, 1, 8000)

			ginkgo.By("Sending a completion with no rewrite header: body model name is forwarded unchanged")
			respModel := runCompletionWithModelRewrite(simplePrompt, simModelName, "")
			gomega.Expect(respModel).Should(gomega.Equal(simModelName))

			ginkgo.By("Sending a completion with the rewrite header: request body is rewritten to the target, response is rewritten back to the client-facing name")
			respModel = runCompletionWithModelRewrite(simplePrompt, "client-facing-name", simModelName)
			gomega.Expect(respModel).Should(gomega.Equal("client-facing-name"))
		})
	}))

	ginkgo.When("Running leader election", ginkgo.Ordered, testWrapper(func() {
		ginkgo.It("Should elect one leader and have other pods as not ready", func() {
			numOfPods := 3

			createModelServersDecode(1)

			router := standalone.Create(standaloneConfig(), simpleConfig, numOfPods, 8000)
			nsName := getNamespace()

			ginkgo.By("Verifying that exactly one EPP pod is ready")
			standalone.WaitForReadyLeader(standaloneConfig(), numOfPods, nsName, router.Selector)
		})

		ginkgo.It("Should successfully failover and serve traffic after the leader pod is deleted", func() {
			numOfPods := 3
			numTargetPorts := 1

			createModelServersDecode(1)

			router := standalone.Create(standaloneConfig(), simpleConfig, numOfPods, 8000)
			nsName := getNamespace()

			ginkgo.By("STEP 1: Verifying initial leader is working correctly before failover")
			leaderPod := standalone.WaitForReadyLeader(standaloneConfig(), numOfPods, nsName, router.Selector)
			generateAndCheckLoad(5)
			verifyMetrics(router.PoolName, numTargetPorts)

			ginkgo.By("Found initial leader pod: " + leaderPod.Name)

			ginkgo.By(fmt.Sprintf("Deleting leader pod %s to trigger failover", leaderPod.Name))
			gomega.Expect(testConfig.K8sClient.Delete(testConfig.Context, leaderPod)).To(gomega.Succeed())

			ginkgo.By("STEP 3: Waiting for a new and different leader to be elected")
			// The deployment controller will create a new pod. We need to wait for the total number of pods
			// to be back to 3, and for one of the other pods to become the new leader.
			var newLeaderPod *corev1.Pod
			gomega.Eventually(func(g gomega.Gomega) {
				newLeaderPod = standalone.WaitForReadyLeader(standaloneConfig(), numOfPods, nsName, router.Selector)
				g.Expect(newLeaderPod.Name).NotTo(gomega.Equal(leaderPod.Name), "The new leader should not be the same as the old deleted leader")
			}, testConfig.ReadyTimeout, testConfig.Interval).Should(gomega.Succeed())
			ginkgo.By("Found new leader pod: " + newLeaderPod.Name)

			ginkgo.By("STEP 4: Verifying the new leader is working correctly after failover")
			router.WaitForRouting()
			generateAndCheckLoad(5)
			verifyMetrics(router.PoolName, numTargetPorts)
		})
	}))

	ginkgo.When("Running a PD configuration with nixlv2 connector and metrics validation", ginkgo.Ordered, testWrapper(func() {
		ginkgo.It("should run successfully", func() {
			prefillReplicas := 1
			decodeReplicas := 4
			createModelServersPDNixlV2(prefillReplicas, decodeReplicas)

			standalone.Create(standaloneConfig(), pdConfig, 1, 8000)
			nsName := getNamespace()

			metricsURL := fmt.Sprintf("http://localhost:%d/metrics", getMetricsPort())

			prefillPods, decodePods := utils.GetModelServerPods(testConfig, podSelector, prefillSelector, decodeSelector, nsName)
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
			labelFilter := fmt.Sprintf(`decision_type=%q,model_name="%s"`, disagg.DecisionTypePrefillDecode, simModelName)
			prefillDecodeCount := utils.GetCounterMetric(metricsURL, "llm_d_epp_disagg_decision_total", labelFilter)

			labelFilter2 := fmt.Sprintf(`decision_type=%q,model_name="%s"`, disagg.DecisionTypeDecodeOnly, simModelName)
			decodeOnlyCount := utils.GetCounterMetric(metricsURL, "llm_d_epp_disagg_decision_total", labelFilter2)

			gomega.Expect(prefillDecodeCount).Should(gomega.Equal(4))
			gomega.Expect(decodeOnlyCount).Should(gomega.Equal(2))
		})
	}))

	for _, tc := range []struct {
		name   string
		config string
	}{
		{"disagg-profile-handler", pdConfig},
	} {
		config := tc.config // capture for closure
		ginkgo.When("Running a PD configuration with shared-storage connector using "+tc.name, ginkgo.Ordered, testWrapper(func() {
			ginkgo.It("should run regular (non-streaming) requests successfully", func() {
				prefillReplicas := 1
				decodeReplicas := 2
				createModelServersPDSharedStorage(decodeReplicas)

				standalone.Create(standaloneConfig(), config, 1, 8000)
				nsName := getNamespace()

				prefillPods, decodePods := utils.GetModelServerPods(testConfig, podSelector, prefillSelector, decodeSelector, nsName)
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
			})

			ginkgo.It("should run streaming requests successfully", func() {
				prefillReplicas := 1
				decodeReplicas := 2
				createModelServersPDSharedStorage(decodeReplicas)

				standalone.Create(standaloneConfig(), config, 1, 8000)
				nsName := getNamespace()

				prefillPods, decodePods := utils.GetModelServerPods(testConfig, podSelector, prefillSelector, decodeSelector, nsName)
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
			})

			ginkgo.It("should handle decode-first success scenario with cache_hit_threshold", func() {
				// This test verifies the decode-first optimization:
				// When cache_hit_threshold is set and the decode succeeds (cache hit),
				// the request should complete without falling back to P/D.
				// IMPORTANT: The prefill pod should NOT process any requests in this scenario.

				prefillReplicas := 1
				decodeReplicas := 2
				createModelServersPDSharedStorage(decodeReplicas)

				standalone.Create(standaloneConfig(), config, 1, 8000)
				nsName := getNamespace()

				prefillPods, decodePods := utils.GetModelServerPods(testConfig, podSelector, prefillSelector, decodeSelector, nsName)
				gomega.Expect(prefillPods).Should(gomega.HaveLen(prefillReplicas))
				gomega.Expect(decodePods).Should(gomega.HaveLen(decodeReplicas))

				// Get prefill request count BEFORE the test
				prefillCountBefore := utils.GetPodRequestCount(testConfig, nsName, prefillPods[0])
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
				prefillCountAfter := utils.GetPodRequestCount(testConfig, nsName, prefillPods[0])
				ginkgo.By(fmt.Sprintf("Prefill request count after decode-first test: %d", prefillCountAfter))

				// VERIFY: Prefill pod should NOT have processed any new requests
				// (decode-first succeeded, so no P/D fallback occurred)
				gomega.Expect(prefillCountAfter).Should(gomega.Equal(prefillCountBefore),
					"Prefill pod should NOT process requests when cache threshold is met (decode-first success)")
			})

			ginkgo.It("should handle decode-first fallback to P/D when cache threshold not met", func() {
				// This test verifies the decode-first fallback scenario:
				// When cache_hit_threshold is set and the decode returns cache_threshold finish_reason,
				// the sidecar should fall back to P/D disaggregation.
				// IMPORTANT: The prefill pod SHOULD process requests in this scenario.

				prefillReplicas := 1
				decodeReplicas := 2
				createModelServersPDSharedStorage(decodeReplicas)

				standalone.Create(standaloneConfig(), config, 1, 8000)
				nsName := getNamespace()

				prefillPods, decodePods := utils.GetModelServerPods(testConfig, podSelector, prefillSelector, decodeSelector, nsName)
				gomega.Expect(prefillPods).Should(gomega.HaveLen(prefillReplicas))
				gomega.Expect(decodePods).Should(gomega.HaveLen(decodeReplicas))

				// Get prefill request count BEFORE the test
				prefillCountBefore := utils.GetPodRequestCount(testConfig, nsName, prefillPods[0])
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
				prefillCountAfter := utils.GetPodRequestCount(testConfig, nsName, prefillPods[0])
				ginkgo.By(fmt.Sprintf("Prefill request count after P/D fallback test: %d", prefillCountAfter))

				// VERIFY: Prefill pod SHOULD have processed 2 new requests (1 regular + 1 streaming)
				// (decode-first failed, so P/D fallback occurred and prefill was invoked)
				gomega.Expect(prefillCountAfter).Should(gomega.BeNumerically(">", prefillCountBefore),
					"Prefill pod SHOULD process requests when cache threshold is NOT met (P/D fallback)")
				gomega.Expect(prefillCountAfter-prefillCountBefore).Should(gomega.Equal(2),
					"Prefill pod should have processed exactly 2 requests (1 regular + 1 streaming)")
			})
		}))
	}

	ginkgo.When("Running a PD configuration with mooncake connector (disagg-profile-handler)", ginkgo.Ordered, testWrapper(func() {
		ginkgo.It("should run regular (non-streaming) requests successfully", func() {
			prefillReplicas := 1
			decodeReplicas := 2
			createModelServersPDMooncake(decodeReplicas)

			standalone.Create(standaloneConfig(), pdConfig, 1, 8000)
			nsName := getNamespace()

			prefillPods, decodePods := utils.GetModelServerPods(testConfig, podSelector, prefillSelector, decodeSelector, nsName)
			gomega.Expect(prefillPods).Should(gomega.HaveLen(prefillReplicas))
			gomega.Expect(decodePods).Should(gomega.HaveLen(decodeReplicas))

			nsHdr, podHdr, _ := runCompletion(simplePrompt, simModelName)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.BeElementOf(decodePods))

			nsHdr, podHdr, _ = runChatCompletion(simplePrompt, simModelName)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.BeElementOf(decodePods))
		})

		ginkgo.It("should run streaming requests successfully", func() {
			prefillReplicas := 1
			decodeReplicas := 2
			createModelServersPDMooncake(decodeReplicas)

			standalone.Create(standaloneConfig(), pdConfig, 1, 8000)
			nsName := getNamespace()

			prefillPods, decodePods := utils.GetModelServerPods(testConfig, podSelector, prefillSelector, decodeSelector, nsName)
			gomega.Expect(prefillPods).Should(gomega.HaveLen(prefillReplicas))
			gomega.Expect(decodePods).Should(gomega.HaveLen(decodeReplicas))

			nsHdr, podHdr := runStreamingCompletion(simplePrompt, simModelName)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.BeElementOf(decodePods))

			nsHdr, podHdr = runStreamingChatCompletion(simplePrompt)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.BeElementOf(decodePods))
		})
	}))

	ginkgo.When("Running a PD configuration with disagg-profile-handler and metrics validation", ginkgo.Ordered, testWrapper(func() {
		ginkgo.It("should run successfully", func() {
			prefillReplicas := 1
			decodeReplicas := 4
			createModelServersPDSharedStorage(decodeReplicas)

			standalone.Create(standaloneConfig(), pdConfig, 1, 8000)
			nsName := getNamespace()

			metricsURL := fmt.Sprintf("http://localhost:%d/metrics", getMetricsPort())

			prefillPods, decodePods := utils.GetModelServerPods(testConfig, podSelector, prefillSelector, decodeSelector, nsName)
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
			labelFilter := fmt.Sprintf(`decision_type=%q,model_name="%s"`, disagg.DecisionTypePrefillDecode, simModelName)
			prefillDecodeCount := utils.GetCounterMetric(metricsURL, "llm_d_epp_disagg_decision_total", labelFilter)

			labelFilter2 := fmt.Sprintf(`decision_type=%q,model_name="%s"`, disagg.DecisionTypeDecodeOnly, simModelName)
			decodeOnlyCount := utils.GetCounterMetric(metricsURL, "llm_d_epp_disagg_decision_total", labelFilter2)

			gomega.Expect(prefillDecodeCount).Should(gomega.Equal(4))
			gomega.Expect(decodeOnlyCount).Should(gomega.Equal(2))
		})
	}))

	ginkgo.When("Running simple non-PD configuration with disagg-profile-handler", ginkgo.Ordered, testWrapper(func() {
		ginkgo.It("should run successfully", func() {
			createModelServersDecode(1)

			standalone.Create(standaloneConfig(), decodeOnlyConfig, 1, 8000)
			nsName := getNamespace()

			prefillPods, decodePods := utils.GetModelServerPods(testConfig, podSelector, prefillSelector, decodeSelector, nsName)
			gomega.Expect(prefillPods).Should(gomega.BeEmpty())
			gomega.Expect(decodePods).Should(gomega.HaveLen(1))

			nsHdr, podHdr, _ := runCompletion(simplePrompt, simModelName)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.Equal(decodePods[0]))

			nsHdr, podHdr, _ = runChatCompletion(simplePrompt, simModelName)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.Equal(decodePods[0]))
		})
	}))

	ginkgo.When("Running an E/PD (Encode/Prefill-Decode) configuration", ginkgo.Ordered, testWrapper(func() {
		ginkgo.It("should route multimodal requests through encode and decode pods", func() {
			encodeReplicas := 2
			decodeReplicas := 1
			createModelServersEpDDisagg(encodeReplicas, decodeReplicas)

			standalone.Create(standaloneConfig(), epdEncodeDecodeConfig, 1, 8000)
			nsName := getNamespace()

			metricsURL := fmt.Sprintf("http://localhost:%d/metrics", getMetricsPort())

			encodePods := utils.GetPodNames(testConfig, encodeSelector, nsName)
			prefillDecodePods := utils.GetPodNames(testConfig, prefillDecodeSelector, nsName)
			gomega.Expect(encodePods).Should(gomega.HaveLen(encodeReplicas))
			gomega.Expect(prefillDecodePods).Should(gomega.HaveLen(decodeReplicas))

			// Text request: encode stage skipped, routed directly to a prefill-decode pod
			nsHdr, podHdr, _ := runCompletion(simplePrompt, simModelName)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.BeElementOf(prefillDecodePods))

			// Each entry drives one encode-stage multimodal request. The metric
			// assertion at the bottom of the block uses len(mmRequests) as its
			// expected count, so adding a row here bumps the assertion for free.
			mmRequests := []func() (string, string){
				// Two single-image requests; the second may hit prefix cache.
				func() (string, string) { return runChatCompletionWithImages(testImageURL) },
				func() (string, string) { return runChatCompletionWithImages(testImageURL) },
				// Multi-image request: two images in one request.
				func() (string, string) { return runChatCompletionWithImages(testImageURL, testImageURL2) },
				// Video request: video_url triggers encode stage.
				func() (string, string) { return runChatCompletionWithVideos() },
				// Audio request: input_audio triggers encode stage.
				func() (string, string) { return runChatCompletionWithAudios() },
				// Multi-audio request: two input_audio blocks in a single request.
				func() (string, string) { return runChatCompletionWithAudios(testAudioData, testAudioData) },
				// Multi-video request: two distinct video URLs so the router's URL
				// dedup does not collapse them, exercising per-item fan-out.
				func() (string, string) { return runChatCompletionWithVideos(testVideoURL, testVideoURL2) },
				// Mixed-media request: image + audio + video combined.
				func() (string, string) {
					return runChatCompletionWithMixedMedia(
						[]string{testImageURL},
						[]string{testAudioData},
						[]string{testVideoURL},
					)
				},
			}
			for _, req := range mmRequests {
				nsHdr, podHdr = req()
				gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
				gomega.Expect(podHdr).Should(gomega.BeElementOf(prefillDecodePods))
			}

			// image_embeds request: pre-encoded tensor, encode stage skipped, routes to prefill-decode pod
			nsHdr, podHdr = runChatCompletionWithImageEmbeds()
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.BeElementOf(prefillDecodePods))

			// Metrics: text + image_embeds requests recorded as decode-only (encode skipped)
			decodeOnlyFilter := fmt.Sprintf(`decision_type=%q,model_name="%s"`, disagg.DecisionTypeDecodeOnly, simModelName)
			decodeOnlyCount := utils.GetCounterMetric(metricsURL, "llm_d_epp_disagg_decision_total", decodeOnlyFilter)
			gomega.Expect(decodeOnlyCount).Should(gomega.Equal(2))

			// Metrics: encode-decode decisions recorded, one per entry in mmRequests.
			labelFilter := fmt.Sprintf(`decision_type=%q,model_name="%s"`, disagg.DecisionTypeEncodeDecode, simModelName)
			encodeDecodeCount := utils.GetCounterMetric(metricsURL, "llm_d_epp_disagg_decision_total", labelFilter)
			gomega.Expect(encodeDecodeCount).Should(gomega.Equal(len(mmRequests)))
		})
	}))

	ginkgo.When("Running an E/P/D (encode/prefill/decode) configuration", ginkgo.Ordered, testWrapper(func() {
		ginkgo.It("should route multimodal requests through encode, prefill, and decode pods", func() {
			encodeReplicas := 2
			prefillReplicas := 1
			decodeReplicas := 1
			createModelServersEPDDisagg(encodeReplicas, prefillReplicas, decodeReplicas)

			standalone.Create(standaloneConfig(), epdConfig, 1, 8000)
			nsName := getNamespace()

			metricsURL := fmt.Sprintf("http://localhost:%d/metrics", getMetricsPort())

			encodePods := utils.GetPodNames(testConfig, encodeSelector, nsName)
			prefillPods := utils.GetPodNames(testConfig, prefillSelector, nsName)
			decodePods := utils.GetPodNames(testConfig, decodeSelector, nsName)
			gomega.Expect(encodePods).Should(gomega.HaveLen(encodeReplicas))
			gomega.Expect(prefillPods).Should(gomega.HaveLen(prefillReplicas))
			gomega.Expect(decodePods).Should(gomega.HaveLen(decodeReplicas))

			// Text request: encode stage skipped, prefill triggered by prefix-based-pd-decider
			nsHdr, podHdr, _ := runCompletion(simplePrompt, simModelName)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.BeElementOf(decodePods))

			// Each entry drives one multimodal request that reaches encode + decode.
			// The commented-out metric assertion at the bottom uses len(mmRequests)
			// as its expected count, so adding a row here bumps that assertion for free.
			mmRequests := []func() (string, string){
				// First image: unique content, encode-prefill-decode.
				func() (string, string) { return runChatCompletionWithImages(testImageURL) },
				// Same image again: prefix cache may hit, producing encode-decode.
				func() (string, string) { return runChatCompletionWithImages(testImageURL) },
				// Multi-image request.
				func() (string, string) { return runChatCompletionWithImages(testImageURL, testImageURL2) },
				// Video request.
				func() (string, string) { return runChatCompletionWithVideos() },
				// Audio request.
				func() (string, string) { return runChatCompletionWithAudios() },
			}
			for _, req := range mmRequests {
				nsHdr, podHdr = req()
				gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
				gomega.Expect(podHdr).Should(gomega.BeElementOf(decodePods))
			}

			// image_embeds request: pre-encoded tensor, encode stage skipped, routes to decode pod
			nsHdr, podHdr = runChatCompletionWithImageEmbeds()
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.BeElementOf(decodePods))

			// Metrics: text + image_embeds requests recorded as decode-only or prefill-decode (encode skipped)
			pdLabelFilter := fmt.Sprintf(`decision_type=%q,model_name="%s"`, disagg.DecisionTypePrefillDecode, simModelName)
			doLabelFilter := fmt.Sprintf(`decision_type=%q,model_name="%s"`, disagg.DecisionTypeDecodeOnly, simModelName)
			pdCount := utils.GetCounterMetric(metricsURL, "llm_d_epp_disagg_decision_total", pdLabelFilter)
			doCount := utils.GetCounterMetric(metricsURL, "llm_d_epp_disagg_decision_total", doLabelFilter)
			gomega.Expect(pdCount + doCount).Should(gomega.Equal(2))

			// TODO(#1253): re-enable the multimodal decision-counter assertions
			// below once the router reports EPD decisions correctly. Each entry
			// in mmRequests contributes one encode-prefill-decode or encode-decode
			// decision (encode-decode when the prefix cache hits on the second
			// same-image request).
			// epdLabelFilter := fmt.Sprintf(`decision_type=%q,model_name="%s"`, disagg.DecisionTypeEncodePrefillDecode, simModelName)
			// edLabelFilter := fmt.Sprintf(`decision_type=%q,model_name="%s"`, disagg.DecisionTypeEncodeDecode, simModelName)
			// epdCount := utils.GetCounterMetric(metricsURL, "llm_d_epp_disagg_decision_total", epdLabelFilter)
			// edCount := utils.GetCounterMetric(metricsURL, "llm_d_epp_disagg_decision_total", edLabelFilter)
			// gomega.Expect(epdCount + edCount).Should(gomega.Equal(len(mmRequests)))
		})
	}))

	ginkgo.When("Running an EPD (no disaggregation) configuration", ginkgo.Ordered, testWrapper(func() {
		ginkgo.It("should route text and multimodal requests to the single deployment", func() {
			// Single deployment labeled encode-prefill-decode: matches encode-filter, prefill-filter,
			// and decode-filter, so all EPD stages are handled by the same deployment.
			replicas := 1
			createModelServersEPDUnified(replicas)

			// Using epdConfig instead of decodeOnlyConfig to validate the EPD logic path within
			// a single pod; multimodal stages will resolve to this same deployment.
			standalone.Create(standaloneConfig(), epdConfig, 1, 8000)
			nsName := getNamespace()

			metricsURL := fmt.Sprintf("http://localhost:%d/metrics", getMetricsPort())

			epdPods := utils.GetPodNames(testConfig, epdSingleSelector, nsName)
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
			pdLabelFilter := fmt.Sprintf(`decision_type=%q,model_name="%s"`, disagg.DecisionTypePrefillDecode, simModelName)
			doLabelFilter := fmt.Sprintf(`decision_type=%q,model_name="%s"`, disagg.DecisionTypeDecodeOnly, simModelName)
			pdCount := utils.GetCounterMetric(metricsURL, "llm_d_epp_disagg_decision_total", pdLabelFilter)
			doCount := utils.GetCounterMetric(metricsURL, "llm_d_epp_disagg_decision_total", doLabelFilter)
			gomega.Expect(pdCount + doCount).Should(gomega.Equal(2))

			// Multimodal request: encode and decode profiles both resolve to the same single deployment
			nsHdr, podHdr = runChatCompletionWithImages(testImageURL)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.Equal(epdPods[0]))

			// Multi-image request: all stages handled by single deployment
			nsHdr, podHdr = runChatCompletionWithImages(testImageURL, testImageURL2)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.Equal(epdPods[0]))

			// Video request: all stages handled by single deployment
			nsHdr, podHdr = runChatCompletionWithVideos()
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.Equal(epdPods[0]))

			// Audio request: all stages handled by single deployment
			nsHdr, podHdr = runChatCompletionWithAudios()
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.Equal(epdPods[0]))

			// image_embeds request: encode skipped, routes to single deployment
			nsHdr, podHdr = runChatCompletionWithImageEmbeds()
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			gomega.Expect(podHdr).Should(gomega.Equal(epdPods[0]))
		})
	}))

	ginkgo.When("Running simple non-PD KV enabled configuration", ginkgo.Ordered, testWrapper(func() {
		ginkgo.It("should run successfully", func() {
			createModelServersDecodeKV(1)
			standalone.Create(standaloneConfig(), kvConfig(), 1, 8000)
			nsName := getNamespace()

			prefillPods, decodePods := utils.GetModelServerPods(testConfig, podSelector, prefillSelector, decodeSelector, nsName)
			gomega.Expect(prefillPods).Should(gomega.BeEmpty())
			gomega.Expect(decodePods).Should(gomega.HaveLen(1))

			for range 5 {
				nsHdr, podHdr, _ := runCompletion(simplePrompt, kvModelName)
				gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
				gomega.Expect(podHdr).Should(gomega.Equal(decodePods[0]))
			}
		})
	}))

	ginkgo.When("Running KV configuration with external tokenizer DataProducer plugin", ginkgo.Ordered, testWrapper(func() {
		ginkgo.It("should run successfully", func() {
			createModelServersDecodeKV(1)
			standalone.Create(standaloneConfig(), kvExternalTokenizerConfig(), 1, 8000)
			nsName := getNamespace()

			prefillPods, decodePods := utils.GetModelServerPods(testConfig, podSelector, prefillSelector, decodeSelector, nsName)
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
		})
	}))

	ginkgo.When("Scaling up and down the model servers", ginkgo.Ordered, testWrapper(func() {
		ginkgo.It("should distribute inference requests across all model servers", func() {
			modelServers := createModelServersDecode(1)

			standalone.Create(standaloneConfig(), scaleConfig, 1, 8000)
			nsName := getNamespace()

			prefillPods, decodePods := utils.GetModelServerPods(testConfig, podSelector, prefillSelector, decodeSelector, nsName)
			gomega.Expect(prefillPods).Should(gomega.BeEmpty())
			gomega.Expect(decodePods).Should(gomega.HaveLen(1))

			var nsHdr, podHdr string
			for range 5 {
				nsHdr, podHdr, _ = runCompletion(simplePrompt, simModelName)
				gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
				gomega.Expect(podHdr).Should(gomega.Equal(decodePods[0]))
			}

			utils.ScaleDeployment(testConfig, nsName, modelServers, 1)

			scaledUpPrefillPods, scaledUpDecodePods := utils.GetModelServerPods(testConfig, podSelector, prefillSelector, decodeSelector, nsName)
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

			utils.ScaleDeployment(testConfig, nsName, modelServers, -1)

			scaledDownPrefillPods, scaledDownDecodePods := utils.GetModelServerPods(testConfig, podSelector, prefillSelector, decodeSelector, nsName)
			gomega.Expect(scaledDownPrefillPods).Should(gomega.BeEmpty())
			gomega.Expect(scaledDownDecodePods).Should(gomega.HaveLen(1))
			gomega.Expect(scaledDownDecodePods[0]).Should(gomega.BeElementOf(scaledUpDecodePods))

			// Run multiple times and insure that they are scheduled on the remaining pod
			for range 5 {
				nsHdr, podHdr, _ = runCompletion(simplePrompt, simModelName)
				gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
				gomega.Expect(podHdr).Should(gomega.Equal(scaledDownDecodePods[0]))
			}
		})
	}))

	ginkgo.When("Running a vLLM Data Parallel configuration", ginkgo.Ordered, testWrapper(func() {
		ginkgo.It("should schedule inference on all ranks", func() {
			createModelServersDecodeDP(1)

			standalone.Create(standaloneConfig(), dataParallelConfig, 1, 8000, 8001)
			nsName := getNamespace()

			prefillPods, decodePods := utils.GetModelServerPods(testConfig, podSelector, prefillSelector, decodeSelector, nsName)
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
		})
	}))
})
