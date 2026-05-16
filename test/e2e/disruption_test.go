package e2e

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	testutils "github.com/llm-d/llm-d-router/test/utils"
)

const (
	// podRemovalTimeout is how long to wait for a deleted pod to disappear from the API.
	podRemovalTimeout = 60 * time.Second
	// eppRecoveryTimeout is how long to wait for the EPP to restart and pass health checks.
	eppRecoveryTimeout = 60 * time.Second
	// trafficProbeTimeout is how long to wait for traffic to observe a state change.
	trafficProbeTimeout = 30 * time.Second
	// requestInterval is the delay between requests in traffic loops.
	requestInterval = 200 * time.Millisecond
)

var disruptionClient = &http.Client{Timeout: 10 * time.Second}

// sendRawCompletion sends a completion request and returns the HTTP status code.
func sendRawCompletion() (int, error) {
	body := fmt.Sprintf(`{"model":"%s","prompt":"%s","max_tokens":10}`, simModelName, simplePrompt)
	req, err := http.NewRequest(http.MethodPost, fmt.Sprintf("http://localhost:%s/v1/completions", port), strings.NewReader(body))
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := disruptionClient.Do(req)
	if err != nil {
		return 0, err
	}
	defer func() { _ = resp.Body.Close() }()
	return resp.StatusCode, nil
}

// podGone returns true when the named pod no longer appears in the pod list
// and at least minRemaining pods exist.
func podGone(podName string, minRemaining int) func() bool {
	return func() bool {
		_, currentDecode := getModelServerPods(podSelector, prefillSelector, decodeSelector)
		for _, pod := range currentDecode {
			if pod == podName {
				return false
			}
		}
		return len(currentDecode) >= minRemaining
	}
}

// eppPodReady returns true when a new EPP pod (not oldPodName) is Running and Ready.
func eppPodReady(oldPodName string) func() bool {
	return func() bool {
		pods := getPods(map[string]string{"app": "e2e-epp"})
		for _, p := range pods {
			if p.Name == oldPodName {
				continue
			}
			if p.Status.Phase != corev1.PodRunning {
				continue
			}
			for _, c := range p.Status.Conditions {
				if c.Type == corev1.PodReady && c.Status == corev1.ConditionTrue {
					return true
				}
			}
		}
		return false
	}
}

var _ = ginkgo.Describe("Disruption tests", ginkgo.Ordered, ginkgo.Label("Disruptive"), func() {
	ginkgo.When("A decode pod is killed mid-request", func() {
		ginkgo.It("should recover and route to surviving pods", func() {
			infPoolObjects = createInferencePool(1, true)

			modelServers := createModelServersDecode(2)
			ginkgo.DeferCleanup(testutils.DeleteObjects, testConfig, modelServers)

			epp := createEndPointPicker(simpleConfig)
			ginkgo.DeferCleanup(testutils.DeleteObjects, testConfig, epp)

			prefillPods, decodePods := getModelServerPods(podSelector, prefillSelector, decodeSelector)
			gomega.Expect(prefillPods).Should(gomega.BeEmpty())
			gomega.Expect(decodePods).Should(gomega.HaveLen(2))

			ginkgo.By("Verifying requests route successfully before disruption")
			nsHdr, _, _ := runCompletion(simplePrompt, simModelName)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))

			targetPod := decodePods[0]

			ginkgo.By("Force-deleting decode pod " + targetPod)
			deletePodByName(targetPod, 0)

			ginkgo.By("Waiting for killed pod to be replaced")
			gomega.Eventually(podGone(targetPod, 1), podRemovalTimeout, 1*time.Second).Should(gomega.BeTrue())

			ginkgo.By("Verifying new requests eventually route to a pod other than the killed one")
			gomega.Eventually(func() error {
				nsHdr, podHdr, _, err := tryCompletion(simplePrompt, simModelName)
				if err != nil {
					return err
				}
				if nsHdr != nsName {
					return fmt.Errorf("expected namespace %q, got %q", nsName, nsHdr)
				}
				if podHdr == targetPod {
					return fmt.Errorf("request still routed to killed pod %q", targetPod)
				}
				return nil
			}, eppRecoveryTimeout, 1*time.Second).Should(gomega.Succeed())

			ginkgo.By("Waiting for replacement pod to become ready")
			gomega.Eventually(func() int {
				_, currentDecode := getModelServerPods(podSelector, prefillSelector, decodeSelector)
				return len(currentDecode)
			}, readyTimeout, 2*time.Second).Should(gomega.Equal(2))

			ginkgo.By("Verifying requests succeed after recovery")
			for range 3 {
				nsHdr, _, _ = runCompletion(simplePrompt, simModelName)
				gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			}
		})
	})

	ginkgo.When("A decode pod is killed while a streaming request is in-flight", func() {
		ginkgo.It("should not hang and should recover routing", func() {
			infPoolObjects = createInferencePool(1, true)

			modelServers := createModelServersDecode(2)
			ginkgo.DeferCleanup(testutils.DeleteObjects, testConfig, modelServers)

			epp := createEndPointPicker(simpleConfig)
			ginkgo.DeferCleanup(testutils.DeleteObjects, testConfig, epp)

			prefillPods, decodePods := getModelServerPods(podSelector, prefillSelector, decodeSelector)
			gomega.Expect(prefillPods).Should(gomega.BeEmpty())
			gomega.Expect(decodePods).Should(gomega.HaveLen(2))

			ginkgo.By("Starting a streaming request")
			connected := make(chan string, 1)
			errCh := make(chan error, 1)
			go func() {
				defer ginkgo.GinkgoRecover()
				errCh <- sendStreamingCompletion(connected)
			}()

			targetPod := <-connected
			gomega.Expect(targetPod).ShouldNot(gomega.BeEmpty(),
				"streaming request failed to connect")

			ginkgo.By(fmt.Sprintf("Force-deleting decode pod %s during the stream", targetPod))
			deletePodByName(targetPod, 0)

			err := <-errCh
			if err != nil {
				ginkgo.By(fmt.Sprintf("Stream terminated with error: %v", err))
				gomega.Expect(err).ShouldNot(gomega.MatchError(context.DeadlineExceeded),
					"stream should fail fast on pod kill, not hang until client timeout")
			} else {
				ginkgo.By("Stream completed before pod was killed")
			}

			ginkgo.By("Waiting for replacement pod")
			gomega.Eventually(func() int {
				_, currentDecode := getModelServerPods(podSelector, prefillSelector, decodeSelector)
				return len(currentDecode)
			}, readyTimeout, 2*time.Second).Should(gomega.Equal(2))

			ginkgo.By("Verifying requests succeed after recovery")
			for range 3 {
				nsHdr, _, _ := runCompletion(simplePrompt, simModelName)
				gomega.Expect(nsHdr).Should(gomega.Equal(nsName))
			}
		})
	})

	ginkgo.When("All pods are gone", func() {
		ginkgo.It("should return 503 to the client", func() {
			infPoolObjects = createInferencePool(1, true)

			modelServers := createModelServersDecode(1)
			ginkgo.DeferCleanup(testutils.DeleteObjects, testConfig, modelServers)

			epp := createEndPointPicker(simpleConfig)
			ginkgo.DeferCleanup(testutils.DeleteObjects, testConfig, epp)

			_, decodePods := getModelServerPods(podSelector, prefillSelector, decodeSelector)
			gomega.Expect(decodePods).Should(gomega.HaveLen(1))

			ginkgo.By("Verifying requests succeed before disruption")
			nsHdr, _, _ := runCompletion(simplePrompt, simModelName)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))

			ginkgo.By("Scaling deployment to zero")
			scaleDeployment(modelServers, -1)

			ginkgo.By("Waiting for all pods to be removed")
			gomega.Eventually(func() int {
				_, currentDecode := getModelServerPods(podSelector, prefillSelector, decodeSelector)
				return len(currentDecode)
			}, podRemovalTimeout, 1*time.Second).Should(gomega.Equal(0))

			ginkgo.By("Waiting for requests to return 503")
			gomega.Eventually(func() int {
				status, err := sendRawCompletion()
				if err != nil {
					return 0
				}
				return status
			}, trafficProbeTimeout, 500*time.Millisecond).Should(gomega.Equal(http.StatusServiceUnavailable))

			ginkgo.By("Scaling deployment back up")
			scaleDeployment(modelServers, 1)

			ginkgo.By("Verifying requests succeed after recovery")
			gomega.Eventually(func() string {
				nsHdr, _, _ := runCompletion(simplePrompt, simModelName)
				return nsHdr
			}, eppRecoveryTimeout, 2*time.Second).Should(gomega.Equal(nsName))
		})
	})

	ginkgo.When("The EPP is killed while requests are in flight", func() {
		ginkgo.It("should recover and resume routing after restart", func() {
			infPoolObjects = createInferencePool(1, true)

			modelServers := createModelServersDecode(1)
			ginkgo.DeferCleanup(testutils.DeleteObjects, testConfig, modelServers)

			epp := createEndPointPicker(simpleConfig)
			ginkgo.DeferCleanup(testutils.DeleteObjects, testConfig, epp)

			ginkgo.By("Verifying requests succeed before EPP disruption")
			nsHdr, _, _ := runCompletion(simplePrompt, simModelName)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))

			ginkgo.By("Finding EPP pod")
			eppPods := getPods(map[string]string{"app": "e2e-epp"})
			gomega.Expect(eppPods).Should(gomega.HaveLen(1))
			eppPodName := eppPods[0].Name

			ginkgo.By("Force-deleting EPP pod " + eppPodName)
			deletePodByName(eppPodName, 0)

			ginkgo.By("Verifying requests fail while EPP is down")
			gomega.Eventually(func() bool {
				status, err := sendRawCompletion()
				if err != nil {
					return true // connection error = EPP is down
				}
				return status >= 500
			}, trafficProbeTimeout, 500*time.Millisecond).Should(gomega.BeTrue(),
				"requests should fail while EPP is down")

			ginkgo.By("Waiting for EPP to recover")
			gomega.Eventually(eppPodReady(eppPodName), readyTimeout, 2*time.Second).Should(gomega.BeTrue())

			ginkgo.By("Verifying requests succeed after EPP recovery")
			gomega.Eventually(func() error {
				status, err := sendRawCompletion()
				if err != nil {
					return err
				}
				if status != http.StatusOK {
					return fmt.Errorf("expected 200, got %d", status)
				}
				return nil
			}, eppRecoveryTimeout, 2*time.Second).Should(gomega.Succeed())
		})
	})

	ginkgo.When("Traffic is flowing during scale-to-zero and back", func() {
		ginkgo.It("should return 503s when empty and recover when scaled back", func() {
			infPoolObjects = createInferencePool(1, true)

			modelServers := createModelServersDecode(1)
			ginkgo.DeferCleanup(testutils.DeleteObjects, testConfig, modelServers)

			epp := createEndPointPicker(simpleConfig)
			ginkgo.DeferCleanup(testutils.DeleteObjects, testConfig, epp)

			ginkgo.By("Verifying requests succeed before disruption")
			nsHdr, _, _ := runCompletion(simplePrompt, simModelName)
			gomega.Expect(nsHdr).Should(gomega.Equal(nsName))

			ginkgo.By("Starting background traffic")
			ctx, cancel := context.WithCancel(context.Background())
			tc := &trafficCounter{}
			done := make(chan struct{})
			go func() {
				defer close(done)
				runTrafficLoop(ctx, tc)
			}()

			ginkgo.By("Scaling to zero")
			scaleDeployment(modelServers, -1)

			ginkgo.By("Waiting for all pods to be removed")
			gomega.Eventually(func() int {
				_, currentDecode := getModelServerPods(podSelector, prefillSelector, decodeSelector)
				return len(currentDecode)
			}, podRemovalTimeout, 1*time.Second).Should(gomega.Equal(0))

			ginkgo.By("Waiting for traffic to observe failures")
			gomega.Eventually(tc.failures, trafficProbeTimeout, 500*time.Millisecond).Should(gomega.BeNumerically(">", 0))

			ginkgo.By("Scaling back to 1")
			scaleDeployment(modelServers, 1)

			ginkgo.By("Waiting for traffic to observe recovery")
			successBaseline := tc.successes()
			gomega.Eventually(func() int {
				return tc.successes() - successBaseline
			}, eppRecoveryTimeout, 500*time.Millisecond).Should(gomega.BeNumerically(">", 0))

			cancel()
			<-done

			ginkgo.By(fmt.Sprintf("Traffic results: %d successes, %d failures", tc.successes(), tc.failures()))
		})
	})

})

func sendStreamingCompletion(connected chan<- string) error {
	longPrompt := strings.Repeat("This is a longer prompt to keep the stream open. ", 20)
	body := fmt.Sprintf(`{"model":"%s","prompt":"%s","max_tokens":100,"stream":true}`, simModelName, longPrompt)
	req, err := http.NewRequest(http.MethodPost, fmt.Sprintf("http://localhost:%s/v1/completions", port), strings.NewReader(body))
	if err != nil {
		connected <- ""
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := disruptionClient.Do(req)
	if err != nil {
		connected <- ""
		return err
	}
	defer func() { _ = resp.Body.Close() }()

	connected <- resp.Header.Get("x-inference-pod")
	_, err = io.ReadAll(resp.Body)
	return err
}

type trafficCounter struct {
	mu           sync.Mutex
	successCount int
	failCount    int
}

func (tc *trafficCounter) record(status int, err error) {
	tc.mu.Lock()
	defer tc.mu.Unlock()
	if err != nil || status != http.StatusOK {
		tc.failCount++
	} else {
		tc.successCount++
	}
}

func (tc *trafficCounter) failures() int {
	tc.mu.Lock()
	defer tc.mu.Unlock()
	return tc.failCount
}

func (tc *trafficCounter) successes() int {
	tc.mu.Lock()
	defer tc.mu.Unlock()
	return tc.successCount
}

func runTrafficLoop(ctx context.Context, tc *trafficCounter) {
	defer ginkgo.GinkgoRecover()
	ticker := time.NewTicker(requestInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			status, err := sendRawCompletion()
			tc.record(status, err)
		}
	}
}

// deletePodByName force-deletes a pod by name with the given grace period (0 = immediate).
func deletePodByName(podName string, gracePeriodSeconds int64) {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: nsName,
		},
	}
	opts := &client.DeleteOptions{
		GracePeriodSeconds: &gracePeriodSeconds,
	}
	err := testConfig.K8sClient.Delete(testConfig.Context, pod, opts)
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
}
