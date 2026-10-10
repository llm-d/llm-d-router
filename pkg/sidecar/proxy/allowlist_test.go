/*
Copyright 2025 The llm-d Authors.

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

/*
Copyright 2025 The llm-d Authors

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

package proxy

import (
	"sync"
	"time"

	. "github.com/onsi/ginkgo/v2" // nolint:revive
	. "github.com/onsi/gomega"    // nolint:revive

	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	dynamicfake "k8s.io/client-go/dynamic/fake"
	clientfeatures "k8s.io/client-go/features"
	clientfeaturestesting "k8s.io/client-go/features/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/utils/set"
)

// newTestAllowlistValidator creates an enabled validator with a fake Kubernetes client.
func newTestAllowlistValidator() *AllowlistValidator {
	GinkgoHelper()
	clientfeaturestesting.SetFeatureDuringTest(GinkgoTB(), clientfeatures.WatchListClient, false)
	poolGVR := schema.GroupVersionResource{Group: routing.InferencePoolAPIGroup, Version: "v1", Resource: inferencePoolResource}
	podGVR := schema.GroupVersionResource{Version: "v1", Resource: "pods"}
	client := dynamicfake.NewSimpleDynamicClientWithCustomListKinds(runtime.NewScheme(), map[schema.GroupVersionResource]string{
		poolGVR: "InferencePoolList",
		podGVR:  "PodList",
	})
	validator := &AllowlistValidator{
		enabled:        true,
		dynamicClient:  client,
		namespace:      "test",
		poolName:       "test",
		gvr:            poolGVR,
		allowedTargets: set.New[string](),
		podInformers:   make(map[string]cache.SharedInformer),
		podStopChans:   make(map[string]chan struct{}),
		poolPorts:      make(map[string][]string),
		stopCh:         make(chan struct{}),
	}
	return validator
}

var _ = Describe("AllowlistValidator", func() {
	Context("lifecycle", func() {
		var validator *AllowlistValidator

		BeforeEach(func() {
			validator = newTestAllowlistValidator()
			DeferCleanup(validator.Stop)
		})

		It("should reject repeated starts and allow repeated stops", func() {
			ctx := newTestContext()
			Expect(validator.Start(ctx)).To(Succeed())
			Expect(validator.Start(ctx)).To(HaveOccurred())
			validator.Stop()
			Expect(validator.Stop).ToNot(Panic())
			Expect(validator.Start(ctx)).To(HaveOccurred())
		})

		It("should stop concurrently without creating more pod informers", func() {
			Expect(validator.Start(newTestContext())).To(Succeed())
			validator.createPodInformer(validator.poolName, labels.Everything(), []string{"8000"})
			validator.podInformersMu.RLock()
			podInformer := validator.podInformers[validator.poolName]
			validator.podInformersMu.RUnlock()
			Eventually(podInformer.HasSynced, 3*time.Second, 10*time.Millisecond).Should(BeTrue())

			var callers sync.WaitGroup
			for range 8 {
				callers.Add(1)
				go func() {
					defer callers.Done()
					validator.Stop()
					validator.createPodInformer(validator.poolName, labels.Everything(), []string{"8000"})
				}()
			}
			callers.Wait()
			Eventually(validator.poolInformer.IsStopped, 3*time.Second, 10*time.Millisecond).Should(BeTrue())
			Eventually(podInformer.IsStopped, 3*time.Second, 10*time.Millisecond).Should(BeTrue())
			Expect(validator.podInformers).To(BeEmpty())
			Expect(validator.podStopChans).To(BeEmpty())
		})
	})

	Context("when SSRF protection is disabled", func() {
		var validator *AllowlistValidator

		BeforeEach(func() {
			var err error
			validator, err = NewAllowlistValidator(false, routing.InferencePoolAPIGroup, "test-namespace", "test-pool")
			Expect(err).ToNot(HaveOccurred())
		})

		It("should allow all targets", func() {
			Expect(validator.IsAllowed("malicious.example.com:8080")).To(BeTrue())
			Expect(validator.IsAllowed("10.0.0.1:8000")).To(BeTrue())
			Expect(validator.IsAllowed("http://evil.host/ssrf")).To(BeTrue())
		})
	})

	Context("poolSelector", func() {
		It("should extract selector from GA InferencePool (matchLabels)", func() {
			av := &AllowlistValidator{
				gvr: schema.GroupVersionResource{
					Group:    routing.InferencePoolAPIGroup,
					Version:  "v1",
					Resource: "inferencepools",
				},
			}
			pool := &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "inference.networking.k8s.io/v1",
					"kind":       "InferencePool",
					"metadata":   map[string]interface{}{"name": "test-pool"},
					"spec": map[string]interface{}{
						"selector": map[string]interface{}{
							"matchLabels": map[string]interface{}{
								"app.kubernetes.io/name": "my-model",
								"component":              "serving",
							},
						},
					},
				},
			}

			selector, err := av.poolSelector(pool)
			Expect(err).ToNot(HaveOccurred())
			Expect(selector.String()).To(SatisfyAll(
				ContainSubstring("app.kubernetes.io/name=my-model"),
				ContainSubstring("component=serving"),
			))
		})

		It("should fail for GA pool with flat selector (no matchLabels)", func() {
			av := &AllowlistValidator{
				gvr: schema.GroupVersionResource{
					Group:    routing.InferencePoolAPIGroup,
					Version:  "v1",
					Resource: "inferencepools",
				},
			}
			pool := &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "inference.networking.k8s.io/v1",
					"kind":       "InferencePool",
					"metadata":   map[string]interface{}{"name": "test-pool"},
					"spec": map[string]interface{}{
						"selector": map[string]interface{}{
							"app": "my-model",
						},
					},
				},
			}

			_, err := av.poolSelector(pool)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("matchLabels"))
		})

		It("should fail when spec is missing", func() {
			av := &AllowlistValidator{
				gvr: schema.GroupVersionResource{
					Group:    routing.InferencePoolAPIGroup,
					Version:  "v1",
					Resource: "inferencepools",
				},
			}
			pool := &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "inference.networking.k8s.io/v1",
					"kind":       "InferencePool",
					"metadata":   map[string]interface{}{"name": "test-pool"},
				},
			}

			_, err := av.poolSelector(pool)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("spec"))
		})

		It("should fail when selector is missing", func() {
			av := &AllowlistValidator{
				gvr: schema.GroupVersionResource{
					Group:    routing.InferencePoolAPIGroup,
					Version:  "v1",
					Resource: "inferencepools",
				},
			}
			pool := &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "inference.networking.k8s.io/v1",
					"kind":       "InferencePool",
					"metadata":   map[string]interface{}{"name": "test-pool"},
					"spec":       map[string]interface{}{},
				},
			}

			_, err := av.poolSelector(pool)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("selector"))
		})
	})

	Context("poolTargetPorts", func() {
		gaValidator := &AllowlistValidator{
			gvr: schema.GroupVersionResource{
				Group:    routing.InferencePoolAPIGroup,
				Version:  "v1",
				Resource: "inferencepools",
			},
		}

		It("should extract every target port from GA InferencePool", func() {
			pool := &unstructured.Unstructured{
				Object: map[string]interface{}{
					"spec": map[string]interface{}{
						"targetPorts": []interface{}{
							map[string]interface{}{"number": int64(8000)},
							map[string]interface{}{"number": int64(8001)},
						},
					},
				},
			}

			ports, err := gaValidator.poolTargetPorts(pool)
			Expect(err).ToNot(HaveOccurred())
			Expect(ports).To(Equal([]string{"8000", "8001"}))
		})

		It("should extract targetPortNumber from deprecated alpha InferencePool", func() {
			av := &AllowlistValidator{
				gvr: schema.GroupVersionResource{
					Group:    "inference.networking.x-k8s.io",
					Version:  "v1alpha2",
					Resource: "inferencepools",
				},
			}
			pool := &unstructured.Unstructured{
				Object: map[string]interface{}{
					"spec": map[string]interface{}{"targetPortNumber": int64(8000)},
				},
			}

			ports, err := av.poolTargetPorts(pool)
			Expect(err).ToNot(HaveOccurred())
			Expect(ports).To(Equal([]string{"8000"}))
		})

		It("should fail when GA InferencePool has no target ports", func() {
			pool := &unstructured.Unstructured{
				Object: map[string]interface{}{
					"spec": map[string]interface{}{"targetPorts": []interface{}{}},
				},
			}

			_, err := gaValidator.poolTargetPorts(pool)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("targetPorts"))
		})
	})

	Context("when SSRF protection is enabled", func() {
		var validator *AllowlistValidator

		BeforeEach(func() {
			validator = &AllowlistValidator{
				enabled:        true,
				namespace:      "test-namespace",
				allowedTargets: set.New[string](),
			}
			pod := &unstructured.Unstructured{
				Object: map[string]interface{}{
					"metadata": map[string]interface{}{"name": "valid-pod"},
					"status":   map[string]interface{}{"podIP": "10.244.1.100"},
				},
			}
			validator.addPodToAllowlist(pod, "test-pool", []string{"8000", "8001"})
		})

		It("should reject the inference.networking.x-k8s.io pool group", func() {
			_, err := NewAllowlistValidator(true, "inference.networking.x-k8s.io", "test-namespace", "test-pool")
			Expect(err).To(MatchError(ContainSubstring("pool-group must be")))
		})

		It("should allow pod addresses on each target port", func() {
			Expect(validator.IsAllowed("10.244.1.100:8000")).To(BeTrue())
			Expect(validator.IsAllowed("10.244.1.100:8001")).To(BeTrue())
			Expect(validator.IsAllowed("valid-pod:8000")).To(BeTrue())
			Expect(validator.IsAllowed("http://10.244.1.100:8000")).To(BeTrue())
		})

		It("should block allowlisted hosts on ports outside the pool target ports", func() {
			Expect(validator.IsAllowed("10.244.1.100:9090")).To(BeFalse())
			Expect(validator.IsAllowed("valid-pod:9999")).To(BeFalse())
		})

		It("should block targets without a port", func() {
			Expect(validator.IsAllowed("10.244.1.100")).To(BeFalse())
			Expect(validator.IsAllowed("valid-pod")).To(BeFalse())
		})

		It("should block targets not in the allowlist", func() {
			Expect(validator.IsAllowed("malicious.example.com:8080")).To(BeFalse())
			Expect(validator.IsAllowed("10.0.0.1:8000")).To(BeFalse())
			Expect(validator.IsAllowed("evil-pod:8000")).To(BeFalse())
		})

		It("should bracket IPv6 pod addresses", func() {
			pod := &unstructured.Unstructured{
				Object: map[string]interface{}{
					"metadata": map[string]interface{}{"name": "v6-pod"},
					"status":   map[string]interface{}{"podIP": "fd00::1"},
				},
			}
			validator.addPodToAllowlist(pod, "test-pool", []string{"8000"})
			Expect(validator.IsAllowed("[fd00::1]:8000")).To(BeTrue())
			Expect(validator.IsAllowed("[fd00::1]:9090")).To(BeFalse())
		})

		It("should parse host:port correctly", func() {
			// Test host:port format parsing
			Expect(extractHost("10.244.1.100:8000")).To(Equal("10.244.1.100"))
			Expect(extractHost("valid-pod:8000")).To(Equal("valid-pod"))
			// Just hostname (no port)
			Expect(extractHost("valid-pod")).To(Equal("valid-pod"))
			// IPv6 addresses (net.SplitHostPort handles these correctly
			Expect(extractHost("[::1]:8000")).To(Equal(testLoopbackIPv6))
			// IPv6 without port
			Expect(extractHost(testLoopbackIPv6)).To(Equal(testLoopbackIPv6))
		})
	})
})
