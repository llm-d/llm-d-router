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
	. "github.com/onsi/ginkgo/v2" // nolint:revive
	. "github.com/onsi/gomega"    // nolint:revive

	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"k8s.io/utils/set"
)

var _ = Describe("AllowlistValidator", func() {
	Context("when SSRF protection is disabled", func() {
		var validator *AllowlistValidator

		BeforeEach(func() {
			var err error
			validator, err = NewAllowlistValidator(false, routing.DefaultPoolGroup, "test-namespace", "test-pool")
			Expect(err).ToNot(HaveOccurred())
		})

		It("should allow all targets", func() {
			Expect(validator.IsAllowed("malicious.example.com:8080")).To(BeTrue())
			Expect(validator.IsAllowed("10.0.0.1:8000")).To(BeTrue())
			Expect(validator.IsAllowed("http://evil.host/ssrf")).To(BeTrue())
		})
	})

	Context("when SSRF protection is enabled", func() {
		var validator *AllowlistValidator

		BeforeEach(func() {
			validator = &AllowlistValidator{
				enabled:   true,
				namespace: "test-namespace",
				allowedTargets: set.New(
					"10.244.1.100",
					"valid-pod",
					"valid-pod.test-namespace.svc.cluster.local",
				),
			}
		})

		It("should allow targets in the allowlist", func() {
			Expect(validator.IsAllowed("10.244.1.100:8000")).To(BeTrue())
			Expect(validator.IsAllowed("valid-pod:8000")).To(BeTrue())
			Expect(validator.IsAllowed("valid-pod.test-namespace.svc.cluster.local:8000")).To(BeTrue())
			Expect(validator.IsAllowed("10.244.1.100:8001")).To(BeTrue()) // Different port, same host
			Expect(validator.IsAllowed("valid-pod:9999")).To(BeTrue())    // Any port on allowed host
		})

		It("should block targets not in the allowlist", func() {
			Expect(validator.IsAllowed("malicious.example.com:8080")).To(BeFalse())
			Expect(validator.IsAllowed("10.0.0.1:8000")).To(BeFalse())
			Expect(validator.IsAllowed("evil-pod:8000")).To(BeFalse())
		})

		It("should parse host:port correctly", func() {
			// Test host:port format parsing
			Expect(extractHost("10.244.1.100:8000")).To(Equal("10.244.1.100"))
			Expect(extractHost("valid-pod:8000")).To(Equal("valid-pod"))
			// Just hostname (no port)
			Expect(extractHost("valid-pod")).To(Equal("valid-pod"))
			// IPv6 addresses (net.SplitHostPort handles these correctly
			Expect(extractHost("[::1]:8000")).To(Equal("::1"))
			// IPv6 without port
			Expect(extractHost("::1")).To(Equal("::1"))
		})
	})
})
