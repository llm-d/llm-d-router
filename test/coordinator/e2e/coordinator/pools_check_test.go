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

package coordinate2e

import (
	"github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/types"
	inferenceapi "sigs.k8s.io/gateway-api-inference-extension/api/v1"
)

// expectPoolExists asserts that the single InferencePool covering the encode,
// prefill, and decode worker pods exists in the test namespace. Its existence
// is the single hard signal that the env wired correctly: every other route in
// the pipeline depends on it. The name is poolNameBase (e.g.
// qwen3-vl-2b-instruct-inference-pool).
func expectPoolExists() {
	nsName := getNamespace()
	pool := &inferenceapi.InferencePool{}
	key := types.NamespacedName{Namespace: nsName, Name: poolNameBase}
	gomega.Eventually(func() error {
		return testConfig.K8sClient.Get(testConfig.Context, key, pool)
	}, readyTimeout, defaultInterval).Should(
		gomega.Succeed(),
		"InferencePool %q not found in namespace %q", poolNameBase, nsName,
	)
}
