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

// expectPoolExists asserts that every InferencePool for the active topology
// exists in the test namespace: the single pool covering all three roles
// (single-EPP), or the three role-scoped pools (3-EPP). Their existence is the
// hard signal that the env wired correctly, since every route in the pipeline
// depends on them.
func expectPoolExists() {
	nsName := getNamespace()
	for _, name := range poolNames() {
		pool := &inferenceapi.InferencePool{}
		key := types.NamespacedName{Namespace: nsName, Name: name}
		gomega.Eventually(func() error {
			return testConfig.K8sClient.Get(testConfig.Context, key, pool)
		}, readyTimeout, defaultInterval).Should(
			gomega.Succeed(),
			"InferencePool %q not found in namespace %q", name, nsName,
		)
	}
}
