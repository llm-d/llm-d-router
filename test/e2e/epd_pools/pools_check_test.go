/*
Copyright 2026 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package epdpools

import (
	"os"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/types"
	infextv1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	igwtestutils "github.com/llm-d/llm-d-router/test/utils/igw"
)

var _ = ginkgo.Describe("e-p-d-pools env resources", func() {
	ginkgo.It("AllPoolsWired: encode, prefill, decode InferencePools exist", func() {
		expectAllPoolsExist(testConfig)
	})
})

// expectedPools enumerates the three phase-specific InferencePools the
// e-p-d-pools env brings up. Their existence is the single hard signal that
// the env wired correctly — every other route in the pipeline depends on
// them. Names are derived from the templated $POOL_NAME (e.g.
// `qwen3-vl-2b-instruct-inference-pool`); set E2E_POOL_NAME to override.
func expectedPools() []string {
	base := os.Getenv("E2E_POOL_NAME")
	if base == "" {
		base = "qwen3-vl-2b-instruct-inference-pool"
	}
	return []string{base + "-encode", base + "-prefill", base + "-decode"}
}

// expectAllPoolsExist asserts that all three InferencePools (encode,
// prefill, decode) exist in the namespace tracked by tc. Used by the
// AllPoolsWired test and as a suite precondition.
func expectAllPoolsExist(tc *igwtestutils.TestConfig) {
	for _, name := range expectedPools() {
		pool := &infextv1.InferencePool{}
		key := types.NamespacedName{Name: name, Namespace: tc.NsName}
		gomega.Eventually(func() error {
			return tc.K8sClient.Get(tc.Context, key, pool)
		}, tc.ExistsTimeout, tc.Interval).Should(
			gomega.Succeed(),
			"InferencePool %q not found in namespace %q", name, tc.NsName,
		)
	}
}
