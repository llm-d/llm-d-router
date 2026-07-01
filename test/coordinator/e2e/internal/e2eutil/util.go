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

// Package e2eutil holds helpers shared across coordinator e2e test suites.
// The internal/ path prevents import from outside test/e2e/.
package e2eutil

import (
	"os/exec"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gexec"
)

// RunKustomize runs `kubectl kustomize <dir>` and returns the YAML documents
// as a string slice split on `\n---`.
func RunKustomize(kustomizeDir string) []string {
	command := exec.Command("kubectl", "kustomize", kustomizeDir)
	session, err := gexec.Start(command, nil, ginkgo.GinkgoWriter)
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	gomega.Eventually(session).WithTimeout(600 * time.Second).Should(gexec.Exit(0))
	return strings.Split(string(session.Out.Contents()), "\n---")
}

// SubstituteMany replaces all keys in substitutions with their values in each
// input string and returns the results.
func SubstituteMany(inputs []string, substitutions map[string]string) []string {
	outputs := make([]string, len(inputs))
	for idx, input := range inputs {
		output := input
		for key, value := range substitutions {
			output = strings.ReplaceAll(output, key, value)
		}
		outputs[idx] = output
	}
	return outputs
}

// RemoveEmptyArgs strips YAML list items that are empty strings after variable
// substitution (e.g. `- ""` produced when VLLM_EXTRA_ARGS_* is unset).
//
// This is line-based string surgery: it drops any line trimming to `- ""` or
// `-`. It assumes the test manifests use such list items only as substitution
// placeholders, never as legitimate empty args.
func RemoveEmptyArgs(inputs []string) []string {
	outputs := make([]string, len(inputs))
	for idx, input := range inputs {
		lines := strings.Split(input, "\n")
		filtered := make([]string, 0, len(lines))
		for _, line := range lines {
			if strings.TrimSpace(line) == `- ""` {
				continue
			}
			if strings.TrimSpace(line) == `-` {
				continue
			}
			filtered = append(filtered, line)
		}
		outputs[idx] = strings.Join(filtered, "\n")
	}
	return outputs
}

// FilterKinds returns docs whose `kind:` field is not in the excluded set.
// Use this to drop resource types that the test utility's apply loop doesn't
// support (e.g. ValidatingAdmissionPolicy from the Gateway API bundle).
func FilterKinds(docs []string, excluded ...string) []string {
	skip := make(map[string]bool, len(excluded))
	for _, k := range excluded {
		skip[strings.ToLower(k)] = true
	}
	out := make([]string, 0, len(docs))
	for _, doc := range docs {
		kind := extractKind(doc)
		if !skip[strings.ToLower(kind)] {
			out = append(out, doc)
		}
	}
	return out
}

// extractKind returns the value of the top-level `kind:` field from a YAML doc.
func extractKind(doc string) string {
	for _, line := range strings.Split(doc, "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "kind:") {
			return strings.TrimSpace(strings.TrimPrefix(trimmed, "kind:"))
		}
	}
	return ""
}

// RemoveEmptyLabels strips YAML lines where the llm-d.ai/role label value is
// empty after variable substitution.
//
// This is line-based string surgery: it drops any line ending in `:` that
// contains llm-d.ai/role. It assumes the test manifests use that label only as
// a substitution placeholder at the expected indentation.
func RemoveEmptyLabels(inputs []string) []string {
	outputs := make([]string, len(inputs))
	for idx, input := range inputs {
		lines := strings.Split(input, "\n")
		filtered := make([]string, 0, len(lines))
		for _, line := range lines {
			trimmed := strings.TrimSpace(line)
			if strings.HasSuffix(trimmed, ":") && strings.Contains(trimmed, "llm-d.ai/role") {
				continue
			}
			filtered = append(filtered, line)
		}
		outputs[idx] = strings.Join(filtered, "\n")
	}
	return outputs
}
