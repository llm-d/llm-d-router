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

package manifest

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gexec"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	k8syaml "k8s.io/apimachinery/pkg/util/yaml"
)

func RunKustomize(kustomizeDir string) []string {
	// Use "kubectl kustomize" rather than the standalone "kustomize" binary.
	// CI/dev environments guarantee kubectl but may not have kustomize installed
	// (see Makefile.tools.mk check-kustomize target).
	// #nosec G204 -- Fixed kubectl executable; the kustomize directory is a test-controlled constant passed as a separate argv, without a shell.
	command := exec.Command("kubectl", "kustomize", kustomizeDir)
	session, err := gexec.Start(command, nil, ginkgo.GinkgoWriter)
	gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
	gomega.Eventually(session).WithTimeout(600 * time.Second).Should(gexec.Exit(0))
	return strings.Split(string(session.Out.Contents()), "\n---")
}

// RemoveEmptyArgs strips YAML list items that are empty strings after variable
// substitution (e.g. '- ""' produced when VLLM_EXTRA_ARGS_* is unset).
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

// RemoveEmptyLabels strips YAML lines like "llm-d.ai/role: " where the value
// is empty after variable substitution. Kubernetes accepts empty-value labels,
// but the test pod-selector logic treats the key's presence as meaningful.
func RemoveEmptyLabels(inputs []string) []string {
	outputs := make([]string, len(inputs))
	for idx, input := range inputs {
		lines := strings.Split(input, "\n")
		filtered := make([]string, 0, len(lines))
		for _, line := range lines {
			trimmed := strings.TrimSpace(line)
			// Skip lines like "llm-d.ai/role:" (key with empty value after TrimSpace)
			if strings.HasSuffix(trimmed, ":") {
				if strings.Contains(trimmed, "llm-d.ai/role") {
					continue
				}
			}
			filtered = append(filtered, line)
		}
		outputs[idx] = strings.Join(filtered, "\n")
	}
	return outputs
}

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

// DecodeCaseObjects decodes a multi-document YAML stream into unstructured
// objects and binds them to namespace.
func DecodeCaseObjects(data []byte, namespace string) ([]*unstructured.Unstructured, error) {
	var objects []*unstructured.Unstructured
	decoder := k8syaml.NewYAMLOrJSONDecoder(bytes.NewReader(data), 4096)
	for {
		obj := &unstructured.Unstructured{}
		if err := decoder.Decode(obj); err != nil {
			if errors.Is(err, io.EOF) {
				return objects, nil
			}
			return nil, fmt.Errorf("decode case resource: %w", err)
		}
		if len(obj.Object) == 0 {
			continue
		}
		obj.SetNamespace(namespace)
		objects = append(objects, obj)
	}
}
