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

// Package utils provides test utilities for the llm-d inference scheduler.
// DeleteObjects and getClientObject restore the function removed from
// sigs.k8s.io/gateway-api-inference-extension/test/utils in v1.5.0.
package utils

import (
	"strings"

	"github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apiextv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	gaiev1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	igwtestutils "github.com/llm-d/llm-d-router/test/utils/igw"
)

type TestConfig = igwtestutils.TestConfig

var (
	NewTestConfig      = igwtestutils.NewTestConfig
	ApplyYAMLFile      = igwtestutils.ApplyYAMLFile
	CreateObjsFromYaml = igwtestutils.CreateObjsFromYaml
	ReadYaml           = igwtestutils.ReadYaml
)

// DeleteObjects deletes a set of Kubernetes objects in the form of kind/name.
func DeleteObjects(testConfig *TestConfig, kindAndNames []string) {
	for _, kindAndName := range kindAndNames {
		split := strings.Split(kindAndName, "/")
		clientObj := getClientObject(split[0])
		err := testConfig.K8sClient.Get(testConfig.Context,
			types.NamespacedName{Namespace: testConfig.NsName, Name: split[1]}, clientObj)
		gomega.Expect(err).NotTo(gomega.HaveOccurred())
		err = testConfig.K8sClient.Delete(testConfig.Context, clientObj)
		gomega.Expect(err).NotTo(gomega.HaveOccurred())
		gomega.Eventually(func() bool {
			clientObj := getClientObject(split[0])
			err := testConfig.K8sClient.Get(testConfig.Context,
				types.NamespacedName{Namespace: testConfig.NsName, Name: split[1]}, clientObj)
			return apierrors.IsNotFound(err)
		}, testConfig.ExistsTimeout, testConfig.Interval).Should(gomega.BeTrue())
	}
}

func getClientObject(kind string) client.Object {
	switch strings.ToLower(kind) {
	case "clusterrole":
		return &rbacv1.ClusterRole{}
	case "clusterrolebinding":
		return &rbacv1.ClusterRoleBinding{}
	case "configmap":
		return &corev1.ConfigMap{}
	case "customresourcedefinition":
		return &apiextv1.CustomResourceDefinition{}
	case "deployment":
		return &appsv1.Deployment{}
	case "inferencepool":
		return &gaiev1.InferencePool{}
	case "pod":
		return &corev1.Pod{}
	case "role":
		return &rbacv1.Role{}
	case "rolebinding":
		return &rbacv1.RoleBinding{}
	case "secret":
		return &corev1.Secret{}
	case "service":
		return &corev1.Service{}
	case "serviceaccount":
		return &corev1.ServiceAccount{}
	default:
		panic("unsupported K8S kind: " + kind)
	}
}
