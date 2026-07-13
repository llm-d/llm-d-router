/*
Copyright 2024 The Kubernetes Authors.

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

package utils

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apiextv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/config"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	"github.com/llm-d/llm-d-router/pkg/epp/util/env"
)

const (
	// defaultExistsTimeout is the default timeout for a resource to exist in the api server.
	defaultExistsTimeout = 30 * time.Second
	// defaultReadyTimeout is the default timeout for a resource to report a ready state.
	defaultReadyTimeout = 3 * time.Minute
	// defaultModelReadyTimeout is the default timeout for the model server deployment to report a ready state.
	defaultModelReadyTimeout = 10 * time.Minute
	// defaultInterval is the default interval to check if a resource exists or ready conditions.
	defaultInterval = time.Millisecond * 250
)

// TestConfig groups various fields together for use in the test helpers
type TestConfig struct {
	Context           context.Context
	KubeCli           *kubernetes.Clientset
	K8sClient         client.Client
	RestConfig        *rest.Config
	NsName            string
	Scheme            *runtime.Scheme
	ExistsTimeout     time.Duration
	ReadyTimeout      time.Duration
	ModelReadyTimeout time.Duration
	Interval          time.Duration
}

// NewTestConfig creates a new TestConfig instance
func NewTestConfig(nsName string, k8sContext string) *TestConfig {
	cfg, err := config.GetConfigWithContext(k8sContext)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	gomega.Expect(cfg).NotTo(gomega.BeNil())

	kubeCli, err := kubernetes.NewForConfig(cfg)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	gomega.Expect(kubeCli).NotTo(gomega.BeNil())

	ginkgo.By("API server endpoint: " + cfg.Host)
	loadingRules := clientcmd.NewDefaultClientConfigLoadingRules()
	configOverrides := &clientcmd.ConfigOverrides{}
	if k8sContext != "" {
		configOverrides.CurrentContext = k8sContext
	}
	kubeConfig := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(loadingRules, configOverrides)
	rawConfig, err := kubeConfig.RawConfig()
	if err == nil {
		ctxName := rawConfig.CurrentContext
		if k8sContext != "" {
			ctxName = k8sContext
		}
		ginkgo.By("Kubeconfig context: " + ctxName)
		if ctx, ok := rawConfig.Contexts[ctxName]; ok {
			ginkgo.By(fmt.Sprintf("Cluster: %s, AuthInfo: %s", ctx.Cluster, ctx.AuthInfo))
		}
	}
	serverVersion, err := kubeCli.Discovery().ServerVersion()
	if err == nil {
		ginkgo.By("Kubernetes server version: " + serverVersion.GitVersion)
	}

	return &TestConfig{
		Context:           context.Background(),
		KubeCli:           kubeCli,
		NsName:            nsName,
		RestConfig:        cfg,
		Scheme:            runtime.NewScheme(),
		ExistsTimeout:     env.GetEnvDuration("EXISTS_TIMEOUT", defaultExistsTimeout, ginkgo.GinkgoLogr),
		ReadyTimeout:      env.GetEnvDuration("READY_TIMEOUT", defaultReadyTimeout, ginkgo.GinkgoLogr),
		ModelReadyTimeout: env.GetEnvDuration("MODEL_READY_TIMEOUT", defaultModelReadyTimeout, ginkgo.GinkgoLogr),
		Interval:          defaultInterval,
	}
}

// CreateCli creates the Kubernetes client used in the tests, invoked after the scheme has been setup.
func (testConfig *TestConfig) CreateCli() {
	var err error
	testConfig.K8sClient, err = client.New(testConfig.RestConfig, client.Options{Scheme: testConfig.Scheme})
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	gomega.Expect(testConfig.K8sClient).NotTo(gomega.BeNil())
}

// PodReady checks if the given Pod reports the "Ready" status condition before the given timeout.
func PodReady(testConfig *TestConfig, pod *corev1.Pod) {
	ginkgo.By(fmt.Sprintf("Checking pod %s/%s status is: %s", pod.Namespace, pod.Name, corev1.PodReady))
	conditions := []corev1.PodCondition{
		{
			Type:   corev1.PodReady,
			Status: corev1.ConditionTrue,
		},
	}
	gomega.Eventually(checkPodStatus, testConfig.ExistsTimeout, testConfig.Interval).
		WithArguments(testConfig, pod, conditions).Should(gomega.BeTrue())
}

// checkPodStatus checks if the given Pod status matches the expected conditions.
func checkPodStatus(testConfig *TestConfig, pod *corev1.Pod, conditions []corev1.PodCondition) (bool, error) {
	var fetchedPod corev1.Pod
	if err := testConfig.K8sClient.Get(testConfig.Context, types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}, &fetchedPod); err != nil {
		return false, err
	}
	found := 0
	for _, want := range conditions {
		for _, c := range fetchedPod.Status.Conditions {
			if c.Type == want.Type && c.Status == want.Status {
				found++
			}
		}
	}
	return found == len(conditions), nil
}

// DeploymentAvailable checks if the given Deployment reports the "Available" status condition before the given timeout.
func DeploymentAvailable(testConfig *TestConfig, deploy *appsv1.Deployment) {
	ginkgo.By(fmt.Sprintf("Checking if deployment %s/%s status is: %s", deploy.Namespace, deploy.Name, appsv1.DeploymentAvailable))
	conditions := []appsv1.DeploymentCondition{
		{
			Type:   appsv1.DeploymentAvailable,
			Status: corev1.ConditionTrue,
		},
	}
	gomega.Eventually(checkDeploymentStatus, testConfig.ModelReadyTimeout, testConfig.Interval).
		WithArguments(testConfig.Context, testConfig.K8sClient, deploy, conditions).
		Should(gomega.BeTrue())
}

// DeploymentReadyReplicas checks if the given Deployment has at least `count` ready replicas before the given timeout.
func DeploymentReadyReplicas(testConfig *TestConfig, deploy *appsv1.Deployment, count int) {
	ginkgo.By(fmt.Sprintf("Checking if deployment %s/%s has at least %d ready replica(s)", deploy.Namespace, deploy.Name, count))
	gomega.Eventually(func(g gomega.Gomega) {
		var fetchedDeploy appsv1.Deployment
		err := testConfig.K8sClient.Get(testConfig.Context, types.NamespacedName{Namespace: deploy.Namespace, Name: deploy.Name}, &fetchedDeploy)
		g.Expect(err).NotTo(gomega.HaveOccurred())
		g.Expect(fetchedDeploy.Status.ReadyReplicas).To(gomega.BeNumerically(">=", count),
			fmt.Sprintf("Deployment only has %d ready replicas, want at least %d", fetchedDeploy.Status.ReadyReplicas, count))
	}, testConfig.ModelReadyTimeout, testConfig.Interval).Should(gomega.Succeed())
}

// checkDeploymentStatus checks if the given Deployment status matches the expected conditions.
func checkDeploymentStatus(ctx context.Context, cli client.Client, deploy *appsv1.Deployment, conditions []appsv1.DeploymentCondition) (bool, error) {
	var fetchedDeploy appsv1.Deployment
	if err := cli.Get(ctx, types.NamespacedName{Namespace: deploy.Namespace, Name: deploy.Name}, &fetchedDeploy); err != nil {
		return false, err
	}
	found := 0
	for _, want := range conditions {
		for _, c := range fetchedDeploy.Status.Conditions {
			if c.Type == want.Type && c.Status == want.Status {
				found++
			}
		}
	}
	return found == len(conditions), nil
}

// CRDEstablished checks if the given CRD reports the "Established" status condition before the given timeout.
func CRDEstablished(testConfig *TestConfig, crd *apiextv1.CustomResourceDefinition) {
	ginkgo.By(fmt.Sprintf("Checking CRD %s status is: %s", crd.Name, apiextv1.Established))
	conditions := []apiextv1.CustomResourceDefinitionCondition{
		{
			Type:   apiextv1.Established,
			Status: apiextv1.ConditionTrue,
		},
	}
	gomega.Eventually(checkCrdStatus, testConfig.ReadyTimeout, testConfig.Interval).
		WithArguments(testConfig.Context, testConfig.K8sClient, crd, conditions).
		Should(gomega.BeTrue())
}

// checkCrdStatus checks if the given CRD status matches the expected conditions.
func checkCrdStatus(
	ctx context.Context,
	cli client.Client,
	crd *apiextv1.CustomResourceDefinition,
	conditions []apiextv1.CustomResourceDefinitionCondition,
) (bool, error) {
	var fetchedCrd apiextv1.CustomResourceDefinition
	if err := cli.Get(ctx, types.NamespacedName{Name: crd.Name}, &fetchedCrd); err != nil {
		return false, err
	}
	found := 0
	for _, want := range conditions {
		for _, c := range fetchedCrd.Status.Conditions {
			if c.Type == want.Type && c.Status == want.Status {
				found++
			}
		}
	}
	return found == len(conditions), nil
}

// EventuallyExists checks if a Kubernetes resource exists and returns nil if successful.
// It takes a function `getResource` which retrieves the resource and returns an error if it doesn't exist.
func EventuallyExists(testConfig *TestConfig, getResource func() error) {
	gomega.Eventually(func() error {
		return getResource()
	}, testConfig.ExistsTimeout, testConfig.Interval).Should(gomega.Succeed())
}

func CreateAndVerifyObjs(testConfig *TestConfig, objs []*unstructured.Unstructured) []string {
	objNames := CreateObjsWithVerifier(testConfig, objs,
		func(kind string, clientObj client.Object) {
			switch kind {
			case "CustomResourceDefinition":
				CRDEstablished(testConfig, clientObj.(*apiextv1.CustomResourceDefinition))
			case "Deployment":
				DeploymentAvailable(testConfig, clientObj.(*appsv1.Deployment))
			case "Pod":
				PodReady(testConfig, clientObj.(*corev1.Pod))
			}
		})

	return objNames
}

func CreateObjsWithVerifier(testConfig *TestConfig, objs []*unstructured.Unstructured, verifier func(kind string, clientObj client.Object)) []string {
	objNames := make([]string, len(objs))
	for idx, unstrObj := range objs {
		ginkgo.By(fmt.Sprintf("Processing GVK: %s", unstrObj.GroupVersionKind()))
		unstrObj.SetNamespace(testConfig.NsName)

		kind := unstrObj.GetKind()
		name := unstrObj.GetName()
		objNames[idx] = kind + "/" + name

		err := testConfig.K8sClient.Create(testConfig.Context, unstrObj, &client.CreateOptions{})
		if apierrors.IsAlreadyExists(err) {
			ginkgo.By(fmt.Sprintf("%s %s already exists, skipping creation", kind, name))
		} else {
			gomega.Expect(err).NotTo(gomega.HaveOccurred(),
				fmt.Sprintf("Failed to create %s %s", kind, name))
		}

		clientObj := getClientObject(kind)
		EventuallyExists(testConfig, func() error {
			return testConfig.K8sClient.Get(testConfig.Context,
				types.NamespacedName{Namespace: testConfig.NsName, Name: name}, clientObj)
		})

		verifier(kind, clientObj)
	}
	return objNames
}

// CreateObjsFromYaml creates K8S objects from yaml and waits for them to be instantiated
func CreateObjsFromYaml(testConfig *TestConfig, docs []string) []string {
	objs := CreateUnstructuredObjs(testConfig, docs)
	return CreateAndVerifyObjs(testConfig, objs)
}

// CreateUnstructuredObjs creates K8S UnstructuredObject structs from an array of YAMLs
func CreateUnstructuredObjs(testConfig *TestConfig, docs []string) []*unstructured.Unstructured {
	objs := make([]*unstructured.Unstructured, 0, len(docs))
	decoder := serializer.NewCodecFactory(testConfig.Scheme).UniversalDeserializer()

	for _, doc := range docs {
		trimmed := strings.TrimSpace(doc)
		if trimmed == "" {
			continue
		}
		// Decode into a runtime.Object
		obj, gvk, decodeErr := decoder.Decode([]byte(trimmed), nil, nil)
		gomega.Expect(decodeErr).NotTo(gomega.HaveOccurred(),
			"Failed to decode YAML document to a Kubernetes object")

		ginkgo.By(fmt.Sprintf("Decoded GVK: %s", gvk))

		unstrObj, ok := obj.(*unstructured.Unstructured)
		if !ok {
			// Fallback if it's a typed object
			unstrObj = &unstructured.Unstructured{}
			// Convert typed to unstructured
			err := testConfig.Scheme.Convert(obj, unstrObj, nil)
			gomega.Expect(err).NotTo(gomega.HaveOccurred())
		}
		objs = append(objs, unstrObj)
	}
	return objs
}

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

// ApplyYAMLFile reads a file containing YAML (possibly multiple docs)
// and applies each object to the cluster.
func ApplyYAMLFile(testConfig *TestConfig, filePath string) []string {
	// Create the resources from the manifest file
	return CreateObjsFromYaml(testConfig, ReadYaml(filePath))
}

// ReadYaml is a helper function to read in K8S YAML files and split by the --- separator
func ReadYaml(filePath string) []string {
	ginkgo.By("Reading YAML file: " + filePath)
	yamlBytes, err := os.ReadFile(filePath)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	// Split multiple docs, if needed
	return strings.Split(string(yamlBytes), "\n---")
}

// ValidateCRDsEstablished verifies that each of the given CRD names is established in the cluster.
func ValidateCRDsEstablished(testConfig *TestConfig, crdNames []string) {
	for _, name := range crdNames {
		crd := &apiextv1.CustomResourceDefinition{
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
			},
		}
		CRDEstablished(testConfig, crd)
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
		return &v1.InferencePool{}
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
		ginkgo.Fail("unsupported K8S kind "+kind, 1)
		return nil
	}
}
