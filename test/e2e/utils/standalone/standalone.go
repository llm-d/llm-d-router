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

package standalone

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os/exec"
	"path/filepath"
	goruntime "runtime"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/yaml"

	"github.com/llm-d/llm-d-router/test/e2e/utils"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

// Chart and values paths resolve from this source file because callers run
// with different working directories (the e2e suite runs from test/e2e, unit
// tests from this package).
var (
	_, sourceFile, _, _ = goruntime.Caller(0)
	chartPath           = filepath.Join(filepath.Dir(sourceFile), "..", "..", "..", "..", "config", "charts", "llm-d-router-standalone")
	valuesPath          = filepath.Join(filepath.Dir(sourceFile), "standalone-values.yaml")
)

// Config carries the suite settings the standalone router helpers need.
type Config struct {
	TestConfig    *testutils.TestConfig
	Namespace     string
	EPPImage      string
	HTTPPort      int
	MetricsPort   int
	K8sContext    string
	PodSelector   map[string]string
	ReleaseName   string
	KeepOnFailure bool
}

// Router describes the router one test case rendered and created.
type Router struct {
	// Selector is the router Deployment's pod selector.
	Selector map[string]string
	// PoolName is the rendered InferencePool's name.
	PoolName string

	objects []*unstructured.Unstructured
	cfg     Config
}

// Create renders the standalone chart for one test case, creates all objects,
// and blocks until the router can serve traffic.
func Create(cfg Config, plugins string, replicas int, targetPorts ...int32) *Router {
	ctx, cancel := context.WithTimeout(cfg.TestConfig.Context, cfg.TestConfig.ReadyTimeout)
	defer cancel()
	router, err := renderRouter(ctx, cfg, plugins, replicas, targetPorts)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	access, err := runtime.DefaultUnstructuredConverter.ToUnstructured(router.accessService(cfg.Namespace, cfg.HTTPPort, cfg.MetricsPort))
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	router.objects = append(router.objects, &unstructured.Unstructured{Object: access})
	resources := &utils.CaseResources{Client: cfg.TestConfig.K8sClient}
	var stopForward func()
	utils.DeferCaseCleanup(cfg.TestConfig, cfg.KeepOnFailure, resources, cfg.Namespace, func() {
		if stopForward != nil {
			stopForward()
		}
	})
	ginkgo.By("Creating router resources from " + chartPath)
	gomega.Expect(resources.Create(ctx, router.objects)).To(gomega.Succeed())
	WaitForReadyLeader(cfg, replicas, cfg.Namespace, router.Selector)
	if cfg.K8sContext != "" {
		stopForward = startPortForward(cfg, router.Selector)
	}
	router.WaitForRouting()
	return router
}

// WaitForRouting blocks until the proxy answers a request and the EPP has
// discovered the model server Pods.
func (r *Router) WaitForRouting() {
	// Envoy's active health check can lag behind Kubernetes readiness.
	ginkgo.By("Waiting for the standalone proxy to reach EPP")
	probe := &http.Client{Timeout: 5 * time.Second}
	gomega.Eventually(func() bool {
		resp, err := probe.Get(fmt.Sprintf("http://localhost:%d/v1/models", r.cfg.HTTPPort))
		if err != nil {
			return false
		}
		defer func() { _ = resp.Body.Close() }()
		body, err := io.ReadAll(resp.Body)
		return err == nil && (resp.StatusCode == http.StatusOK || len(body) > 0)
	}, r.cfg.TestConfig.ReadyTimeout, time.Second).Should(gomega.BeTrue())
	utils.WaitForEPPToDiscoverPods(r.cfg.TestConfig, r.cfg.MetricsPort, r.PoolName)
}

func renderRouter(ctx context.Context, cfg Config, plugins string, replicas int, targetPorts []int32) (*Router, error) {
	imageValues, err := imageValues(cfg.EPPImage)
	if err != nil {
		return nil, err
	}
	ports := make([]map[string]int32, len(targetPorts))
	for i, port := range targetPorts {
		ports[i] = map[string]int32{"number": port}
	}
	values, err := yaml.Marshal(map[string]any{
		"router": map[string]any{
			"modelServers": map[string]any{"matchLabels": cfg.PodSelector, "targetPorts": ports},
			"epp": map[string]any{
				"image": imageValues, "replicas": replicas,
				"pluginsCustomConfig": map[string]string{"epp-config.yaml": plugins},
			},
		},
	})
	if err != nil {
		return nil, err
	}
	command := exec.CommandContext(ctx, "helm", "template", cfg.ReleaseName, chartPath,
		"--namespace", cfg.Namespace, "-f", valuesPath, "-f", "-")
	command.Stdin = bytes.NewReader(values)
	var stderr bytes.Buffer
	command.Stderr = &stderr
	output, err := command.Output()
	if err != nil {
		return nil, fmt.Errorf("render standalone chart: %w: %s", err, stderr.String())
	}
	objects, err := utils.DecodeCaseObjects(output, cfg.Namespace)
	if err != nil {
		return nil, err
	}
	router := &Router{objects: objects, cfg: cfg}
	for _, obj := range objects {
		switch obj.GetKind() {
		case "Deployment":
			if router.Selector != nil {
				return nil, errors.New("standalone chart must have one router Deployment")
			}
			router.Selector, _, err = unstructured.NestedStringMap(obj.Object, "spec", "selector", "matchLabels")
			if err != nil {
				return nil, err
			}
		case "InferencePool":
			router.PoolName = obj.GetName()
		}
	}
	if len(router.Selector) == 0 || router.PoolName == "" {
		return nil, errors.New("standalone chart must render a router selector and InferencePool")
	}
	return router, nil
}

func imageValues(image string) (map[string]string, error) {
	if image == "" || strings.ContainsAny(image, "@ \t\n") {
		return nil, fmt.Errorf("EPP_IMAGE must be a tagged image name, got %q", image)
	}
	name, tag := image, "latest"
	if colon := strings.LastIndex(image, ":"); colon > strings.LastIndex(image, "/") {
		name, tag = image[:colon], image[colon+1:]
	}
	registry, repository := "docker.io", name
	if first, rest, found := strings.Cut(name, "/"); found {
		if strings.ContainsAny(first, ".:") || first == "localhost" {
			registry, repository = first, rest
		}
	} else {
		repository = "library/" + name
	}
	if name == "" || repository == "" || tag == "" {
		return nil, fmt.Errorf("invalid EPP_IMAGE %q", image)
	}
	return map[string]string{"registry": registry, "repository": repository, "tag": tag, "pullPolicy": "IfNotPresent"}, nil
}

func (r *Router) accessService(namespace string, httpPort, metricsPort int) *corev1.Service {
	return &corev1.Service{
		TypeMeta:   metav1.TypeMeta{APIVersion: "v1", Kind: "Service"},
		ObjectMeta: metav1.ObjectMeta{Name: "router-access", Namespace: namespace},
		Spec: corev1.ServiceSpec{
			Type: corev1.ServiceTypeNodePort, Selector: r.Selector,
			Ports: []corev1.ServicePort{
				{Name: "http", Port: 8081, TargetPort: intstr.FromInt32(8081), NodePort: int32(httpPort)},
				{Name: "metrics", Port: 9090, TargetPort: intstr.FromInt32(9090), NodePort: int32(metricsPort)},
			},
		},
	}
}

// WaitForReadyLeader blocks until replicas live router Pods exist and exactly
// one is Ready, then returns that leader Pod.
func WaitForReadyLeader(cfg Config, replicas int, namespace string, selector map[string]string) *corev1.Pod {
	var leaderPod *corev1.Pod
	gomega.Eventually(func() error {
		podList := &corev1.PodList{}
		if err := cfg.TestConfig.K8sClient.List(cfg.TestConfig.Context, podList, client.InNamespace(namespace), client.MatchingLabels(selector)); err != nil {
			return err
		}
		var err error
		leaderPod, err = readyLeader(podList.Items, replicas)
		return err
	}, cfg.TestConfig.ReadyTimeout, cfg.TestConfig.Interval).Should(gomega.Succeed())
	return leaderPod
}

// ReadyRouterPod reports whether both router containers are running and Ready.
func ReadyRouterPod(pod *corev1.Pod) bool {
	if pod.DeletionTimestamp != nil || pod.Status.Phase != corev1.PodRunning || len(pod.Status.ContainerStatuses) != 2 {
		return false
	}
	for _, container := range pod.Status.ContainerStatuses {
		if !container.Ready || container.State.Running == nil {
			return false
		}
	}
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady && condition.Status == corev1.ConditionTrue {
			return true
		}
	}
	return false
}

func readyLeader(pods []corev1.Pod, replicas int) (*corev1.Pod, error) {
	live, ready := 0, 0
	var leader *corev1.Pod
	for _, pod := range pods {
		if pod.DeletionTimestamp != nil {
			continue
		}
		live++
		if pod.Status.Phase != corev1.PodRunning || len(pod.Status.ContainerStatuses) != 2 {
			return nil, fmt.Errorf("router Pod %s has not started both containers", pod.Name)
		}
		for _, container := range pod.Status.ContainerStatuses {
			if container.State.Running == nil || (container.Name == "envoy-proxy" && !container.Ready) {
				return nil, fmt.Errorf("router container %s/%s is not running", pod.Name, container.Name)
			}
		}
		if ReadyRouterPod(&pod) {
			ready++
			leader = pod.DeepCopy()
		}
	}
	if live != replicas || ready != 1 {
		return nil, fmt.Errorf("router has %d live Pods and %d Ready Pods, want %d and 1", live, ready, replicas)
	}
	return leader, nil
}

type forwardProcess struct {
	done <-chan struct{}
	stop func()
}

type routerPortForward struct {
	podUID  types.UID
	process *forwardProcess
	start   func(context.Context, *corev1.Pod) (*forwardProcess, error)
}

func (f *routerPortForward) close() {
	if f.process != nil {
		f.process.stop()
		<-f.process.done
		f.process = nil
		f.podUID = ""
	}
}

func (f *routerPortForward) reconcile(ctx context.Context, pods []corev1.Pod) error {
	var leader *corev1.Pod
	for _, pod := range pods {
		if ReadyRouterPod(&pod) {
			if leader != nil {
				f.close()
				return errors.New("multiple Ready router Pods")
			}
			leader = pod.DeepCopy()
		}
	}
	if f.process != nil {
		select {
		case <-f.process.done:
			f.close()
		default:
		}
	}
	if leader == nil || leader.UID != f.podUID {
		f.close()
	}
	if leader == nil || f.process != nil {
		return nil
	}
	process, err := f.start(ctx, leader)
	if err != nil {
		return err
	}
	f.process, f.podUID = process, leader.UID
	return nil
}

// startPortForward follows the Ready leader Pod with a kubectl port-forward
// subprocess and returns a function that stops it.
func startPortForward(cfg Config, selector map[string]string) func() {
	ctx, cancel := context.WithCancel(cfg.TestConfig.Context)
	done := make(chan struct{})
	forward := &routerPortForward{start: func(ctx context.Context, pod *corev1.Pod) (*forwardProcess, error) {
		// #nosec G204 -- Fixed kubectl executable; API Pod names, integer ports and test settings are separate argv, without a shell.
		command := exec.CommandContext(ctx, "kubectl", "port-forward", "pod/"+pod.Name,
			fmt.Sprintf("%d:8081", cfg.HTTPPort), fmt.Sprintf("%d:9090", cfg.MetricsPort),
			"--context="+cfg.K8sContext, "--namespace="+cfg.Namespace, "--address=127.0.0.1")
		command.Stdout, command.Stderr = ginkgo.GinkgoWriter, ginkgo.GinkgoWriter
		if err := command.Start(); err != nil {
			return nil, err
		}
		exited := make(chan struct{})
		go func() {
			_ = command.Wait()
			close(exited)
		}()
		return &forwardProcess{done: exited, stop: func() { _ = command.Process.Kill() }}, nil
	}}
	go func() {
		defer close(done)
		defer forward.close()
		ticker := time.NewTicker(cfg.TestConfig.Interval)
		defer ticker.Stop()
		for {
			pods := &corev1.PodList{}
			err := cfg.TestConfig.K8sClient.List(ctx, pods, client.InNamespace(cfg.Namespace), client.MatchingLabels(selector))
			if err == nil {
				err = forward.reconcile(ctx, pods.Items)
			}
			if err != nil && ctx.Err() == nil {
				ginkgo.GinkgoLogr.Error(err, "Router port-forward will retry")
			}
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
			}
		}
	}()
	return func() {
		cancel()
		<-done
	}
}
