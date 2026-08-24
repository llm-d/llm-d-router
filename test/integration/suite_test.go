//go:build integration_tests
// +build integration_tests

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

package integration_test

import (
	"fmt"
	"net/http"
	"os"
	"testing"

	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/clientcmd"
	gwapiv1 "sigs.k8s.io/gateway-api/apis/v1"

	"github.com/llm-d/llm-d-router/apix/v1alpha2"
)

const (
	gatewayURL = "http://localhost:30080" // TODO: make configurable
)

var (
	kube *kubernetes.Clientset
)

func TestMain(m *testing.M) {
	if err := initializeKubernetesClient(); err != nil {
		fmt.Fprintf(os.Stderr, "failed to initialize kubernetes client: %v\n", err)
		os.Exit(1)
	}

	if err := initializeGateway(); err != nil {
		fmt.Fprintf(os.Stderr, "failed to initialize gateway: %v\n", err)
		os.Exit(1)
	}

	code := m.Run()
	os.Exit(code)
}

func initializeKubernetesClient() error {
	kubeConfigPath := os.Getenv("KUBECONFIG")
	if kubeConfigPath == "" {
		return fmt.Errorf("no KUBECONFIG set")
	}

	kubeConfig, err := clientcmd.BuildConfigFromFlags("", kubeConfigPath)
	if err != nil {
		return err
	}

	if err := gwapiv1.Install(scheme.Scheme); err != nil {
		return err
	}

	if err := v1alpha2.Install(scheme.Scheme); err != nil {
		return err
	}

	kube, err = kubernetes.NewForConfig(kubeConfig)
	if err != nil {
		return err
	}

	_, err = kube.ServerVersion()
	if err != nil {
		return fmt.Errorf("request to kubernetes api failed: %w", err)
	}

	return nil
}

func initializeGateway() (err error) {
	resp, err := http.Get(gatewayURL)
	if err != nil {
		return err
	}

	if resp.StatusCode != http.StatusNotFound {
		return fmt.Errorf("expected gateway to return 404, found: %s", resp.Status)
	}

	serverHeader := resp.Header.Get("Server")
	if serverHeader == "" {
		return fmt.Errorf(`expected gateway to return "istio-envoy" server header, found no value`)
	}
	if serverHeader != "istio-envoy" {
		return fmt.Errorf(`expected gateway to return "istio-envoy" server header, found: %s`, serverHeader)
	}

	return nil
}
