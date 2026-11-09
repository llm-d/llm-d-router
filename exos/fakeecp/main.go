/*
Copyright 2026 Google LLC.

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

// Command fakeecp stands in for the Exos Control Plane (ECP) in the GKE PoC.
//
// Every interval it:
//   - reads linked-runner.runner-instance/<runner> annotations (same JSON as the
//     linked-runner controller writes: NAME, STATUS, IP_ADDRESS, PROVIDER_ID) on
//     the nodes matching --node-selector;
//   - probes http://<IP_ADDRESS>:<port><health-path> for each RUNNING runner;
//   - keeps a selector-less headless Service and one EndpointSlice per runner
//     (glm52-server-<runner>-borneo, as ECP names them) whose ready/serving
//     conditions follow the probe;
//   - deletes slices it manages for runners that are gone or not RUNNING.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/utils/ptr"
)

const (
	runnerAnnotationPrefix = "linked-runner.runner-instance/"
	managedByValue         = "fake-ecp"
)

type config struct {
	namespace    string
	service      string
	port         int
	nodeSelector string
	healthPath   string
	interval     time.Duration
	probeTimeout time.Duration
	kubeconfig   string
}

type runner struct {
	Name       string `json:"NAME"`
	Status     string `json:"STATUS"`
	IPAddress  string `json:"IP_ADDRESS"`
	ProviderID string `json:"PROVIDER_ID"`
}

func main() {
	var c config
	flag.StringVar(&c.namespace, "namespace", "default", "namespace of the headless Service and EndpointSlices")
	flag.StringVar(&c.service, "service", "glm52-server", "headless Service name")
	flag.IntVar(&c.port, "port", 9080, "model server port on the runners")
	flag.StringVar(&c.nodeSelector, "node-selector", "exos-poc/control-node=true", "label selector of control nodes")
	flag.StringVar(&c.healthPath, "health-path", "/health", "health probe path")
	flag.DurationVar(&c.interval, "interval", 5*time.Second, "reconcile interval")
	flag.DurationVar(&c.probeTimeout, "probe-timeout", 2*time.Second, "health probe timeout")
	flag.StringVar(&c.kubeconfig, "kubeconfig", "", "kubeconfig path; in-cluster config when empty")
	flag.Parse()

	cfg, err := restConfig(c.kubeconfig)
	if err != nil {
		log.Fatalf("client config: %v", err)
	}
	cs := kubernetes.NewForConfigOrDie(cfg)
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	probe := &http.Client{Timeout: c.probeTimeout}
	log.Printf("fake-ecp starting: %+v", c)
	t := time.NewTicker(c.interval)
	defer t.Stop()
	for {
		if err := reconcile(ctx, cs, probe, c); err != nil {
			log.Printf("reconcile error: %v", err)
		}
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
	}
}

func restConfig(kubeconfig string) (*rest.Config, error) {
	if kubeconfig != "" {
		return clientcmd.BuildConfigFromFlags("", kubeconfig)
	}
	return rest.InClusterConfig()
}

func reconcile(ctx context.Context, cs kubernetes.Interface, probe *http.Client, c config) error {
	nodes, err := cs.CoreV1().Nodes().List(ctx, metav1.ListOptions{LabelSelector: c.nodeSelector})
	if err != nil {
		return fmt.Errorf("listing control nodes: %w", err)
	}
	runners := map[string]runner{}
	for _, n := range nodes.Items {
		for k, v := range n.Annotations {
			if !strings.HasPrefix(k, runnerAnnotationPrefix) {
				continue
			}
			var r runner
			if err := json.Unmarshal([]byte(v), &r); err != nil {
				log.Printf("node %s annotation %s: %v", n.Name, k, err)
				continue
			}
			if r.Name == "" {
				r.Name = strings.TrimPrefix(k, runnerAnnotationPrefix)
			}
			if r.Status != "RUNNING" || net.ParseIP(r.IPAddress) == nil {
				continue
			}
			runners[r.Name] = r
		}
	}

	if len(runners) > 0 {
		if err := ensureService(ctx, cs, c, firstFamily(runners)); err != nil {
			return err
		}
	}

	desired := map[string]bool{}
	for _, r := range runners {
		name := sliceName(c.service, r.Name)
		desired[name] = true
		ready := healthy(ctx, probe, r.IPAddress, c)
		if err := applySlice(ctx, cs, c, name, r, ready); err != nil {
			log.Printf("runner %s: %v", r.Name, err)
		}
	}

	slices, err := cs.DiscoveryV1().EndpointSlices(c.namespace).List(ctx, metav1.ListOptions{
		LabelSelector: discoveryv1.LabelManagedBy + "=" + managedByValue + "," + discoveryv1.LabelServiceName + "=" + c.service,
	})
	if err != nil {
		return fmt.Errorf("listing managed slices: %w", err)
	}
	for _, s := range slices.Items {
		if !desired[s.Name] {
			log.Printf("deleting slice %s (runner gone or not RUNNING)", s.Name)
			if err := cs.DiscoveryV1().EndpointSlices(c.namespace).Delete(ctx, s.Name, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
				log.Printf("deleting slice %s: %v", s.Name, err)
			}
		}
	}
	return nil
}

func sliceName(service, runner string) string { return service + "-" + runner + "-borneo" }

func family(ip string) (corev1.IPFamily, discoveryv1.AddressType) {
	if strings.Contains(ip, ":") {
		return corev1.IPv6Protocol, discoveryv1.AddressTypeIPv6
	}
	return corev1.IPv4Protocol, discoveryv1.AddressTypeIPv4
}

func firstFamily(runners map[string]runner) corev1.IPFamily {
	for _, r := range runners {
		f, _ := family(r.IPAddress)
		return f
	}
	return corev1.IPv4Protocol
}

func healthy(ctx context.Context, probe *http.Client, ip string, c config) bool {
	url := "http://" + net.JoinHostPort(ip, strconv.Itoa(c.port)) + c.healthPath
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return false
	}
	resp, err := probe.Do(req)
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// ensureService mirrors ecp ensureHeadlessServiceInNamespace: selector-less,
// headless, single-stack in the runners' address family.
func ensureService(ctx context.Context, cs kubernetes.Interface, c config, fam corev1.IPFamily) error {
	svc := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: c.service, Namespace: c.namespace,
			Labels: map[string]string{"app": "sglang-runner-tunnel", "exos/service": c.service},
		},
		Spec: corev1.ServiceSpec{
			ClusterIP:      corev1.ClusterIPNone,
			IPFamilyPolicy: ptr.To(corev1.IPFamilyPolicySingleStack),
			IPFamilies:     []corev1.IPFamily{fam},
			Ports:          []corev1.ServicePort{{Name: "http", Port: int32(c.port), Protocol: corev1.ProtocolTCP}},
		},
	}
	_, err := cs.CoreV1().Services(c.namespace).Create(ctx, svc, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		return fmt.Errorf("creating service %s: %w", c.service, err)
	}
	return nil
}

// applySlice mirrors ecp applyEndpointSliceInNamespace, except that ready and
// serving follow the health probe instead of being always true.
func applySlice(ctx context.Context, cs kubernetes.Interface, c config, name string, r runner, ready bool) error {
	_, addrType := family(r.IPAddress)
	want := &discoveryv1.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name: name, Namespace: c.namespace,
			Labels: map[string]string{
				discoveryv1.LabelServiceName: c.service,
				discoveryv1.LabelManagedBy:   managedByValue,
				"exos/runner-id":             r.Name,
				"exos/mode":                  "borneo",
			},
			Annotations: map[string]string{"exos/provider-id": r.ProviderID},
		},
		AddressType: addrType,
		Ports:       []discoveryv1.EndpointPort{{Name: ptr.To("http"), Port: ptr.To(int32(c.port)), Protocol: ptr.To(corev1.ProtocolTCP)}},
		Endpoints: []discoveryv1.Endpoint{{
			Addresses:  []string{r.IPAddress},
			Hostname:   ptr.To(r.Name),
			Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(ready), Serving: ptr.To(ready)},
		}},
	}
	slices := cs.DiscoveryV1().EndpointSlices(c.namespace)
	got, err := slices.Get(ctx, name, metav1.GetOptions{})
	switch {
	case apierrors.IsNotFound(err):
		_, err = slices.Create(ctx, want, metav1.CreateOptions{})
		if err == nil {
			log.Printf("created slice %s address=%s ready=%v", name, r.IPAddress, ready)
		}
		return err
	case err != nil:
		return err
	}
	if got.AddressType != want.AddressType {
		// addressType is immutable: recreate.
		if err := slices.Delete(ctx, name, metav1.DeleteOptions{}); err != nil {
			return err
		}
		_, err = slices.Create(ctx, want, metav1.CreateOptions{})
		return err
	}
	if sameSlice(got, want) {
		return nil
	}
	want.ResourceVersion = got.ResourceVersion
	if _, err := slices.Update(ctx, want, metav1.UpdateOptions{}); err != nil {
		return err
	}
	log.Printf("updated slice %s address=%s ready=%v", name, r.IPAddress, ready)
	return nil
}

func sameSlice(a, b *discoveryv1.EndpointSlice) bool {
	if len(a.Endpoints) != 1 || len(b.Endpoints) != 1 || len(a.Ports) != 1 {
		return false
	}
	ea, eb := a.Endpoints[0], b.Endpoints[0]
	return len(ea.Addresses) == 1 && ea.Addresses[0] == eb.Addresses[0] &&
		ptr.Deref(ea.Conditions.Ready, false) == ptr.Deref(eb.Conditions.Ready, false) &&
		ptr.Deref(a.Ports[0].Port, 0) == ptr.Deref(b.Ports[0].Port, 0) &&
		a.Labels["exos/runner-id"] == b.Labels["exos/runner-id"]
}
