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

// Package simecp reproduces, against any Kubernetes API server, the objects
// that exist in an Exos linked-runner GKE cluster with Borneo runners:
//
//   - a GKE control node carrying one linked-runner.runner-instance/<runner>
//     annotation per linked Borneo runner (written by the linked-runner
//     controller in production);
//   - hostNetwork model-server pods bound to that control node, whose podIP is
//     the control node's IP (what kubelet reports for a linked-runner pod);
//   - what ECP derives from the annotations (ecp/node_annotations.go
//     syncBorneoSlices + server_main.go applyEndpointSliceInNamespace): the
//     selector-less headless Services glm52-server and borg/<user>-<cell>, and
//     per runner the EndpointSlices glm52-server-<runner>-borneo and
//     borg/bns-<runner>-borneo with the runner's Borneo IPv6.
//
// Field values (labels, annotations, ports, hard-coded ready/serving=true, no
// targetRef/nodeName, IPv6 SingleStack) follow ECP at google3 HEAD.
package simecp

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/utils/ptr"
)

const (
	RunnerAnnotationPrefix = "linked-runner.runner-instance/"
	HeadlessSvcName        = "glm52-server" // ecp server_main.go headlessSvcName
	BorgNamespace          = "borg"
	ServicePort            = 9080 // ecp -service_port in prod
)

// Runner is one linked Borneo runner.
type Runner struct {
	Name  string // runner ID, e.g. borneo-p0
	IPv6  string // Borneo guest IPv6 (2606:ec80:10::/44)
	Role  string // exos/role on the model pod: prefill | decode | ""
	User  string // Borg user, e.g. uniserve-dev-jobs
	Cell  string // Borg cell, e.g. pnatlix
	Job   string // Borg job
	Task  int
	Ready bool // what a readiness-aware ECP would write; today's ECP always writes true
}

// Topology is the simulated control node and its runners.
type Topology struct {
	ControlNode   string
	ControlNodeIP string
	ModelNS       string
	Runners       []Runner
}

func (r Runner) providerID() string {
	return fmt.Sprintf("provider://bns/%s/borg/%s/bns/%s/%s/%d", r.Cell, r.Cell, r.User, r.Job, r.Task)
}
func (r Runner) taskJob() string {
	return fmt.Sprintf("%d-%s", r.Task, strings.ReplaceAll(r.Job, "_", "-"))
}
func (r Runner) userCell() string {
	return strings.ReplaceAll(r.User, "_", "-") + "-" + r.Cell
}

// K8sFQDN is the Borg-ordered name the sglang router uses as worker URL host.
func (r Runner) K8sFQDN() string {
	return fmt.Sprintf("%s.%s.%s.svc.cluster.local", r.taskJob(), r.userCell(), BorgNamespace)
}

// SetupCluster creates the control node (with linked-runner annotations) and
// the hostNetwork model-server pods. It does not create Services/slices; call
// Reconcile for that (it plays the role of ECP).
func SetupCluster(ctx context.Context, cs kubernetes.Interface, t Topology) error {
	anns := map[string]string{}
	for _, r := range t.Runners {
		v, _ := json.Marshal(map[string]string{
			"NAME": r.Name, "STATUS": "RUNNING", "IP_ADDRESS": r.IPv6, "PROVIDER_ID": r.providerID(),
		})
		anns[RunnerAnnotationPrefix+r.Name] = string(v)
	}
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:        t.ControlNode,
			Labels:      map[string]string{"cloud.google.com/gke-nodepool": "control-pool"},
			Annotations: anns,
		},
	}
	created, err := cs.CoreV1().Nodes().Create(ctx, node, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		return fmt.Errorf("creating control node: %w", err)
	}
	if err == nil {
		created.Status.Addresses = []corev1.NodeAddress{{Type: corev1.NodeInternalIP, Address: t.ControlNodeIP}}
		created.Status.Conditions = []corev1.NodeCondition{{Type: corev1.NodeReady, Status: corev1.ConditionTrue}}
		if _, err := cs.CoreV1().Nodes().UpdateStatus(ctx, created, metav1.UpdateOptions{}); err != nil {
			return fmt.Errorf("updating control node status: %w", err)
		}
	}

	if err := ensureNamespace(ctx, cs, t.ModelNS); err != nil {
		return err
	}
	for _, r := range t.Runners {
		pod := &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "sglang-" + r.Name,
				Namespace: t.ModelNS,
				Labels:    map[string]string{"app": "glm52", "exos/role": r.Role, "exos/runner-id": r.Name},
			},
			Spec: corev1.PodSpec{
				NodeName:    t.ControlNode,
				HostNetwork: true,
				Containers: []corev1.Container{{
					Name: "sglang", Image: "sglang:fake",
					Ports: []corev1.ContainerPort{{Name: "http", ContainerPort: ServicePort}},
				}},
			},
		}
		p, err := cs.CoreV1().Pods(t.ModelNS).Create(ctx, pod, metav1.CreateOptions{})
		if apierrors.IsAlreadyExists(err) {
			continue
		}
		if err != nil {
			return fmt.Errorf("creating pod %s: %w", pod.Name, err)
		}
		// No kubelet in envtest: write what kubelet reports for a hostNetwork
		// linked-runner pod -- podIP/hostIP are the control node's IP.
		p.Status = corev1.PodStatus{
			Phase:  corev1.PodRunning,
			HostIP: t.ControlNodeIP,
			PodIP:  t.ControlNodeIP,
			PodIPs: []corev1.PodIP{{IP: t.ControlNodeIP}},
			Conditions: []corev1.PodCondition{
				{Type: corev1.PodReady, Status: corev1.ConditionTrue},
				{Type: corev1.ContainersReady, Status: corev1.ConditionTrue},
			},
		}
		if _, err := cs.CoreV1().Pods(t.ModelNS).UpdateStatus(ctx, p, metav1.UpdateOptions{}); err != nil {
			return fmt.Errorf("updating pod %s status: %w", pod.Name, err)
		}
	}
	return nil
}

// Reconcile plays ECP: it reads the control node's linked-runner annotations
// and publishes the headless Services and EndpointSlices. roleLabels adds
// exos/role to the slice labels (ECP does not do this today; set false to
// reproduce today's ECP exactly).
func Reconcile(ctx context.Context, cs kubernetes.Interface, t Topology, roleLabels bool) error {
	node, err := cs.CoreV1().Nodes().Get(ctx, t.ControlNode, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("getting control node: %w", err)
	}
	byName := map[string]Runner{}
	for _, r := range t.Runners {
		byName[r.Name] = r
	}
	if err := ensureHeadlessService(ctx, cs, "default", HeadlessSvcName); err != nil {
		return err
	}
	if err := ensureNamespace(ctx, cs, BorgNamespace); err != nil {
		return err
	}
	for k, v := range node.Annotations {
		if !strings.HasPrefix(k, RunnerAnnotationPrefix) {
			continue
		}
		var ann map[string]string // {"NAME","STATUS","IP_ADDRESS","PROVIDER_ID"}
		if err := json.Unmarshal([]byte(v), &ann); err != nil {
			return fmt.Errorf("annotation %s: %w", k, err)
		}
		r, ok := byName[ann["NAME"]]
		if !ok || ann["STATUS"] != "RUNNING" {
			continue
		}
		extra := map[string]string{}
		if roleLabels && r.Role != "" {
			extra["exos/role"] = r.Role
		}
		if err := ensureHeadlessService(ctx, cs, BorgNamespace, r.userCell()); err != nil {
			return err
		}
		if err := applySlice(ctx, cs, BorgNamespace, r.userCell(), "bns-"+r.Name+"-borneo", r, r.taskJob(), ann["IP_ADDRESS"], extra); err != nil {
			return err
		}
		if err := applySlice(ctx, cs, "default", HeadlessSvcName, HeadlessSvcName+"-"+r.Name+"-borneo", r, r.Name, ann["IP_ADDRESS"], extra); err != nil {
			return err
		}
	}
	return nil
}

// SetRunnerReady rewrites the runner's default-namespace slice with the given
// ready/serving condition (what a readiness-aware ECP would do).
func SetRunnerReady(ctx context.Context, cs kubernetes.Interface, r Runner, ready bool) error {
	name := HeadlessSvcName + "-" + r.Name + "-borneo"
	s, err := cs.DiscoveryV1().EndpointSlices("default").Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return err
	}
	for i := range s.Endpoints {
		s.Endpoints[i].Conditions.Ready = ptr.To(ready)
		s.Endpoints[i].Conditions.Serving = ptr.To(ready)
	}
	_, err = cs.DiscoveryV1().EndpointSlices("default").Update(ctx, s, metav1.UpdateOptions{})
	return err
}

func ensureNamespace(ctx context.Context, cs kubernetes.Interface, ns string) error {
	_, err := cs.CoreV1().Namespaces().Create(ctx, &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: ns}}, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		return fmt.Errorf("creating namespace %s: %w", ns, err)
	}
	return nil
}

// ensureHeadlessService mirrors ecp ensureHeadlessServiceInNamespace.
func ensureHeadlessService(ctx context.Context, cs kubernetes.Interface, ns, name string) error {
	svc := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: name, Namespace: ns,
			Labels: map[string]string{"app": "sglang-runner-tunnel", "exos/service": name},
		},
		Spec: corev1.ServiceSpec{
			ClusterIP:      corev1.ClusterIPNone,
			IPFamilyPolicy: ptr.To(corev1.IPFamilyPolicySingleStack),
			IPFamilies:     []corev1.IPFamily{corev1.IPv6Protocol},
			Ports:          []corev1.ServicePort{{Name: "http", Port: ServicePort, Protocol: corev1.ProtocolTCP}},
		},
	}
	_, err := cs.CoreV1().Services(ns).Create(ctx, svc, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		return fmt.Errorf("creating headless service %s/%s: %w", ns, name, err)
	}
	return nil
}

// applySlice mirrors ecp applyEndpointSliceInNamespace for a Borneo runner.
func applySlice(ctx context.Context, cs kubernetes.Interface, ns, svc, sliceName string, r Runner, hostname, addr string, extraLabels map[string]string) error {
	labels := map[string]string{
		discoveryv1.LabelServiceName: svc,
		"exos/runner-id":             r.Name,
		"exos/mode":                  "borneo",
	}
	for k, v := range extraLabels {
		labels[k] = v
	}
	s := &discoveryv1.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name: sliceName, Namespace: ns, Labels: labels,
			Annotations: map[string]string{
				"exos/bns":      strings.TrimPrefix(r.providerID(), "provider:/"),
				"exos/borg-dns": fmt.Sprintf("%d.%s.%s.%s.borg.google.com", r.Task, r.Job, r.User, r.Cell),
				"exos/k8s-fqdn": r.K8sFQDN(),
			},
		},
		AddressType: discoveryv1.AddressTypeIPv6,
		Ports:       []discoveryv1.EndpointPort{{Name: ptr.To("http"), Port: ptr.To[int32](ServicePort), Protocol: ptr.To(corev1.ProtocolTCP)}},
		Endpoints: []discoveryv1.Endpoint{{
			Addresses: []string{addr},
			Hostname:  ptr.To(hostname),
			// ECP hard-codes these to true today.
			Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(r.Ready), Serving: ptr.To(r.Ready)},
		}},
	}
	existing, err := cs.DiscoveryV1().EndpointSlices(ns).Get(ctx, sliceName, metav1.GetOptions{})
	switch {
	case apierrors.IsNotFound(err):
		_, err = cs.DiscoveryV1().EndpointSlices(ns).Create(ctx, s, metav1.CreateOptions{})
	case err == nil:
		s.ResourceVersion = existing.ResourceVersion
		_, err = cs.DiscoveryV1().EndpointSlices(ns).Update(ctx, s, metav1.UpdateOptions{})
	}
	if err != nil {
		return fmt.Errorf("applying slice %s/%s: %w", ns, sliceName, err)
	}
	return nil
}
