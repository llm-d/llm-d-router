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

// Package poc is the step-1 PoC: can a standalone EPP with a custom
// EndpointSlice discovery plugin discover Exos Borneo runners?
//
// It starts a real kube-apiserver+etcd (envtest), reproduces the Exos cluster
// state (control node + linked-runner annotations + hostNetwork model pods +
// ECP-style headless Services/EndpointSlices, see simecp), runs the real
// EPP binary (cmd/epp) in standalone/custom-discovery mode against it, and talks
// ext_proc to it like Envoy would.
//
// Run: KUBEBUILDER_ASSETS=$(setup-envtest use 1.36.2 -p path) go test -v ./exos/poc/
package poc

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	envoyCorev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"sigs.k8s.io/controller-runtime/pkg/envtest"

	"github.com/llm-d/llm-d-router/exos/poc/simecp"
	testnet "github.com/llm-d/llm-d-router/test/framework/net"
)

var topo = simecp.Topology{
	ControlNode:   "gke-exos-dev-1-control-pool-0",
	ControlNodeIP: "10.128.0.5",
	ModelNS:       "prediction",
	Runners: []simecp.Runner{
		{Name: "borneo-p0", IPv6: "2606:ec80:10:263d::10", Role: "prefill", User: "uniserve-dev-jobs", Cell: "pnatlix", Job: "exos_glm_p", Task: 0, Ready: true},
		{Name: "borneo-d0", IPv6: "2606:ec80:10:263d::20", Role: "decode", User: "uniserve-dev-jobs", Cell: "pnatlix", Job: "exos_glm_d", Task: 0, Ready: true},
		{Name: "borneo-d1", IPv6: "2606:ec80:10:263d::21", Role: "decode", User: "uniserve-dev-jobs", Cell: "pnatlix", Job: "exos_glm_d", Task: 1, Ready: true},
	},
}

const eppConfig = `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- name: exos-disc
  type: endpointslice-discovery
  parameters:
    namespace: default
    selector: kubernetes.io/service-name=glm52-server
    portName: http
    kubeconfig: %s
- type: decode-filter
- type: random-picker
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: decode-filter
  - pluginRef: random-picker
dataLayer:
  discovery:
    endpoints:
      pluginRef: exos-disc
`

func TestEPPDiscoversExosEndpointSlices(t *testing.T) {
	if os.Getenv("KUBEBUILDER_ASSETS") == "" {
		t.Skip("KUBEBUILDER_ASSETS not set; run setup-envtest first")
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	work := t.TempDir()

	// --- 1. Real API server. Dual-stack service range so the IPv6 SingleStack
	// headless Services ECP creates are accepted, as on the BYOIPv6 GKE cluster.
	env := &envtest.Environment{AttachControlPlaneOutput: os.Getenv("ENVTEST_DEBUG") != ""}
	apiArgs := env.ControlPlane.GetAPIServer().Configure()
	apiArgs.Set("service-cluster-ip-range", "10.96.0.0/16,fd00:10:96::/112")
	apiArgs.Set("advertise-address", "127.0.0.1") // hermetic: no default route needed
	restCfg, err := env.Start()
	if err != nil {
		t.Fatalf("starting envtest: %v", err)
	}
	defer func() { _ = env.Stop() }()
	cs := kubernetes.NewForConfigOrDie(restCfg)

	// --- 2. Reproduce the Exos cluster: control node, linked runners, hostNetwork pods, ECP slices.
	if err := simecp.SetupCluster(ctx, cs, topo); err != nil {
		t.Fatal(err)
	}
	if err := simecp.Reconcile(ctx, cs, topo, true); err != nil {
		t.Fatal(err)
	}

	// What pod-based discovery (the EPP default) would see: every model pod has the control node IP.
	pods, err := cs.CoreV1().Pods(topo.ModelNS).List(ctx, metav1.ListOptions{LabelSelector: "app=glm52"})
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range pods.Items {
		t.Logf("pod-based view: %s podIP=%s hostNetwork=%v", p.Name, p.Status.PodIP, p.Spec.HostNetwork)
		if p.Status.PodIP != topo.ControlNodeIP {
			t.Errorf("pod %s podIP=%s, want control node IP %s", p.Name, p.Status.PodIP, topo.ControlNodeIP)
		}
	}

	// --- 3. Run the real EPP binary in custom-discovery (standalone) mode.
	kubeconfig := writeKubeconfig(t, env, restCfg, work)
	cfgPath := filepath.Join(work, "epp.yaml")
	if err := os.WriteFile(cfgPath, []byte(fmt.Sprintf(eppConfig, kubeconfig)), 0o600); err != nil {
		t.Fatal(err)
	}
	bin := buildEPP(t, work)
	grpcPort, healthPort, metricsPort := freePort(t), freePort(t), freePort(t)
	logFile, _ := os.Create(filepath.Join(work, "epp.log"))
	defer logFile.Close()
	cmd := exec.CommandContext(ctx, bin,
		"--pool-name", "glm52", "--pool-namespace", "default",
		"--config-file", cfgPath,
		"--grpc-port", fmt.Sprint(grpcPort), "--grpc-health-port", fmt.Sprint(healthPort),
		"--metrics-port", fmt.Sprint(metricsPort),
		"--secure-serving=false", "--allow-experimental-plugins", "--v=2")
	cmd.Stdout, cmd.Stderr = logFile, logFile
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		if t.Failed() {
			b, _ := os.ReadFile(logFile.Name())
			t.Logf("---- epp log ----\n%s", tail(string(b), 120))
		}
	}()

	// --- 4. EPP's own view of its datastore (plugin StateDumper reports endpoint IDs only;
	// addresses are checked through ext_proc below).
	state := waitForDiscovery(t, metricsPort, len(topo.Runners))
	t.Logf("EPP discovery state: %s", state)
	for _, r := range topo.Runners {
		if id := "default/glm52-server." + r.Name; !strings.Contains(state, id) {
			t.Errorf("runner %s (endpoint %s) not in EPP datastore", r.Name, id)
		}
	}

	// --- 5. Envoy-style ext_proc: EPP must route to a decode runner's Borneo IPv6.
	conn, err := grpc.NewClient(fmt.Sprintf("localhost:%d", grpcPort), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	decodeIPs := map[string]bool{}
	for i := 0; i < 20; i++ {
		dest := pickViaExtProc(t, ctx, conn)
		decodeIPs[dest] = true
	}
	t.Logf("ext_proc destinations over 20 requests: %v", decodeIPs)
	for dest := range decodeIPs {
		if dest != "[2606:ec80:10:263d::20]:9080" && dest != "[2606:ec80:10:263d::21]:9080" {
			t.Errorf("routed to %q, want one of the decode runners", dest)
		}
	}

	// --- 6. Readiness: a not-ready slice (readiness-aware ECP) leaves the pool.
	if err := simecp.SetRunnerReady(ctx, cs, topo.Runners[2], false); err != nil {
		t.Fatal(err)
	}
	waitForDiscovery(t, metricsPort, len(topo.Runners)-1)
	for i := 0; i < 10; i++ {
		if dest := pickViaExtProc(t, ctx, conn); dest != "[2606:ec80:10:263d::20]:9080" {
			t.Errorf("after borneo-d1 not ready, routed to %q", dest)
		}
	}
}

func pickViaExtProc(t *testing.T, ctx context.Context, conn *grpc.ClientConn) string {
	t.Helper()
	rctx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	stream, err := extProcPb.NewExternalProcessorClient(conn).Process(rctx)
	if err != nil {
		t.Fatalf("opening ext_proc stream: %v", err)
	}
	hdr := func(k, v string) *envoyCorev3.HeaderValue {
		return &envoyCorev3.HeaderValue{Key: k, RawValue: []byte(v)}
	}
	body, _ := json.Marshal(map[string]any{
		"model": "glm-5.2-nvfp4", "max_tokens": 16,
		"messages": []map[string]string{{"role": "user", "content": "hello from Borneo"}},
	})
	msgs := []*extProcPb.ProcessingRequest{
		{Request: &extProcPb.ProcessingRequest_RequestHeaders{RequestHeaders: &extProcPb.HttpHeaders{
			Headers: &envoyCorev3.HeaderMap{Headers: []*envoyCorev3.HeaderValue{
				hdr(":method", "POST"), hdr(":path", "/v1/chat/completions"), hdr(":authority", "inference-envoy"),
				hdr("content-type", "application/json"), hdr("x-request-id", fmt.Sprint(time.Now().UnixNano())),
			}}}}},
		{Request: &extProcPb.ProcessingRequest_RequestBody{RequestBody: &extProcPb.HttpBody{Body: body, EndOfStream: true}}},
	}
	for _, m := range msgs {
		if err := stream.Send(m); err != nil {
			t.Fatalf("ext_proc send: %v", err)
		}
	}
	for {
		resp, err := stream.Recv()
		if err != nil {
			t.Fatalf("ext_proc recv (no destination yet): %v", err)
		}
		if ir := resp.GetImmediateResponse(); ir != nil {
			t.Fatalf("EPP returned immediate response %v: %s", ir.GetStatus(), ir.GetBody())
		}
		var common interface {
			GetResponse() *extProcPb.CommonResponse
		}
		switch {
		case resp.GetRequestHeaders() != nil:
			common = resp.GetRequestHeaders()
		case resp.GetRequestBody() != nil:
			common = resp.GetRequestBody()
		default:
			continue
		}
		for _, h := range common.GetResponse().GetHeaderMutation().GetSetHeaders() {
			if h.GetHeader().GetKey() == "x-gateway-destination-endpoint" {
				v := string(h.GetHeader().GetRawValue())
				if v == "" {
					v = h.GetHeader().GetValue()
				}
				_ = stream.CloseSend()
				return v
			}
		}
	}
}

func waitForDiscovery(t *testing.T, metricsPort, want int) string {
	t.Helper()
	url := fmt.Sprintf("http://localhost:%d/debug/plugins/state", metricsPort)
	var last string
	deadline := time.Now().Add(60 * time.Second)
	for time.Now().Before(deadline) {
		if resp, err := http.Get(url); err == nil {
			b, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			last = string(b)
			if strings.Contains(last, fmt.Sprintf(`"totalEndpoints":%d`, want)) {
				return last
			}
		}
		time.Sleep(250 * time.Millisecond)
	}
	t.Fatalf("EPP never reported %d endpoints; last state: %s", want, last)
	return ""
}

func writeKubeconfig(t *testing.T, env *envtest.Environment, _ any, dir string) string {
	t.Helper()
	u, err := env.AddUser(envtest.User{Name: "exos-epp", Groups: []string{"system:masters"}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	kc, err := u.KubeConfig()
	if err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(dir, "kubeconfig")
	if err := os.WriteFile(p, kc, 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

func buildEPP(t *testing.T, dir string) string {
	t.Helper()
	out := filepath.Join(dir, "epp")
	root, err := exec.Command("go", "env", "GOMOD").Output()
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command("go", "build", "-o", out, "./cmd/epp")
	cmd.Dir = filepath.Dir(strings.TrimSpace(string(root)))
	if b, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("building epp: %v\n%s", err, b)
	}
	return out
}

func freePort(t *testing.T) int {
	t.Helper()
	p, err := testnet.GetFreePort()
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func tail(s string, n int) string {
	lines := strings.Split(s, "\n")
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	return strings.Join(lines, "\n")
}
