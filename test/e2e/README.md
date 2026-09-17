# End-to-End Tests

This document provides instructions on how to run the end-to-end tests.

## Overview

The end-to-end tests validate router functionality against a Kubernetes cluster using Ginkgo.
Each case renders the repository's `config/charts/llm-d-router-standalone` chart with its local
`routerlib` dependency. The default standalone topology runs Envoy and EPP in the same Pod.
The chart supplies the InferencePool, plugin configuration, Services, and RBAC. Model simulators,
disaggregation sidecars, and the shared renderer use the development Kustomize manifests.

`utils/standalone/standalone-values.yaml` supplies the test resource requests, experimental plugin flags, unauthenticated
metrics, and KV event port. Each case supplies its plugin configuration, EPP image, replica count,
and model target ports. The suite creates all rendered objects before waiting for readiness.
With three EPP replicas, exactly one router Pod must be Ready and two must remain standbys.
These tests exercise rendered chart workloads; they do not exercise Helm release upgrades or rollbacks.

## Prerequisites

- Docker or Podman to run the builder container, which includes Go, kubectl, Kind, and Helm.
- [Make](https://www.gnu.org/software/make/manual/make.html) installed to run the end-to-end test target.
- (Optional) When using the GPU-based vLLM deployment, a Hugging Face Hub token with access to the
  [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) model is required.
  After obtaining the token and being granted access to the model, set the `HF_TOKEN` environment variable:

   ```sh
   export HF_TOKEN=<MY_HF_TOKEN>
   ```

## Running the End-to-End Tests in Parallel

By default the end to end tests run in groups that run in parallel to each other on the same Kubernetes cluster.
Each process uses an independent Namespace and a pair of NodePorts for HTTP and metrics. A test-only
`router-access` Service selects the chart's router Pods. The chart's EPP Service also accepts simulator
KV events on port 5557. Each case registers cleanup before creating chart and model resources and waits
for their Pods to terminate before reusing resource names. `E2E_KEEP_CLUSTER_ON_FAILURE` preserves failed
cases for inspection.

## Running the End-to-End Tests

Follow these steps to run the end-to-end tests:

1. **Use this repository**: Ensure you are working from a checkout of this repository, then change to the repository root before running the tests:

   ```sh
   cd <path-to-this-repository>
   ```

1. **Optional Settings**

   - **Running all of the tests serially** By default the end to end tests are run in groups that are parallel to
     each other. The number of groups running in parallel at any time is controlled via the E2E_NUM_PROCS environment
     variable, which defaults to five. To run all of the tests in a serial fashion, use `E2E_NUM_PROCS=1`.

   - **Run the tests on a real cluster**: By default the end to end tests are run on a kind cluster that is created
     and torn down by the test code. If you want to run the tests on a real Kubernetes cluster, set the following
     environment variable:

     ```sh
     K8S_CONTEXT=<kubernetes context>
     ```

     Where `kubernetes context` is the context of the cluster in question in your Kubernetes config file.

     **Note:** Existing-context runs use a supervised `kubectl port-forward` process for both HTTP and
     metrics. It selects a Ready, non-terminating router Pod and reconnects after Pod deletion, leader
     failover, or process exit. Cleanup stops the forwarder and waits for it to exit.

   - **Set the test namespace**: The namespace(s) in which the tests run vary based on whether or not the tests
     are being run in parallel or not. 

     If the tests are being run in parallel, the e2e test creates resources in namespaces of the form <base>-N,
     where <base> by default is `e2e` and N is the process number of the process running the test. <base> can
     changed by setting the following environment variable:

     ```sh
     export NAMESPACE=<MY_NS>
     ```

     If the test are not being run in parallel, then by default, the e2e test creates resources in the `default`
     namespace. If you would like to change this namespace, set the following environment variable:

     ```sh
     export NAMESPACE=<MY_NS>
     ```

   - **Set the model server image**: By default, the e2e test uses the [vLLM Simulator](https://github.com/llm-d/llm-d-inference-sim)
     to simulate a backend model server. If you would like to change the model server to a real vLLM image, set the following 
     environment variable to the vLLM image of your choice:

     ```sh
     export VLLM_IMAGE=<vLLM image of your choice>
     ```

   - **Set the router image**: `EPP_IMAGE` defaults to
     `ghcr.io/llm-d/llm-d-router-endpoint-picker:dev`. Registry names with ports are supported.
     The chart's image fields require a tag, so digest references are rejected. An omitted tag uses `latest`.

   - **Keep the cluster available after a failure**: Normally the cluster is deleted after the end to end tests run. To keep the cluster
     available after the tests have failed, useful for debugging the state of the cluster after the test has run, set the environment
     variable `E2E_KEEP_CLUSTER_ON_FAILURE` to `true`.

1. **Run the Tests**: Run the `test-e2e` target:

   ```sh
   make test-e2e
   ```

   The test suite prints details for each step. Note that the `vllm-qwen3-32b` model server deployment
   may take several minutes to report an `Available=True` status due to the time required for bootstrapping.

The runner builds the chart dependency once before starting parallel workers. For focused helper tests
inside the builder, run:

```sh
helm dependency build --skip-refresh config/charts/llm-d-router-standalone
go test ./test/e2e/utils/... -count=1
```
