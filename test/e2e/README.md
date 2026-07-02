# End-to-End Tests

This document provides instructions on how to run the end-to-end tests.

## Overview

The end-to-end tests are designed to validate end-to-end Gateway API Inference Extension functionality. These tests are executed against a Kubernetes cluster and use the Ginkgo testing framework to ensure the extension behaves as expected.

## Prerequisites

- [Go](https://golang.org/doc/install) installed on your machine.
- [Make](https://www.gnu.org/software/make/manual/make.html) installed to run the end-to-end test target.
- (Optional) When using the GPU-based vLLM deployment, a Hugging Face Hub token with access to the
  [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) model is required.
  After obtaining the token and being granted access to the model, set the `HF_TOKEN` environment variable:

   ```sh
   export HF_TOKEN=<MY_HF_TOKEN>
   ```

## Running the End-to-End Tests

Follow these steps to run the end-to-end tests:

1. **Use this repository**: Ensure you are working from a checkout of this repository, then change to the repository root before running the tests:

   ```sh
   cd <path-to-this-repository>
   ```

1. **Optional Settings**

   - **Run the tests on a real cluster**: By default the end to end tests are run on a kind cluster that is created
     and torn down by the test code. If you want to run the tests on a real Kubernetes cluster, set the following
     environment variable:

     ```sh
     K8S_CONTEXT=<kubernetes context>
     ```

     Where `kubernetes context` is the context of the cluster in question in your Kubernetes config file.

     **Note:** When running on a real cluster the tests will start a pair of `kubectl port-forward` processes
     to sent various requests to the cluster under test.

   - **Set the test namespace**: By default, the e2e test creates resources in the `default` namespace.
     If you would like to change this namespace, set the following environment variable:

     ```sh
     export NAMESPACE=<MY_NS>
     ```

   - **Set the model server image**: By default, the e2e test uses the [vLLM Simulator](https://github.com/llm-d/llm-d-inference-sim)
     to simulate a backend model server. If you would like to change the model server to a real vLLM image, set the following 
     environment variable to the vLLM image of your choice:

     ```sh
     export VLLM_IMAGE=<vLLM image of your choice>
     ```

   - **Keep the cluster available after a failure**: Normally the cluster is deleted after the end to end tests run. To keep the cluster
     available after the tests have failed, useful for debugging the state of the cluster after the test has run, set the `environment
     variable `E2E_KEEP_CLUSTER_ON_FAILURE` to `true`.

1. **Run the Tests**: Run the `test-e2e` target:

   ```sh
   make test-e2e
   ```

   The test suite prints details for each step. Note that the `vllm-qwen3-32b` model server deployment
   may take several minutes to report an `Available=True` status due to the time required for bootstrapping.
