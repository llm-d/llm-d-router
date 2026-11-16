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

package scripts

import (
	"os"
	"os/exec"
	"reflect"
	"regexp"
	"strings"
	"testing"

	"sigs.k8s.io/yaml"
)

// renderPipeline returns the variables kind-dev-env.sh passes to envsubst for the scenario manifests, and the awk
// programs it pipes them through afterwards, in order.
func renderPipeline(t *testing.T) ([]string, []string) {
	t.Helper()
	src, err := os.ReadFile("kind-dev-env.sh")
	if err != nil {
		t.Fatal(err)
	}
	_, block, ok := strings.Cut(string(src), "# Deploy scenario-specific vLLM components")
	if !ok {
		t.Fatal("scenario render pipeline not found in kind-dev-env.sh")
	}
	block, _, ok = strings.Cut(block, "| kubectl --context ${KUBE_CONTEXT} apply -f -")
	if !ok {
		t.Fatal("the scenario render pipeline does not end in kubectl apply")
	}
	_, rest, ok := strings.Cut(block, "envsubst '")
	if !ok {
		t.Fatal("no envsubst in the scenario render pipeline")
	}
	list, _, ok := strings.Cut(rest, "'")
	if !ok {
		t.Fatal("the envsubst variable list is not closed")
	}
	vars := regexp.MustCompile(`\$\{([A-Z0-9_]+)\}`).FindAllStringSubmatch(list, -1)
	names := make([]string, 0, len(vars))
	for _, v := range vars {
		names = append(names, v[1])
	}
	parts := strings.Split(block, "| awk '\n")[1:]
	stages := make([]string, 0, len(parts))
	for _, part := range parts {
		prog, _, ok := strings.Cut(part, "\n  ' \\")
		if !ok {
			t.Fatal("an awk stage in the scenario render pipeline is not closed")
		}
		stages = append(stages, prog)
	}
	if len(stages) == 0 {
		t.Fatal("no awk stage found in the scenario render pipeline")
	}
	return names, stages
}

// renderArgs renders a decode Deployment, as kubectl kustomize emits it, through the script's pipeline and returns
// its container args, with a null item as "<null>".
func renderArgs(t *testing.T, envsubst, stages []string, value string) []string {
	t.Helper()
	const renderedDeployment = `apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: vllm
        args:
        - --port=8000
        - ${VLLM_EXTRA_ARGS_D}
`
	env := map[string]string{"VLLM_EXTRA_ARGS_D": value}
	out := renderedDeployment
	for _, name := range envsubst {
		out = strings.ReplaceAll(out, "${"+name+"}", env[name])
	}
	for _, prog := range stages {
		cmd := exec.Command("awk", prog)
		cmd.Env = append(os.Environ(), "VLLM_EXTRA_ARGS_D="+value)
		cmd.Stdin = strings.NewReader(out)
		b, err := cmd.Output()
		if err != nil {
			t.Fatalf("awk: %v", err)
		}
		out = string(b)
	}
	var d struct {
		Spec struct {
			Template struct {
				Spec struct {
					Containers []struct {
						Args []any `json:"args"`
					} `json:"containers"`
				} `json:"spec"`
			} `json:"template"`
		} `json:"spec"`
	}
	if err := yaml.Unmarshal([]byte(out), &d); err != nil {
		t.Fatalf("rendered manifest is not valid YAML: %v\n%s", err, out)
	}
	args := make([]string, 0, len(d.Spec.Template.Spec.Containers[0].Args))
	for _, a := range d.Spec.Template.Spec.Containers[0].Args {
		if a == nil {
			args = append(args, "<null>")
			continue
		}
		args = append(args, a.(string))
	}
	return args
}

func TestKindDevEnvExtraArgs(t *testing.T) {
	if _, err := exec.LookPath("awk"); err != nil {
		t.Skip("awk is not installed")
	}
	envsubst, stages := renderPipeline(t)
	for _, tc := range []struct {
		name  string
		value string
		want  []string
	}{
		{name: "empty keeps a null item", value: "", want: []string{"--port=8000", "<null>"}},
		{name: "one flag", value: "--kv-cache-size=256", want: []string{"--port=8000", "--kv-cache-size=256"}},
		{name: "two flags", value: "--max-model-len=131072 --kv-cache-size=8192",
			want: []string{"--port=8000", "--max-model-len=131072", "--kv-cache-size=8192"}},
		{name: "tab separated", value: "--strict\t--max-model-len=131072",
			want: []string{"--port=8000", "--strict", "--max-model-len=131072"}},
		{name: "a JSON value", value: `--mm-processor-kwargs={"num_crops":4}`,
			want: []string{"--port=8000", `--mm-processor-kwargs={"num_crops":4}`}},
		{name: "a JSON value with another flag stays one argument", value: `--mm-processor-kwargs={"num_crops":4} --max-model-len=131072`,
			want: []string{"--port=8000", `--mm-processor-kwargs={"num_crops":4} --max-model-len=131072`}},
		{name: "a quoted value with another flag stays one argument", value: "--chat-template='a b' --max-model-len=4096",
			want: []string{"--port=8000", "--chat-template='a b' --max-model-len=4096"}},
		{name: "a backslash with another flag stays one argument", value: `--model=foo\path --strict`,
			want: []string{"--port=8000", `--model=foo\path --strict`}},
		{name: "multi-line list form", value: "--max-model-len=131072\n        - --strict",
			want: []string{"--port=8000", "--max-model-len=131072", "--strict"}},
		{name: "multi-line continuation folds into one argument", value: "--model=foo\n          --literal",
			want: []string{"--port=8000", "--model=foo --literal"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := renderArgs(t, envsubst, stages, tc.value); !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("args %q, want %q", got, tc.want)
			}
		})
	}
}
