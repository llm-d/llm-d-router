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

package utils

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/onsi/ginkgo/v2"
)

// processPortOffset is the gap between consecutive parallel processes' ports,
// shared by ProcessPort (NodePorts) and BuildExtraPortMappings (Kind's
// containerPort/hostPort mappings) so a process's actual exposed port always
// matches the port it computes and dials.
const processPortOffset = 100

// ProcessPort returns the NodePort for the current Ginkgo parallel process for
// a service whose first process uses basePort. When tests run in parallel,
// each process gets its own NodePort of the form basePort + processPortOffset
// * (the process number minus one), so process one gets exactly basePort.
func ProcessPort(basePort int) int {
	return basePort + processPortOffset*(ginkgo.GinkgoParallelProcess()-1)
}

// NamespaceForProcess returns the namespace assigned to the given Ginkgo
// parallel process number. If numProcesses is 1, the namespace is simply
// baseName; otherwise it is of the form baseName-N.
func NamespaceForProcess(baseName string, numProcesses, processNum int) string {
	if numProcesses == 1 {
		return baseName
	}
	return fmt.Sprintf("%s-%d", baseName, processNum)
}

// Namespace returns the namespace assigned to the current Ginkgo parallel
// process. See NamespaceForProcess.
func Namespace(baseName string, numProcesses int) string {
	return NamespaceForProcess(baseName, numProcesses, ginkgo.GinkgoParallelProcess())
}

// DefaultNsName returns the default base namespace: "default" when not
// running in parallel, or parallelName when numProcesses > 1.
func DefaultNsName(numProcesses int, parallelName string) string {
	if numProcesses == 1 {
		return "default"
	}
	return parallelName
}

// LocalhostURL returns an http://localhost:<port> base URL for the given port.
func LocalhostURL(port int) string {
	return "http://localhost:" + strconv.Itoa(port)
}

// RequireParallelProcessesMatch fails the suite unless numProcesses (from the
// E2E_NUM_PROCS environment variable) matches the actual number of Ginkgo
// parallel processes the suite is running under.
func RequireParallelProcessesMatch(numProcesses int) {
	suiteConfig, _ := ginkgo.GinkgoConfiguration()
	if numProcesses != suiteConfig.ParallelTotal {
		ginkgo.Fail(fmt.Sprintf("The value of the environment variable `E2E_NUM_PROCS` (%d) is not equal to the number of ginkgo processes being run (%d)",
			numProcesses, suiteConfig.ParallelTotal))
	}
}

// BuildExtraPortMappings builds a Kind `extraPortMappings` YAML fragment for
// numProcesses processes, mapping each (containerPortBase, hostPortBase) pair
// in portPairs to per-process ports offset by processPortOffset * (process
// index). Each item uses 2-space indentation to match the
// `extraPortMappings:` field level in a Kind cluster config; callers
// substituting this into their own YAML must keep that indentation in sync.
func BuildExtraPortMappings(numProcesses int, portPairs ...[2]int) string {
	var b strings.Builder
	for idx := range numProcesses {
		inc := idx * processPortOffset
		for _, pair := range portPairs {
			fmt.Fprintf(&b, "\n  - containerPort: %d", pair[0]+inc)
			fmt.Fprintf(&b, "\n    hostPort: %d", pair[1]+inc)
			fmt.Fprintf(&b, "\n    protocol: TCP")
		}
	}
	return b.String()
}
