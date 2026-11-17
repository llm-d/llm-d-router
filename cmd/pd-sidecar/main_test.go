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

package main

import (
	"net"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

// mainArgsEnv carries newline-separated sidecar arguments to the re-executed
// copy of the test binary, which runs main with them. main exits the process,
// so only a child process can report its exit code.
const mainArgsEnv = "LLMD_PD_SIDECAR_MAIN_ARGS"

func TestMainExitsNonZeroOnStartupFailure(t *testing.T) {
	if args, ok := os.LookupEnv(mainArgsEnv); ok {
		os.Args = append([]string{"pd-sidecar"}, strings.Split(args, "\n")...)
		main()
		return
	}

	busy, err := net.Listen("tcp", ":0")
	require.NoError(t, err)
	defer busy.Close()
	busyPort := strconv.Itoa(busy.Addr().(*net.TCPAddr).Port)

	tests := []struct {
		name    string
		args    []string
		wantLog string
	}{
		{name: "options fail to complete", args: []string{"--tls-min-version=bogus"}, wantLog: "Failed to complete configuration"},
		{name: "options fail validation", args: []string{"--kv-connector=bogus"}, wantLog: "Invalid configuration"},
		{name: "proxy fails to start", args: []string{"--port=" + busyPort}, wantLog: "failed to start proxy server"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := exec.Command(os.Args[0], "-test.run=^TestMainExitsNonZeroOnStartupFailure$")
			cmd.Env = append(os.Environ(), mainArgsEnv+"="+strings.Join(tt.args, "\n"))
			out, err := cmd.CombinedOutput()

			var exitErr *exec.ExitError
			require.ErrorAs(t, err, &exitErr, "output:\n%s", out)
			require.Equal(t, 1, exitErr.ExitCode(), "output:\n%s", out)
			require.Contains(t, string(out), tt.wantLog)
		})
	}
}
