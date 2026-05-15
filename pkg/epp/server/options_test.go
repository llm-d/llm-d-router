/*
Copyright 2025 The Kubernetes Authors.

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

package server

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/spf13/pflag"
)

// TestEndpointTargetPorts
func TestEndpointTargetPorts(t *testing.T) {
	tests := []struct {
		name          string
		fs            *pflag.FlagSet
		args          []string
		expectError   bool // expect validation error
		expectedPorts []int
	}{
		{
			name: "Valid multiple flags order check",
			args: []string{
				"--endpoint-target-ports", "8080",
				"--endpoint-target-ports", "9090",
				"--endpoint-target-ports", "80",
			},
			expectError:   false,
			expectedPorts: []int{8080, 9090, 80},
		},
		{
			name: "Valid comma separated list",
			args: []string{
				"--endpoint-target-ports", "8080,9090,80",
			},
			expectError:   false,
			expectedPorts: []int{8080, 9090, 80},
		},
		{
			name: "Handle duplicates order preservation",
			args: []string{
				"--endpoint-target-ports", "8080",
				"--endpoint-target-ports", "9090",
				"--endpoint-target-ports", "8080",
				"--endpoint-target-ports", "9090",
			},
			expectError:   false,
			expectedPorts: []int{8080, 9090},
		},
		{
			name: "Invalid negative port number",
			args: []string{
				"--endpoint-target-ports", "8080",
				"--endpoint-target-ports", "-1",
			},
			expectError:   true,
			expectedPorts: []int{8080, -1},
		},
		{
			name: "Invalid over max port range",
			args: []string{
				"--endpoint-target-ports", "65536",
			},
			expectError:   true,
			expectedPorts: []int{65536},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.fs = pflag.NewFlagSet(tt.name, pflag.ContinueOnError)

			opts := NewOptions()
			opts.AddFlags(tt.fs)

			argv := make([]string, 0, 4+len(tt.args))
			argv = append(argv, "--endpoint-selector", "app=vllm", "--config-file", "fake-config.yaml") // avoid an options validation error
			argv = append(argv, tt.args...)

			if err := tt.fs.Parse(argv); err != nil {
				t.Fatalf("Failed to parse flags: %v", err)
			}

			if err := opts.Complete(); err != nil {
				if !tt.expectError {
					t.Fatalf("Complete failed unexpectedly with error: %v", err)
				}
				return
			}

			err := opts.Validate()
			if tt.expectError {
				if err == nil {
					t.Fatalf("Expected a validation error but got none.")
				}
				return
			}

			if err != nil {
				t.Fatalf("Validate failed unexpectedly with error: %v", err)
			}

			if diff := cmp.Diff(tt.expectedPorts, opts.EndpointTargetPorts); diff != "" {
				t.Errorf("Resulting ports mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestGRPCFlags(t *testing.T) {
	tests := []struct {
		name                string
		args                []string
		expectedMaxRecvSize int
		expectedMaxSendSize int
		expectError         bool
	}{
		{
			name: "Valid flags (raw integers)",
			args: []string{
				"--grpc-max-recv-msg-size", "10485760",
				"--grpc-max-send-msg-size", "20971520",
			},
			expectedMaxRecvSize: 10485760,
			expectedMaxSendSize: 20971520,
		},
		{
			name: "Valid flags with units",
			args: []string{
				"--grpc-max-recv-msg-size", "10Mi",
				"--grpc-max-send-msg-size", "20M",
			},
			expectedMaxRecvSize: 10485760, // 10 * 1024 * 1024
			expectedMaxSendSize: 20000000, // 20 * 1000 * 1000
		},
		{
			name: "Valid flags with B suffix",
			args: []string{
				"--grpc-max-recv-msg-size", "10MiB",
				"--grpc-max-send-msg-size", "20MB",
			},
			expectedMaxRecvSize: 10485760,
			expectedMaxSendSize: 20000000,
		},
		{
			name:                "Defaults",
			args:                []string{},
			expectedMaxRecvSize: 0,
			expectedMaxSendSize: 0,
		},
		{
			name: "Invalid recv size unit",
			args: []string{
				"--grpc-max-recv-msg-size", "10invalid",
			},
			expectError: true,
		},
		{
			name: "Invalid send size unit",
			args: []string{
				"--grpc-max-send-msg-size", "abc",
			},
			expectError: true,
		},
		{
			name: "Negative recv size",
			args: []string{
				"--grpc-max-recv-msg-size", "-10Mi",
			},
			expectError: true,
		},
		{
			name: "Overflow recv size",
			args: []string{
				"--grpc-max-recv-msg-size", "10Ei",
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fs := pflag.NewFlagSet(tt.name, pflag.ContinueOnError)
			opts := NewOptions()
			opts.AddFlags(fs)

			argv := make([]string, 0, 4+len(tt.args))
			argv = append(argv, "--pool-name", "test-pool", "--config-file", "fake-config.yaml")
			argv = append(argv, tt.args...)

			if err := fs.Parse(argv); err != nil {
				t.Fatalf("Failed to parse flags: %v", err)
			}

			err := opts.Complete()
			if err == nil {
				err = opts.Validate()
			}

			if tt.expectError {
				if err == nil {
					t.Fatalf("Expected Complete() or Validate() to fail, but it succeeded")
				}
				return
			}
			if err != nil {
				t.Fatalf("Complete/Validate failed unexpectedly with error: %v", err)
			}

			if opts.GRPCMaxRecvMsgSize != tt.expectedMaxRecvSize {
				t.Errorf("GRPCMaxRecvMsgSize mismatch: got %v, want %v", opts.GRPCMaxRecvMsgSize, tt.expectedMaxRecvSize)
			}
			if opts.GRPCMaxSendMsgSize != tt.expectedMaxSendSize {
				t.Errorf("GRPCMaxSendMsgSize mismatch: got %v, want %v", opts.GRPCMaxSendMsgSize, tt.expectedMaxSendSize)
			}
		})
	}
}

func TestValidateDirectValues(t *testing.T) {
	opts := NewOptions()
	opts.PoolName = "test-pool" // bypass other validations
	opts.GRPCMaxRecvMsgSize = -5
	if err := opts.Validate(); err == nil {
		t.Errorf("Expected Validate() to fail for negative GRPCMaxRecvMsgSize, but it succeeded")
	}

	opts = NewOptions()
	opts.PoolName = "test-pool"
	opts.GRPCMaxSendMsgSize = -5
	if err := opts.Validate(); err == nil {
		t.Errorf("Expected Validate() to fail for negative GRPCMaxSendMsgSize, but it succeeded")
	}
}

func TestEnablePluginStateDebugDefaultAndFlag(t *testing.T) {
	opts := NewOptions()
	if opts.EnablePluginStateDebug {
		t.Fatal("EnablePluginStateDebug should default to false")
	}

	fs := pflag.NewFlagSet("plugin-state-debug", pflag.ContinueOnError)
	opts.AddFlags(fs)
	if err := fs.Parse([]string{"--enable-plugin-state-debug"}); err != nil {
		t.Fatalf("Failed to parse flags: %v", err)
	}

	if !opts.EnablePluginStateDebug {
		t.Fatal("EnablePluginStateDebug should be enabled when --enable-plugin-state-debug is set")
	}
}
