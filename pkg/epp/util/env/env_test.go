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

package env

import (
	"testing"
	"time"

	"github.com/go-logr/logr/testr"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
)

func TestGetEnvFloat(t *testing.T) {
	logger := logutil.NewTestLogger()

	tests := []struct {
		name       string
		key        string
		defaultVal float64
		expected   float64
		setup      func(t *testing.T)
	}{
		{
			name:       "env variable exists and is valid",
			key:        "TEST_FLOAT",
			defaultVal: 0.0,
			expected:   123.456,
			setup: func(t *testing.T) {
				t.Setenv("TEST_FLOAT", "123.456")
			},
		},
		{
			name:       "env variable exists but is invalid",
			key:        "TEST_FLOAT",
			defaultVal: 99.9,
			expected:   99.9,
			setup: func(t *testing.T) {
				t.Setenv("TEST_FLOAT", "invalid")
			},
		},
		{
			name:       "env variable does not exist",
			key:        "TEST_FLOAT_MISSING",
			defaultVal: 42.42,
			expected:   42.42,
			setup:      func(t *testing.T) {},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tc.setup(t)

			result := GetEnvFloat(tc.key, tc.defaultVal, logger.V(logutil.VERBOSE))
			if result != tc.expected {
				t.Errorf("GetEnvFloat(%s, %f) = %f, expected %f", tc.key, tc.defaultVal, result, tc.expected)
			}
		})
	}
}

func TestGetEnvDuration(t *testing.T) {
	logger := testr.New(t)

	tests := []struct {
		name       string
		key        string
		defaultVal time.Duration
		expected   time.Duration
		setup      func(t *testing.T)
	}{
		{
			name:       "env variable exists and is valid",
			key:        "TEST_DURATION",
			defaultVal: 0,
			expected:   1*time.Hour + 30*time.Minute,
			setup: func(t *testing.T) {
				t.Setenv("TEST_DURATION", "1h30m")
			},
		},
		{
			name:       "env variable exists but is invalid",
			key:        "TEST_DURATION",
			defaultVal: 5 * time.Minute,
			expected:   5 * time.Minute,
			setup: func(t *testing.T) {
				t.Setenv("TEST_DURATION", "invalid-duration")
			},
		},
		{
			name:       "env variable does not exist",
			key:        "TEST_DURATION_MISSING",
			defaultVal: 10 * time.Second,
			expected:   10 * time.Second,
			setup:      func(t *testing.T) {},
		},
		{
			name:       "env variable is empty string",
			key:        "TEST_DURATION_EMPTY",
			defaultVal: 1 * time.Millisecond,
			expected:   1 * time.Millisecond,
			setup: func(t *testing.T) {
				t.Setenv("TEST_DURATION_EMPTY", "")
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tc.setup(t)

			result := GetEnvDuration(tc.key, tc.defaultVal, logger.V(logutil.VERBOSE))
			if result != tc.expected {
				t.Errorf("GetEnvDuration(%s, %v) = %v, expected %v", tc.key, tc.defaultVal, result, tc.expected)
			}
		})
	}
}

func TestGetEnvInt(t *testing.T) {
	logger := testr.New(t)

	tests := []struct {
		name       string
		key        string
		defaultVal int
		expected   int
		setup      func(t *testing.T)
	}{
		{
			name:       "env variable exists and is valid",
			key:        "TEST_INT",
			defaultVal: 0,
			expected:   123,
			setup: func(t *testing.T) {
				t.Setenv("TEST_INT", "123")
			},
		},
		{
			name:       "env variable exists but is invalid",
			key:        "TEST_INT",
			defaultVal: 99,
			expected:   99,
			setup: func(t *testing.T) {
				t.Setenv("TEST_INT", "invalid")
			},
		},
		{
			name:       "env variable does not exist",
			key:        "TEST_INT_MISSING",
			defaultVal: 42,
			expected:   42,
			setup:      func(t *testing.T) {},
		},
		{
			name:       "env variable is empty string",
			key:        "TEST_INT_EMPTY",
			defaultVal: 77,
			expected:   77,
			setup: func(t *testing.T) {
				t.Setenv("TEST_INT_EMPTY", "")
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tc.setup(t)

			result := GetEnvInt(tc.key, tc.defaultVal, logger.V(logutil.VERBOSE))
			if result != tc.expected {
				t.Errorf("GetEnvInt(%s, %d) = %d, expected %d", tc.key, tc.defaultVal, result, tc.expected)
			}
		})
	}
}

func TestGetEnvBool(t *testing.T) {
	logger := testr.New(t)

	tests := []struct {
		name       string
		key        string
		defaultVal bool
		expected   bool
		setup      func(t *testing.T)
	}{
		{
			name:       "env variable exists and is valid",
			key:        "TEST_BOOL",
			defaultVal: false,
			expected:   true,
			setup: func(t *testing.T) {
				t.Setenv("TEST_BOOL", "true")
			},
		},
		{
			name:       "env variable exists but is invalid",
			key:        "TEST_BOOL",
			defaultVal: false,
			expected:   false,
			setup: func(t *testing.T) {
				t.Setenv("TEST_BOOL", "invalid")
			},
		},
		{
			name:       "env variable does not exist",
			key:        "TEST_BOOL_MISSING",
			defaultVal: false,
			expected:   false,
			setup:      func(t *testing.T) {},
		},
		{
			name:       "env variable is empty string",
			key:        "TEST_BOOL_EMPTY",
			defaultVal: false,
			expected:   false,
			setup: func(t *testing.T) {
				t.Setenv("TEST_BOOL_EMPTY", "")
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tc.setup(t)

			result := GetEnvBool(tc.key, tc.defaultVal, logger.V(logutil.VERBOSE))
			if result != tc.expected {
				t.Errorf("GetEnvBool(%s, %v) = %v, expected %v", tc.key, tc.defaultVal, result, tc.expected)
			}
		})
	}
}

func TestGetEnvString(t *testing.T) {
	logger := testr.New(t)

	tests := []struct {
		name       string
		key        string
		defaultVal string
		expected   string
		setup      func(t *testing.T)
	}{
		{
			name:       "env variable exists and is valid",
			key:        "TEST_STR",
			defaultVal: "default",
			expected:   "123",
			setup: func(t *testing.T) {
				t.Setenv("TEST_STR", "123")
			},
		},
		{
			name:       "env variable does not exist",
			key:        "TEST_STR_MISSING",
			defaultVal: "default",
			expected:   "default",
			setup:      func(t *testing.T) {},
		},
		{
			name:       "env variable is empty string",
			key:        "TEST_STR_EMPTY",
			defaultVal: "default",
			expected:   "",
			setup: func(t *testing.T) {
				t.Setenv("TEST_STR_EMPTY", "")
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tc.setup(t)

			result := GetEnvString(tc.key, tc.defaultVal, logger.V(logutil.VERBOSE))
			if result != tc.expected {
				t.Errorf("GetEnvString(%s, %s) = %s, expected %s", tc.key, tc.defaultVal, result, tc.expected)
			}
		})
	}
}
