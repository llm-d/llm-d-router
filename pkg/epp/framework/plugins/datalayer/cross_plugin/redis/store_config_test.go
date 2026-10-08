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

package redis

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/stretchr/testify/require"
)

func TestFactoryUsesIndependentStateAndCoordinationTTLs(t *testing.T) {
	server := miniredis.RunT(t)
	plugin, err := RedisStateStoreFactory("redis", json.NewDecoder(strings.NewReader(fmt.Sprintf(
		`{"address": %q, "stateTTL": "750ms", "coordinationTTL": "3m"}`,
		server.Addr(),
	))), nil)
	require.NoError(t, err)
	store := plugin.(*RedisStateStore)
	t.Cleanup(func() { require.NoError(t, store.client.Close()) })

	require.Equal(t, 750*time.Millisecond, store.stateTTL)
	require.Equal(t, 3*time.Minute, store.coordinationTTL)
}

func TestFactoryReadsPasswordFromEnvironment(t *testing.T) {
	server := miniredis.RunT(t)
	t.Setenv("TEST_REDIS_PASSWORD", "environment-secret")
	plugin, err := RedisStateStoreFactory("redis", json.NewDecoder(strings.NewReader(fmt.Sprintf(
		`{"address": %q, "passwordEnv": "TEST_REDIS_PASSWORD"}`,
		server.Addr(),
	))), nil)
	require.NoError(t, err)
	store := plugin.(*RedisStateStore)
	t.Cleanup(func() { require.NoError(t, store.client.Close()) })

	require.Equal(t, "environment-secret", store.client.Options().Password)
}

func TestFactoryReadsPasswordFromFile(t *testing.T) {
	server := miniredis.RunT(t)
	passwordPath := filepath.Join(t.TempDir(), "password")
	require.NoError(t, os.WriteFile(passwordPath, []byte("file-secret"), 0o600))
	plugin, err := RedisStateStoreFactory("redis", json.NewDecoder(strings.NewReader(fmt.Sprintf(
		`{"address": %q, "passwordFile": %q}`,
		server.Addr(), passwordPath,
	))), nil)
	require.NoError(t, err)
	store := plugin.(*RedisStateStore)
	t.Cleanup(func() { require.NoError(t, store.client.Close()) })

	require.Equal(t, "file-secret", store.client.Options().Password)
}

func TestFactoryRejectsUnsafeOrAmbiguousPasswordConfiguration(t *testing.T) {
	server := miniredis.RunT(t)
	t.Setenv("TEST_REDIS_PASSWORD", "secret")
	passwordPath := filepath.Join(t.TempDir(), "password")
	require.NoError(t, os.WriteFile(passwordPath, []byte("secret"), 0o600))

	testCases := []struct {
		name    string
		params  string
		wantErr string
	}{
		{
			name:    "plaintext password",
			params:  fmt.Sprintf(`{"address": %q, "password": "secret"}`, server.Addr()),
			wantErr: "passwordEnv or passwordFile",
		},
		{
			name: "environment and file",
			params: fmt.Sprintf(
				`{"address": %q, "passwordEnv": "TEST_REDIS_PASSWORD", "passwordFile": %q}`,
				server.Addr(), passwordPath,
			),
			wantErr: "mutually exclusive",
		},
		{
			name:    "missing environment variable",
			params:  fmt.Sprintf(`{"address": %q, "passwordEnv": "MISSING_REDIS_PASSWORD"}`, server.Addr()),
			wantErr: "MISSING_REDIS_PASSWORD",
		},
		{
			name:    "missing file",
			params:  fmt.Sprintf(`{"address": %q, "passwordFile": %q}`, server.Addr(), filepath.Join(t.TempDir(), "missing")),
			wantErr: "password file",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			plugin, err := RedisStateStoreFactory("redis", json.NewDecoder(strings.NewReader(tc.params)), nil)
			require.ErrorContains(t, err, tc.wantErr)
			require.Nil(t, plugin)
		})
	}
}

func TestFactoryRejectsUnsupportedTLSAndACL(t *testing.T) {
	for _, params := range []string{`{"tls": {}}`, `{"username": "router"}`} {
		t.Run(params, func(t *testing.T) {
			plugin, err := RedisStateStoreFactory("redis", json.NewDecoder(strings.NewReader(params)), nil)
			if plugin != nil {
				t.Cleanup(func() { require.NoError(t, plugin.(*RedisStateStore).client.Close()) })
			}
			require.ErrorContains(t, err, "unknown field")
			require.Nil(t, plugin)
		})
	}
}
