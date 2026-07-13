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

package config

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// writeConfig writes body to a temp file and returns its path.
func writeConfig(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "coordinator.yaml")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("writing fixture: %v", err)
	}
	return path
}

func TestLoadDefaults(t *testing.T) {
	// An empty file: every assertion below comes from SetDefault, not YAML.
	cfg, err := Load(writeConfig(t, ""))
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}

	checks := []struct {
		name string
		got  any
		want any
	}{
		{"log_level", cfg.LogLevel, 2},
		{"server.listen_addr", cfg.Server.ListenAddr, ":8080"},
		{"server.read_timeout", cfg.Server.ReadTimeout, 30 * time.Second},
		{"server.write_timeout", cfg.Server.WriteTimeout, 120 * time.Second},
		{"server.shutdown_timeout", cfg.Server.ShutdownTimeout, 25 * time.Second},
		{"server.max_request_body_size", cfg.Server.MaxRequestBodySize, int64(DefaultMaxRequestBodySize)},
		{"gateway.max_idle_conns_per_host", cfg.Gateway.MaxIdleConnsPerHost, 100},
		{"gateway.idle_conn_timeout", cfg.Gateway.IdleConnTimeout, 90 * time.Second},
		{"gateway.timeout", cfg.Gateway.Timeout, 60 * time.Second},
		{"pipeline.use_openai_format", cfg.Pipeline.UseOpenAIFormat, true},
	}
	for _, c := range checks {
		if c.got != c.want {
			t.Errorf("%s = %v, want %v", c.name, c.got, c.want)
		}
	}
}

func TestLoadEnvOverride(t *testing.T) {
	const body = "log_level: 2\npipeline:\n  use_openai_format: true\n"

	tests := []struct {
		name   string
		envKey string
		envVal string
		check  func(*Config) (got, want any)
	}{
		{
			name:   "top-level scalar",
			envKey: "COORDINATOR_LOG_LEVEL",
			envVal: "5",
			check:  func(c *Config) (any, any) { return c.LogLevel, 5 },
		},
		{
			// Nested keys reach viper only because Load installs a "." -> "_"
			// env key replacer; config/coordinator/coordinator.yaml documents this var.
			name:   "nested bool",
			envKey: "COORDINATOR_PIPELINE_USE_OPENAI_FORMAT",
			envVal: "false",
			check:  func(c *Config) (any, any) { return c.Pipeline.UseOpenAIFormat, false },
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv(tt.envKey, tt.envVal)
			cfg, err := Load(writeConfig(t, body))
			if err != nil {
				t.Fatalf("Load() error = %v", err)
			}
			if got, want := tt.check(cfg); got != want {
				t.Errorf("%s override: got %v, want %v", tt.envKey, got, want)
			}
		})
	}
}

func TestLoadStepParams(t *testing.T) {
	const body = `log_level: 2
pipeline:
  steps:
    - type: replace-media-urls
      params:
        download_timeout: 10s
        max_concurrent_downloads: 10
        allow_private_networks: true
        allowed_domains:
          - images.example.com
    - type: prefill
`
	cfg, err := Load(writeConfig(t, body))
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}

	if len(cfg.Pipeline.Steps) != 2 {
		t.Fatalf("got %d steps, want 2", len(cfg.Pipeline.Steps))
	}

	first := cfg.Pipeline.Steps[0]
	if first.Type != "replace-media-urls" {
		t.Errorf("step[0].type = %q, want replace-media-urls", first.Type)
	}
	// Params is map[string]any, so each value keeps the type YAML decoded it
	// to; steps assert these at decode time. Pin the representative shapes.
	if v, ok := first.Params["download_timeout"].(string); !ok || v != "10s" {
		t.Errorf("download_timeout = %#v, want string \"10s\"", first.Params["download_timeout"])
	}
	if v, ok := first.Params["max_concurrent_downloads"].(int); !ok || v != 10 {
		t.Errorf("max_concurrent_downloads = %#v, want int 10", first.Params["max_concurrent_downloads"])
	}
	if v, ok := first.Params["allow_private_networks"].(bool); !ok || v != true {
		t.Errorf("allow_private_networks = %#v, want bool true", first.Params["allow_private_networks"])
	}
	if _, ok := first.Params["allowed_domains"].([]any); !ok {
		t.Errorf("allowed_domains = %#v, want []any", first.Params["allowed_domains"])
	}

	// A step with no params block decodes to a nil/empty map, not an error.
	if cfg.Pipeline.Steps[1].Type != "prefill" {
		t.Errorf("step[1].type = %q, want prefill", cfg.Pipeline.Steps[1].Type)
	}
	if len(cfg.Pipeline.Steps[1].Params) != 0 {
		t.Errorf("step[1].params = %#v, want empty", cfg.Pipeline.Steps[1].Params)
	}
}

func TestLoadMissingFile(t *testing.T) {
	if _, err := Load(filepath.Join(t.TempDir(), "does-not-exist.yaml")); err == nil {
		t.Fatal("Load() of missing file: got nil error, want failure")
	}
}
