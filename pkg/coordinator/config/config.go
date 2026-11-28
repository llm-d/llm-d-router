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
	"fmt"
	"strings"
	"time"

	"github.com/spf13/viper"

	"github.com/llm-d/llm-d-router/pkg/common/request"
)

type Config struct {
	LogLevel int            `mapstructure:"log_level"`
	Tracing  bool           `mapstructure:"tracing"`
	Server   ServerConfig   `mapstructure:"server"`
	Gateway  GatewayConfig  `mapstructure:"gateway"`
	Pipeline PipelineConfig `mapstructure:"pipeline"`

	// DeprecatedKeys are keys set in the file or environment that have no
	// effect; main reports them at startup since this package has no logger.
	DeprecatedKeys []string `mapstructure:"-"`

	// DeprecatedAliases are deprecated keys set in the file or environment that
	// remain honored as compatibility aliases; main reports them at startup.
	DeprecatedAliases []DeprecatedAlias `mapstructure:"-"`
}

// DeprecatedAlias names a deprecated config key that is still honored and the
// replacement key operators should migrate to.
type DeprecatedAlias struct {
	Key         string
	Replacement string
}

// BytesPerMB is the number of bytes in one megabyte.
const BytesPerMB = 1024 * 1024

// DefaultMaxRequestBodySize is the default cap for server.max_request_body_size,
// in megabytes. It is generous enough to accommodate multimodal requests that
// inline images as data: URIs; text-only deployments can lower it.
const DefaultMaxRequestBodySize = 64 // 64 MB

type ServerConfig struct {
	ListenAddr         string        `mapstructure:"listen_addr"`
	MetricsPort        int           `mapstructure:"metrics_port"` // default 9090; non-positive disables the endpoint
	MetricsCertDir     string        `mapstructure:"metrics_cert_dir"`
	SecureServing      bool          `mapstructure:"secure_serving"`
	CertPath           string        `mapstructure:"cert_path"`
	TLSMinVersion      string        `mapstructure:"tls_min_version"`
	TLSCipherSuites    []string      `mapstructure:"tls_cipher_suites"`
	ReadTimeout        time.Duration `mapstructure:"read_timeout"`
	ShutdownTimeout    time.Duration `mapstructure:"shutdown_timeout"`
	DrainTimeout       time.Duration `mapstructure:"drain_timeout"`
	MaxRequestBodySize int64         `mapstructure:"max_request_body_size"`
}

type GatewayConfig struct {
	Address             string        `mapstructure:"address"`
	MaxIdleConnsPerHost int           `mapstructure:"max_idle_conns_per_host"`
	IdleConnTimeout     time.Duration `mapstructure:"idle_conn_timeout"`
	Timeout             time.Duration `mapstructure:"timeout"`
}

// Worker protocol values for PipelineConfig.WorkerProtocol.
const (
	WorkerProtocolHTTP = "http"
	WorkerProtocolGRPC = "grpc"
)

// ValidateWorkerProtocol checks that protocol is empty ("http" default),
// "http", or "grpc".
func ValidateWorkerProtocol(protocol string) error {
	switch protocol {
	case "", WorkerProtocolHTTP, WorkerProtocolGRPC:
		return nil
	default:
		return fmt.Errorf("pipeline.worker_protocol must be %q or %q, got %q",
			WorkerProtocolHTTP, WorkerProtocolGRPC, protocol)
	}
}

// Worker format values and parameter keys for PipelineConfig.WorkerFormat and
// step-level overrides on encode and prefill.
const (
	WorkerFormatPassthrough = "passthrough"
	WorkerFormatTokens      = "tokens"

	ParamWorkerFormat    = "worker_format"
	ParamUseOpenAIFormat = "use_openai_format"
)

// ValidateWorkerFormat checks that format is empty ("passthrough" default),
// "passthrough", or "tokens".
func ValidateWorkerFormat(format string) error {
	switch format {
	case "", WorkerFormatPassthrough, WorkerFormatTokens:
		return nil
	default:
		return fmt.Errorf("pipeline.%s must be %q or %q, got %q",
			ParamWorkerFormat, WorkerFormatPassthrough, WorkerFormatTokens, format)
	}
}

// WorkerFormatFromLegacyBool maps the deprecated use_openai_format boolean to
// its worker_format enum value.
func WorkerFormatFromLegacyBool(useOpenAIFormat bool) string {
	if useOpenAIFormat {
		return WorkerFormatPassthrough
	}
	return WorkerFormatTokens
}

type PipelineConfig struct {
	KVConnector    string `mapstructure:"kv_connector"`
	ECConnector    string `mapstructure:"ec_connector"`
	WorkerProtocol string `mapstructure:"worker_protocol"`
	WorkerFormat   string `mapstructure:"worker_format"`
	// UseOpenAIFormat maps true to WorkerFormatPassthrough and false to
	// WorkerFormatTokens.
	//
	// Deprecated: Use WorkerFormat instead.
	UseOpenAIFormat           *bool        `mapstructure:"use_openai_format"`
	ForwardResponseHeaders    []string     `mapstructure:"forward_response_headers"`
	ForceStream               bool         `mapstructure:"force_stream"`
	ForceStreamBufferSize     string       `mapstructure:"force_stream_buffer_size"`
	ForceStreamMaxRequestSize string       `mapstructure:"force_stream_max_request_size"`
	MaxStepReruns             int          `mapstructure:"max_step_reruns"`
	Steps                     []StepConfig `mapstructure:"steps"`
}

// ResolvedWorkerFormat validates and returns the effective worker format from
// WorkerFormat and the deprecated UseOpenAIFormat alias, defaulting to
// WorkerFormatPassthrough when neither is set.
func (p PipelineConfig) ResolvedWorkerFormat() (string, error) {
	if p.WorkerFormat != "" && p.UseOpenAIFormat != nil {
		return "", fmt.Errorf("pipeline.%s and deprecated pipeline.%s cannot both be set",
			ParamWorkerFormat, ParamUseOpenAIFormat)
	}
	if p.WorkerFormat != "" {
		if err := ValidateWorkerFormat(p.WorkerFormat); err != nil {
			return "", err
		}
		return p.WorkerFormat, nil
	}
	if p.UseOpenAIFormat != nil {
		return WorkerFormatFromLegacyBool(*p.UseOpenAIFormat), nil
	}
	return WorkerFormatPassthrough, nil
}

type StepConfig struct {
	Type   string         `mapstructure:"type"`
	Params map[string]any `mapstructure:"params"`
}

func Load(path string) (*Config, error) {
	v := viper.New()
	v.SetConfigFile(path)
	v.SetEnvPrefix("COORDINATOR")
	v.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))
	v.AutomaticEnv()

	v.SetDefault("log_level", 2)
	v.SetDefault("tracing", false)
	v.SetDefault("server.listen_addr", ":8080")
	v.SetDefault("server.metrics_port", 9090)
	v.SetDefault("server.metrics_cert_dir", "")
	v.SetDefault("server.secure_serving", true)
	v.SetDefault("server.cert_path", "")
	v.SetDefault("server.tls_min_version", "")
	v.SetDefault("server.tls_cipher_suites", []string{})
	v.SetDefault("server.read_timeout", 30*time.Second)
	v.SetDefault("server.shutdown_timeout", 25*time.Second)
	v.SetDefault("server.drain_timeout", 0)
	v.SetDefault("server.max_request_body_size", DefaultMaxRequestBodySize)
	v.SetDefault("gateway.max_idle_conns_per_host", 100)
	v.SetDefault("gateway.idle_conn_timeout", 90*time.Second)
	v.SetDefault("gateway.timeout", 60*time.Second)
	v.SetDefault("pipeline.worker_protocol", WorkerProtocolHTTP)
	// Bind both format keys without defaults so v.IsSet distinguishes explicit
	// file or env values from the default for conflict and deprecation checks.
	_ = v.BindEnv("pipeline." + ParamWorkerFormat)
	_ = v.BindEnv("pipeline." + ParamUseOpenAIFormat)
	v.SetDefault("pipeline.forward_response_headers", []string{request.DisaggregatedRevisionHeaderKey})
	v.SetDefault("pipeline.force_stream", false)
	v.SetDefault("pipeline.force_stream_buffer_size", "")
	v.SetDefault("pipeline.force_stream_max_request_size", "")
	v.SetDefault("pipeline.max_step_reruns", 0)

	if err := v.ReadInConfig(); err != nil {
		return nil, fmt.Errorf("reading config: %w", err)
	}

	var cfg Config
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, fmt.Errorf("unmarshalling config: %w", err)
	}
	if v.IsSet("server.write_timeout") {
		cfg.DeprecatedKeys = append(cfg.DeprecatedKeys, "server.write_timeout")
	}

	hasFormat := v.IsSet("pipeline." + ParamWorkerFormat)
	hasLegacy := v.IsSet("pipeline." + ParamUseOpenAIFormat)
	if hasFormat && hasLegacy {
		return nil, fmt.Errorf("pipeline.%s and deprecated pipeline.%s cannot both be set",
			ParamWorkerFormat, ParamUseOpenAIFormat)
	}
	if hasLegacy {
		cfg.DeprecatedAliases = append(cfg.DeprecatedAliases, DeprecatedAlias{
			Key:         "pipeline." + ParamUseOpenAIFormat,
			Replacement: "pipeline." + ParamWorkerFormat,
		})
	}
	workerFormat, err := cfg.Pipeline.ResolvedWorkerFormat()
	if err != nil {
		return nil, err
	}
	cfg.Pipeline.WorkerFormat = workerFormat
	cfg.Pipeline.UseOpenAIFormat = nil

	for i, step := range cfg.Pipeline.Steps {
		if _, ok := step.Params[ParamUseOpenAIFormat]; ok {
			cfg.DeprecatedAliases = append(cfg.DeprecatedAliases, DeprecatedAlias{
				Key:         fmt.Sprintf("pipeline.steps[%d].params.%s", i, ParamUseOpenAIFormat),
				Replacement: fmt.Sprintf("pipeline.steps[%d].params.%s", i, ParamWorkerFormat),
			})
		}
	}
	return &cfg, nil
}
