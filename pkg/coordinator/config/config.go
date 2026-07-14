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
)

type Config struct {
	LogLevel int            `mapstructure:"log_level"`
	Server   ServerConfig   `mapstructure:"server"`
	Gateway  GatewayConfig  `mapstructure:"gateway"`
	Pipeline PipelineConfig `mapstructure:"pipeline"`
}

// BytesPerMB is the number of bytes in one megabyte.
const BytesPerMB = 1024 * 1024

// DefaultMaxRequestBodySize is the default cap for server.max_request_body_size,
// in megabytes. It is generous enough to accommodate multimodal requests that
// inline images as data: URIs; text-only deployments can lower it.
const DefaultMaxRequestBodySize = 64 // 64 MB

type ServerConfig struct {
	ListenAddr         string        `mapstructure:"listen_addr"`
	ReadTimeout        time.Duration `mapstructure:"read_timeout"`
	WriteTimeout       time.Duration `mapstructure:"write_timeout"`
	ShutdownTimeout    time.Duration `mapstructure:"shutdown_timeout"`
	MaxRequestBodySize int64         `mapstructure:"max_request_body_size"`
}

type GatewayConfig struct {
	Address             string        `mapstructure:"address"`
	MaxIdleConnsPerHost int           `mapstructure:"max_idle_conns_per_host"`
	IdleConnTimeout     time.Duration `mapstructure:"idle_conn_timeout"`
	Timeout             time.Duration `mapstructure:"timeout"`
}

type PipelineConfig struct {
	KVConnector     string       `mapstructure:"kv_connector"`
	ECConnector     string       `mapstructure:"ec_connector"`
	UseOpenAIFormat bool         `mapstructure:"use_openai_format"`
	Steps           []StepConfig `mapstructure:"steps"`
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
	v.SetDefault("server.listen_addr", ":8080")
	v.SetDefault("server.read_timeout", 30*time.Second)
	v.SetDefault("server.write_timeout", 120*time.Second)
	v.SetDefault("server.shutdown_timeout", 25*time.Second)
	v.SetDefault("server.max_request_body_size", DefaultMaxRequestBodySize)
	v.SetDefault("gateway.max_idle_conns_per_host", 100)
	v.SetDefault("gateway.idle_conn_timeout", 90*time.Second)
	v.SetDefault("gateway.timeout", 60*time.Second)
	v.SetDefault("pipeline.use_openai_format", true)

	if err := v.ReadInConfig(); err != nil {
		return nil, fmt.Errorf("reading config: %w", err)
	}

	var cfg Config
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, fmt.Errorf("unmarshalling config: %w", err)
	}
	return &cfg, nil
}
