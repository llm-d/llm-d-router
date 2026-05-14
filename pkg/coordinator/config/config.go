package config

import (
	"fmt"
	"time"

	"github.com/spf13/viper"
)

type Config struct {
	Server    ServerConfig    `mapstructure:"server"`
	Gateway   GatewayConfig   `mapstructure:"gateway"`
	Rendering RenderingConfig `mapstructure:"rendering_service"`
	Pipeline  PipelineConfig  `mapstructure:"pipeline"`
}

type ServerConfig struct {
	ListenAddr   string        `mapstructure:"listen_addr"`
	ReadTimeout  time.Duration `mapstructure:"read_timeout"`
	WriteTimeout time.Duration `mapstructure:"write_timeout"`
}

type GatewayConfig struct {
	Address             string        `mapstructure:"address"`
	MaxIdleConnsPerHost int           `mapstructure:"max_idle_conns_per_host"`
	IdleConnTimeout     time.Duration `mapstructure:"idle_conn_timeout"`
	Timeout             time.Duration `mapstructure:"timeout"`
}

type RenderingConfig struct {
	Address string        `mapstructure:"address"`
	Timeout time.Duration `mapstructure:"timeout"`
}

type PipelineConfig struct {
	Steps []StepConfig `mapstructure:"steps"`
}

type StepConfig struct {
	Type   string         `mapstructure:"type"`
	Params map[string]any `mapstructure:"params"`
}

func Load(path string) (*Config, error) {
	v := viper.New()
	v.SetConfigFile(path)

	v.SetDefault("server.listen_addr", ":8080")
	v.SetDefault("server.read_timeout", 30*time.Second)
	v.SetDefault("server.write_timeout", 120*time.Second)
	v.SetDefault("gateway.max_idle_conns_per_host", 100)
	v.SetDefault("gateway.idle_conn_timeout", 90*time.Second)
	v.SetDefault("gateway.timeout", 60*time.Second)
	v.SetDefault("rendering_service.timeout", 30*time.Second)

	if err := v.ReadInConfig(); err != nil {
		return nil, fmt.Errorf("reading config: %w", err)
	}

	var cfg Config
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, fmt.Errorf("unmarshaling config: %w", err)
	}
	return &cfg, nil
}
