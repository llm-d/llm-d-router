package main

import (
	"flag"
	"log/slog"
	"os"

	"github.com/llm-d/coordinator/pkg/config"
	"github.com/llm-d/coordinator/pkg/gateway"
	"github.com/llm-d/coordinator/pkg/pipeline"
	"github.com/llm-d/coordinator/pkg/server"
	_ "github.com/llm-d/coordinator/pkg/steps"
)

func main() {
	configPath := flag.String("config", "configs/coordinator.yaml", "path to configuration file")
	flag.Parse()

	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo})))

	cfg, err := config.Load(*configPath)
	if err != nil {
		slog.Error("failed to load config", "error", err)
		os.Exit(1)
	}

	gwClient := gateway.New(cfg.Gateway)

	steps, err := buildPipeline(cfg, gwClient)
	if err != nil {
		slog.Error("failed to build pipeline", "error", err)
		os.Exit(1)
	}

	p := pipeline.New(steps)
	srv := server.New(cfg.Server, p)

	slog.Info("starting coordinator", "addr", cfg.Server.ListenAddr)
	if err := srv.ListenAndServe(); err != nil {
		slog.Error("server error", "error", err)
		os.Exit(1)
	}
}

func buildPipeline(cfg *config.Config, gwClient *gateway.Client) ([]pipeline.Step, error) {
	var steps []pipeline.Step
	for _, stepCfg := range cfg.Pipeline.Steps {
		step, err := pipeline.Build(stepCfg.Type, stepCfg.Params)
		if err != nil {
			return nil, err
		}

		// Inject dependencies based on step type
		type gatewayAware interface {
			SetGatewayClient(*gateway.Client)
		}
		if ga, ok := step.(gatewayAware); ok {
			ga.SetGatewayClient(gwClient)
		}

		type renderAware interface {
			SetServiceAddress(string)
		}
		if ra, ok := step.(renderAware); ok {
			ra.SetServiceAddress(cfg.Rendering.Address)
		}

		steps = append(steps, step)
	}
	return steps, nil
}
