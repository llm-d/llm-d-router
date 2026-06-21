package main

import (
	"os"

	"github.com/spf13/pflag"
	ctrl "sigs.k8s.io/controller-runtime"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"

	"github.com/llm-d/coordinator/pkg/config"
	"github.com/llm-d/coordinator/pkg/gateway"
	"github.com/llm-d/coordinator/pkg/pipeline"
	"github.com/llm-d/coordinator/pkg/server"
	"github.com/llm-d/coordinator/pkg/steps"
)

func main() {
	configPath := pflag.String("config", "configs/coordinator.yaml", "path to configuration file")

	logOpts := logutil.NewOptions()
	logOpts.AddFlags(pflag.CommandLine)

	pflag.Parse()

	logutil.InitSetupLogging()
	log := ctrl.Log.WithName("coordinator")

	cfg, err := config.Load(*configPath)
	if err != nil {
		log.Error(err, "failed to load config")
		os.Exit(1)
	}

	// CLI -v wins over config log_level.
	if vFlag := pflag.CommandLine.Lookup("v"); vFlag != nil && !vFlag.Changed {
		logOpts.LogVerbosity = cfg.LogLevel
	}
	if err := logOpts.Validate(); err != nil {
		log.Error(err, "invalid logging options")
		os.Exit(1)
	}
	if err := logOpts.Complete(); err != nil {
		log.Error(err, "failed to complete logging options")
		os.Exit(1)
	}
	logutil.InitLogging(&logOpts.ZapOptions)
	log.Info("log level set", "level", logOpts.LogVerbosity)
	log.Info("pipeline connectors",
		"kv_connector", cfg.Pipeline.KVConnector,
		"ec_connector", cfg.Pipeline.ECConnector)
	// Log presence only: proxy URLs can carry basic-auth credentials
	// (http://user:pass@host) and must not reach startup logs. NO_PROXY is a
	// plain host list, so it is safe to log verbatim.
	log.Info("proxy environment",
		"http_proxy_set", os.Getenv("HTTP_PROXY") != "",
		"https_proxy_set", os.Getenv("HTTPS_PROXY") != "",
		"NO_PROXY", os.Getenv("NO_PROXY"))

	gwClient := gateway.New(cfg.Gateway)

	steps, err := buildPipeline(cfg, gwClient)
	if err != nil {
		log.Error(err, "failed to build pipeline")
		os.Exit(1)
	}

	p := pipeline.New(steps)
	srv := server.New(cfg.Server, p)

	log.Info("starting coordinator", "addr", cfg.Server.ListenAddr)
	if err := srv.ListenAndServe(); err != nil {
		log.Error(err, "server error")
		os.Exit(1)
	}
}

func mergeConnectorDefaults(params map[string]any, kvConnector, ecConnector string) map[string]any {
	out := make(map[string]any, len(params))
	for k, v := range params {
		out[k] = v
	}
	if _, ok := out[steps.ParamKVConnector]; !ok && kvConnector != "" {
		out[steps.ParamKVConnector] = kvConnector
	}
	if _, ok := out[steps.ParamECConnector]; !ok && ecConnector != "" {
		out[steps.ParamECConnector] = ecConnector
	}
	return out
}

func buildPipeline(cfg *config.Config, gwClient *gateway.Client) ([]pipeline.Step, error) {
	var steps []pipeline.Step
	for _, stepCfg := range cfg.Pipeline.Steps {
		params := mergeConnectorDefaults(stepCfg.Params, cfg.Pipeline.KVConnector, cfg.Pipeline.ECConnector)
		if _, ok := params["use_openai_format"]; !ok {
			params["use_openai_format"] = cfg.Gateway.UseOpenAIFormat
		}
		step, err := pipeline.Build(stepCfg.Type, params)
		if err != nil {
			return nil, err
		}

		// Inject the gateway client into steps that need it.
		if ga, ok := step.(gateway.ClientAware); ok {
			ga.SetGatewayClient(gwClient)
		}

		steps = append(steps, step)
	}
	return steps, nil
}
