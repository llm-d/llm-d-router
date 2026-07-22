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
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/spf13/pflag"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/sets"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

const (
	DefaultGrpcPort      = 9002
	DefaultPoolNamespace = "default"        // default when pool namespace is empty (CLI flag default is empty)
	DefaultDrainTimeout  = 30 * time.Second // graceful shutdown drain window
)

// deprecatedMetricFlags lists metric flags that are superseded by engineConfigs
// in EndpointPickerConfig. They are rejected if explicitly set and suppressed from logs.
var deprecatedMetricFlags = map[string]struct{}{
	"total-queued-requests-metric":     {},
	"total-running-requests-metric":    {},
	"kv-cache-usage-percentage-metric": {},
	"lora-info-metric":                 {},
	"cache-info-metric":                {},
}

// IsDeprecatedMetricFlag reports whether the given flag name is a deprecated metric flag.
func IsDeprecatedMetricFlag(name string) bool {
	_, ok := deprecatedMetricFlags[name]
	return ok
}

// Options contains configuration values necessary to create and run the EPP.
type Options struct {
	//
	// ext_proc configuration.
	//
	GRPCPort              int           // gRPC port used for communicating with Envoy proxy. (TODO: uint16?)
	EnableLeaderElection  bool          // Enables leader election for high availability
	DrainTimeout          time.Duration // Graceful shutdown drain window; ext_proc keeps serving this long after SIGTERM.
	GRPCMaxRecvMsgSize    int           // Maximum size of a gRPC message to receive (parsed bytes).
	GRPCMaxSendMsgSize    int           // Maximum size of a gRPC message to send (parsed bytes).
	GRPCMaxRecvMsgSizeStr string        // Raw string value from CLI flag for receive limit.
	GRPCMaxSendMsgSizeStr string        // Raw string value from CLI flag for send limit.
	//
	// InferencePool.
	//
	PoolGroup     string // Kubernetes resource group of the InferencePool this Endpoint Picker is associated with.
	PoolNamespace string // Namespace of the InferencePool this Endpoint Picker is associated with.
	PoolName      string // Name of the InferencePool this Endpoint Picker is associated with.
	//
	// Endpoints (in lieu of using an InferencePool for service discovery).
	//
	EndpointSelector            labels.Selector // Parsed selector to filter model server pods on. Set via --endpoint-selector flag and parsed in Complete().
	EndpointTargetPorts         []int           // Target ports of model server pods.
	DisableEndpointSubsetFilter bool            // Disables respecting destination endpoint subset metadata in EPP.
	//
	// MSP metrics scraping.
	//
	RefreshMetricsInterval           time.Duration // Interval to refresh metrics.
	RefreshPrometheusMetricsInterval time.Duration // Interval to flush Prometheus metrics.
	MetricsStalenessThreshold        time.Duration // Duration after which metrics are considered stale.
	TotalQueuedRequestsMetric        string        // Prometheus metric specification for the number of queued requests.
	TotalRunningRequestsMetric       string        // Prometheus metric specification for the number of running requests.
	KVCacheUsagePercentageMetric     string        // Prometheus metric specification for the fraction of KV-cache blocks currently in use.
	LoRAInfoMetric                   string        // Prometheus metric specification for the LoRA info metrics.
	CacheInfoMetric                  string        // Prometheus metric specification for the cache info metrics.
	//
	// Plugin state.
	//
	PluginStateStalenessThreshold time.Duration // Inactivity duration after which a request's plugin state is reaped.
	//
	// Diagnostics.
	//
	logging.LoggingOptions         // Logging configuration.
	Tracing                 bool   // Enables emitting traces.
	HealthChecking          bool   // Enables health checking.
	MetricsPort             int    // The metrics port exposed by EPP. (TODO: uint16)
	GRPCHealthPort          int    // The port used for gRPC liveness and readiness probes. (TODO: uint16)
	EnablePprof             bool   // Enables pprof handlers.
	CertPath                string // The path to the certificate for secure serving.
	EnableCertReload        bool   // Enables certificate reloading of the certificates specified in --cert-path.
	SecureServing           bool   // Enables secure serving.
	MetricsEndpointAuth     bool   // Enables authentication and authorization of the metrics endpoint.
	MetricsClientCAFile     string // PEM CA that requires a verified client cert on the metrics endpoint.
	MetricsCertDir          string // Directory with the metrics server certificates that enables metrics TLS.
	EnableGRPCStreamMetrics bool   // Enables ext_proc gRPC stream metrics (in-flight gauge, hold duration, completions counter by code).
	//
	// Configuration.
	//
	ConfigFile string // The path to the configuration file.
	ConfigText string // The configuration specified as text, in lieu of a file.

	// internal
	fs                  *pflag.FlagSet // FlagSet used in AddFlags() and consulted in Validate()
	endpointSelectorStr string         // Raw string from --endpoint-selector flag, parsed to EndpointSelector in Complete()
}

// NewOptions returns a new Options struct initialized with the default values.
func NewOptions() *Options {
	return &Options{ // "zero" values are no explicitly set
		GRPCPort:                         DefaultGrpcPort,
		DrainTimeout:                     DefaultDrainTimeout,
		PoolGroup:                        routing.InferencePoolAPIGroup,
		EndpointTargetPorts:              []int{},
		DisableEndpointSubsetFilter:      false,
		RefreshMetricsInterval:           50 * time.Millisecond,
		RefreshPrometheusMetricsInterval: 5 * time.Second,
		MetricsStalenessThreshold:        2 * time.Second,
		TotalQueuedRequestsMetric:        "vllm:num_requests_waiting",
		TotalRunningRequestsMetric:       "vllm:num_requests_running",
		KVCacheUsagePercentageMetric:     "vllm:kv_cache_usage_perc",
		LoRAInfoMetric:                   "vllm:lora_requests_info",
		CacheInfoMetric:                  "vllm:cache_config_info",
		PluginStateStalenessThreshold:    fwkplugin.DefaultStalenessThreshold,
		LoggingOptions:                   *logging.NewOptions(),
		Tracing:                          true,
		MetricsPort:                      9090,
		GRPCHealthPort:                   9003,
		EnablePprof:                      true,
		SecureServing:                    true,
		MetricsEndpointAuth:              true,
	}
}

func (opts *Options) AddFlags(fs *pflag.FlagSet) {
	if fs == nil {
		fs = pflag.CommandLine
	}
	opts.fs = fs

	fs.IntVar(&opts.GRPCPort, "grpc-port", opts.GRPCPort, "gRPC port used for communicating with Envoy proxy.")
	fs.BoolVar(&opts.EnableLeaderElection, "ha-enable-leader-election", opts.EnableLeaderElection,
		"Enables leader election for high availability. When enabled, readiness probes will only pass on the leader.")
	fs.DurationVar(&opts.DrainTimeout, "drain-timeout", opts.DrainTimeout,
		"Graceful shutdown drain window. On SIGTERM the EPP goes NotServing and releases its leader lease "+
			"immediately, then keeps serving ext_proc for this duration so in-flight and pre-DNS-refresh requests "+
			"are not rejected.")
	fs.StringVar(&opts.GRPCMaxRecvMsgSizeStr, "grpc-max-recv-msg-size", opts.GRPCMaxRecvMsgSizeStr, "Maximum size of a gRPC message to receive (e.g., 10MiB, 25MB).")
	fs.StringVar(&opts.GRPCMaxSendMsgSizeStr, "grpc-max-send-msg-size", opts.GRPCMaxSendMsgSizeStr, "Maximum size of a gRPC message to send (e.g., 10MiB, 25MB).")
	fs.StringVar(&opts.PoolGroup, "pool-group", opts.PoolGroup,
		"Kubernetes resource group of the InferencePool this Endpoint Picker is associated with. Only `inference.networking.k8s.io/v1` is currently supported.")
	fs.StringVar(&opts.PoolNamespace, "pool-namespace", opts.PoolNamespace,
		"Namespace of the InferencePool this Endpoint Picker is associated with.")
	fs.StringVar(&opts.PoolName, "pool-name", opts.PoolName, "Name of the InferencePool this Endpoint Picker is associated with.")
	fs.StringVar(&opts.endpointSelectorStr, "endpoint-selector", opts.endpointSelectorStr,
		"Selector to filter model server pods on. "+
			"Supports Kubernetes label selector syntax: equality-based (e.g., 'app=vllm,env=prod'), "+
			"set-based (e.g., 'env in (prod,staging),tier!=frontend'), and existence (e.g., 'key,!deprecated').")
	fs.IntSliceVar(&opts.EndpointTargetPorts, "endpoint-target-ports", opts.EndpointTargetPorts, "Target ports of model server pods. "+
		"Format: a comma-separated list of numbers without whitespace (e.g., '3000,3001,3002').")
	fs.BoolVar(&opts.DisableEndpointSubsetFilter, "disable-endpoint-subset-filter", opts.DisableEndpointSubsetFilter,
		"Disables respecting the destination endpoint subset metadata for dispatching requests in EPP.")
	fs.DurationVar(&opts.RefreshMetricsInterval, "refresh-metrics-interval", opts.RefreshMetricsInterval, "Interval to refresh metrics.")
	fs.DurationVar(&opts.RefreshPrometheusMetricsInterval, "refresh-prometheus-metrics-interval", opts.RefreshPrometheusMetricsInterval,
		"Interval to flush Prometheus metrics.")
	fs.DurationVar(&opts.MetricsStalenessThreshold, "metrics-staleness-threshold", opts.MetricsStalenessThreshold,
		"Duration after which metrics are considered stale. This is used to determine if an endpoint's metrics are fresh enough.")
	fs.StringVar(&opts.TotalQueuedRequestsMetric, "total-queued-requests-metric", opts.TotalQueuedRequestsMetric,
		"Prometheus metric for the number of queued requests.")
	_ = fs.MarkDeprecated("total-queued-requests-metric", "use engineConfigs in EndpointPickerConfig instead")
	fs.StringVar(&opts.TotalRunningRequestsMetric, "total-running-requests-metric", opts.TotalRunningRequestsMetric,
		"Prometheus metric for the number of running requests.")
	_ = fs.MarkDeprecated("total-running-requests-metric", "use engineConfigs in EndpointPickerConfig instead")
	fs.StringVar(&opts.KVCacheUsagePercentageMetric, "kv-cache-usage-percentage-metric", opts.KVCacheUsagePercentageMetric,
		"Prometheus metric for the fraction of KV-cache blocks currently in use (from 0 to 1).")
	_ = fs.MarkDeprecated("kv-cache-usage-percentage-metric", "use engineConfigs in EndpointPickerConfig instead")
	fs.StringVar(&opts.LoRAInfoMetric, "lora-info-metric", opts.LoRAInfoMetric,
		"Prometheus metric for the LoRA info metrics (must be in vLLM label format).")
	_ = fs.MarkDeprecated("lora-info-metric", "use engineConfigs in EndpointPickerConfig instead")
	fs.StringVar(&opts.CacheInfoMetric, "cache-info-metric", opts.CacheInfoMetric, "Prometheus metric for the cache info metrics.")
	_ = fs.MarkDeprecated("cache-info-metric", "use engineConfigs in EndpointPickerConfig instead")

	fs.DurationVar(&opts.PluginStateStalenessThreshold, "plugin-state-staleness-threshold", opts.PluginStateStalenessThreshold,
		"Inactivity duration after which a request's plugin state (e.g. in-flight load accounting) is reaped. "+
			"Set this above the maximum expected request duration to avoid releasing in-flight accounting for long-running requests.")

	opts.LoggingOptions.AddFlags(fs) // Add logging flags.

	fs.BoolVar(&opts.Tracing, "tracing", opts.Tracing, "Enables emitting traces.")
	fs.BoolVar(&opts.HealthChecking, "health-checking", opts.HealthChecking, "Enables health checking.")
	fs.IntVar(&opts.MetricsPort, "metrics-port", opts.MetricsPort, "The metrics port exposed by EPP.")
	fs.IntVar(&opts.GRPCHealthPort, "grpc-health-port", opts.GRPCHealthPort,
		"The port used for gRPC liveness and readiness probes.")
	fs.BoolVar(&opts.EnablePprof, "enable-pprof", opts.EnablePprof,
		"Enables pprof handlers. Defaults to true. Set to false to disable pprof handlers.")
	fs.StringVar(&opts.CertPath, "cert-path", opts.CertPath,
		"The path to the certificate for secure serving. The certificate and private key files "+
			"are assumed to be named tls.crt and tls.key, respectively. If not set, and secureServing is enabled, "+
			"then a self-signed certificate is used.")
	fs.BoolVar(&opts.EnableCertReload, "enable-cert-reload", opts.EnableCertReload,
		"Enables certificate reloading of the certificates specified in --cert-path.")
	fs.BoolVar(&opts.EnableGRPCStreamMetrics, "enable-grpc-stream-metrics", opts.EnableGRPCStreamMetrics,
		"Enables ext_proc gRPC stream metrics (in-flight gauge, hold-duration histogram, completions counter by code).")
	fs.BoolVar(&opts.SecureServing, "secure-serving", opts.SecureServing, "Enables secure serving.")
	fs.BoolVar(&opts.MetricsEndpointAuth, "metrics-endpoint-auth", opts.MetricsEndpointAuth,
		"Enables authentication and authorization of the metrics endpoint.")
	fs.StringVar(&opts.MetricsClientCAFile, "metrics-client-ca-file", opts.MetricsClientCAFile,
		"PEM CA for metrics mTLS: require verified client certs.")
	fs.StringVar(&opts.MetricsCertDir, "metrics-cert-dir", opts.MetricsCertDir,
		"Directory with the metrics server certificates. Enables TLS on the metrics endpoint.")
	fs.StringVar(&opts.ConfigFile, "config-file", opts.ConfigFile, "The path to the configuration file.")
	fs.StringVar(&opts.ConfigText, "config-text", opts.ConfigText, "The configuration specified as text, in lieu of a file.")
}

func (opts *Options) Complete() error {
	if opts.endpointSelectorStr != "" {
		selector, err := labels.Parse(opts.endpointSelectorStr)
		if err != nil {
			return fmt.Errorf("invalid endpoint-selector %q: %w", opts.endpointSelectorStr, err)
		}
		opts.EndpointSelector = selector
	}

	opts.EndpointTargetPorts = removeDuplicatePorts(opts.EndpointTargetPorts)

	if opts.GRPCMaxRecvMsgSizeStr != "" {
		s := sanitizeSizeString(opts.GRPCMaxRecvMsgSizeStr)
		q, err := resource.ParseQuantity(s)
		if err != nil {
			return fmt.Errorf("invalid grpc-max-recv-msg-size: %w", err)
		}
		val, ok := q.AsInt64()
		if !ok {
			return fmt.Errorf("grpc-max-recv-msg-size overflows maximum supported size: %s", s)
		}
		if val < 0 {
			return fmt.Errorf("grpc-max-recv-msg-size must be non-negative, got %d", val)
		}
		if val > int64(math.MaxInt) {
			return fmt.Errorf("grpc-max-recv-msg-size overflows int: %d", val)
		}
		opts.GRPCMaxRecvMsgSize = int(val)
	}
	if opts.GRPCMaxSendMsgSizeStr != "" {
		s := sanitizeSizeString(opts.GRPCMaxSendMsgSizeStr)
		q, err := resource.ParseQuantity(s)
		if err != nil {
			return fmt.Errorf("invalid grpc-max-send-msg-size: %w", err)
		}
		val, ok := q.AsInt64()
		if !ok {
			return fmt.Errorf("grpc-max-send-msg-size overflows maximum supported size: %s", s)
		}
		if val < 0 {
			return fmt.Errorf("grpc-max-send-msg-size must be non-negative, got %d", val)
		}
		if val > int64(math.MaxInt) {
			return fmt.Errorf("grpc-max-send-msg-size overflows int: %d", val)
		}
		opts.GRPCMaxSendMsgSize = int(val)
	}

	if opts.MetricsClientCAFile != "" {
		if _, err := os.Stat(opts.MetricsClientCAFile); err != nil {
			return fmt.Errorf("%w: %s: %v", errMetricsCertUnreadable, opts.MetricsClientCAFile, err)
		}
	}
	if opts.MetricsCertDir != "" {
		for _, name := range []string{"tls.crt", "tls.key"} {
			p := filepath.Join(opts.MetricsCertDir, name)
			if _, err := os.Stat(p); err != nil {
				return fmt.Errorf("%w: %s: %v", errMetricsCertUnreadable, p, err)
			}
		}
	}

	// Complete logging options.
	return opts.LoggingOptions.Complete()
}

var (
	errMetricsClientCARequiresCertDir = errors.New(`"metrics-client-ca-file" requires "metrics-cert-dir"`)
	errMetricsTLSWithoutAuth          = errors.New(`"metrics-cert-dir" enables metrics TLS without authentication; set "metrics-client-ca-file" or "metrics-endpoint-auth"`)
	errMetricsCertUnreadable          = errors.New("metrics TLS cert file unreadable")
	errReadMetricsClientCA            = errors.New("reading metrics client CA")
	errNoValidMetricsCA               = errors.New("no valid CA certs in metrics client CA file")
)

// ConfigureMetricsTLS enables TLS, and optional client mTLS, on the metrics server.
func ConfigureMetricsTLS(opts *Options, mo *metricsserver.Options) error {
	if opts.MetricsCertDir == "" && opts.MetricsClientCAFile == "" {
		return nil
	}
	mo.SecureServing = true
	if opts.MetricsCertDir != "" {
		mo.CertDir = opts.MetricsCertDir
	}
	var clientCAs *x509.CertPool
	if opts.MetricsClientCAFile != "" {
		caPEM, err := os.ReadFile(opts.MetricsClientCAFile)
		if err != nil {
			return fmt.Errorf("%w %s: %v", errReadMetricsClientCA, opts.MetricsClientCAFile, err)
		}
		clientCAs = x509.NewCertPool()
		if !clientCAs.AppendCertsFromPEM(caPEM) {
			return fmt.Errorf("%w: %s", errNoValidMetricsCA, opts.MetricsClientCAFile)
		}
	}
	mo.TLSOpts = append(mo.TLSOpts, func(c *tls.Config) {
		c.MinVersion = tls.VersionTLS12
		if clientCAs != nil {
			c.ClientCAs = clientCAs
			c.ClientAuth = tls.RequireAndVerifyClientCert
		}
	})
	return nil
}

func (opts *Options) Validate() error {
	if (opts.PoolName != "" && opts.EndpointSelector != nil) || (opts.PoolName == "" && opts.EndpointSelector == nil) {
		return errors.New("either pool-name or endpoint-selector must be set")
	}
	if opts.EndpointSelector != nil {
		if len(opts.EndpointTargetPorts) == 0 || len(opts.EndpointTargetPorts) > 8 {
			return fmt.Errorf("flag %q should have length from 1 to 8", "endpoint-target-ports")
		}
		for _, port := range opts.EndpointTargetPorts { // valid port range
			if port < 0 || port > 65535 {
				return fmt.Errorf("invalid port number %d in %q", port, "endpoint-target-ports")
			}
		}
	}

	if opts.ConfigText != "" && opts.ConfigFile != "" {
		return fmt.Errorf("both the %q and %q flags cannot be set at the same time", "config-file", "config-text")
	}

	if opts.MetricsClientCAFile != "" && opts.MetricsCertDir == "" {
		return errMetricsClientCARequiresCertDir
	}
	if opts.MetricsCertDir != "" && opts.MetricsClientCAFile == "" && !opts.MetricsEndpointAuth {
		return errMetricsTLSWithoutAuth
	}

	if opts.GRPCMaxRecvMsgSize < 0 {
		return fmt.Errorf("grpc-max-recv-msg-size must be non-negative, got %d", opts.GRPCMaxRecvMsgSize)
	}
	if opts.GRPCMaxSendMsgSize < 0 {
		return fmt.Errorf("grpc-max-send-msg-size must be non-negative, got %d", opts.GRPCMaxSendMsgSize)
	}

	if opts.PluginStateStalenessThreshold <= 0 {
		return fmt.Errorf("plugin-state-staleness-threshold must be positive, got %v", opts.PluginStateStalenessThreshold)
	}

	// Validate deprecated metric flags are not explicitly set
	for flagName := range deprecatedMetricFlags {
		if f := opts.fs.Lookup(flagName); f != nil && f.Changed {
			return fmt.Errorf("flag %q is deprecated and cannot be used; configure metrics via engineConfigs in EndpointPickerConfig instead", flagName)
		}
	}

	// Validate logging options.
	return opts.LoggingOptions.Validate()
}

func sanitizeSizeString(s string) string {
	s = strings.TrimSpace(s)
	if len(s) > 1 && (s[len(s)-1] == 'B' || s[len(s)-1] == 'b') {
		return s[:len(s)-1]
	}
	return s
}

func removeDuplicatePorts(ports []int) []int {
	seen := sets.NewInt()
	unique := make([]int, 0, len(ports))

	for _, val := range ports {
		if !seen.Has(val) {
			unique = append(unique, val)
			seen.Insert(val)
		}
	}
	return unique
}
