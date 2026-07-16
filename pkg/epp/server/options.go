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
	"errors"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/spf13/pflag"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/sets"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
)

const (
	DefaultGrpcPort      = 9002
	DefaultPoolNamespace = "default"        // default when pool namespace is empty (CLI flag default is empty)
	DefaultDrainTimeout  = 30 * time.Second // graceful shutdown drain window

	// DefaultStateAPIPort is the gRPC port a stateful EPP exposes its internal
	// State API on (RFC #1593 feasibility spike).
	DefaultStateAPIPort = 9004
	// DefaultStateAPIRemoteTimeout bounds a single State API call made by a
	// stateless replica, per the RFC's FailOpen/LocalFallback degradation
	// strategy: a slow or unreachable stateful EPP must not stall requests.
	DefaultStateAPIRemoteTimeout = 50 * time.Millisecond
)

// EPP operating modes (RFC #1593 feasibility spike). classic is the default
// and preserves today's active-passive behavior unchanged; stateful and
// stateless implement the RFC's active-active split.
const (
	// EppModeClassic is today's behavior: a single active-passive leader
	// serves ext-proc traffic, no State API involved.
	EppModeClassic = "classic"
	// EppModeStateful holds shared scheduling state and exposes the internal
	// gRPC State API; it does not serve ext-proc traffic.
	EppModeStateful = "stateful"
	// EppModeStateless serves ext-proc traffic from multiple concurrently
	// Ready replicas, reading/writing shared state via a stateful EPP.
	EppModeStateless = "stateless"
)

// Per-capability access modes for stateless mode (RFC #1593 feasibility
// spike), mirroring pkg/epp/statestore.AccessMode as plain strings so this
// package doesn't need to import statestore just for flag validation.
// StateAccessModeLocal isolates "more replicas sharing CPU work" from
// "remote RPC cost" by disabling remote access for a capability even when
// a State API client is configured -- see
// bin/record/stateful-rfc/perf-test/ha-benchmark.sh's Profile-D-style
// comparison, which needs exactly this to decompose a measured throughput
// delta into its two contributing factors.
const (
	// StateAccessModeLocal disables remote access; the capability behaves
	// like classic regardless of --stateful-epp-address.
	StateAccessModeLocal = "Local"
	// StateAccessModeFailOpen prefers a remote read/write and falls back to
	// the local shadow on failure or timeout. Valid for inflight and prefix.
	StateAccessModeFailOpen = "FailOpen"
	// StateAccessModeLocalFallback is flow-control only: the local queue
	// always admits first, then a fleet-wide concurrency lease; falls back
	// to a local concurrency cap when the lease is unreachable.
	StateAccessModeLocalFallback = "LocalFallback"
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
	EnableGRPCStreamMetrics bool   // Enables ext_proc gRPC stream metrics (in-flight gauge, hold duration, completions counter by code).
	//
	// Configuration.
	//
	ConfigFile string // The path to the configuration file.
	ConfigText string // The configuration specified as text, in lieu of a file.
	//
	// Stateful/stateless mode (RFC #1593 feasibility spike).
	//
	EppMode                 string        // classic (default), stateful, or stateless.
	StateAPIPort            int           // gRPC port the stateful EPP exposes its internal State API on. Stateful mode only.
	StatefulEPPAddress      string        // Address (host:port) of the stateful EPP's State API. Required in stateless mode.
	StateAPIRemoteTimeout   time.Duration // Per-call timeout for State API reads/writes from a stateless replica.
	GlobalMaxConcurrency    int64         // Fleet-wide concurrency cap enforced by the stateful EPP. Zero means unlimited.
	LocalMaxConcurrency     int64         // Local fallback concurrency cap used when the State API is unreachable. Stateless mode only.
	StateAPIArtificialDelay time.Duration // Artificial delay before the stateful EPP responds. For e2e perf-spike RTT modeling only.

	StateAccessModeInflight    string // Local or FailOpen (default). Stateless mode only.
	StateAccessModePrefix      string // Local or FailOpen (default). Stateless mode only.
	StateAccessModeFlowControl string // Local or LocalFallback (default). Stateless mode only.

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
		LoggingOptions:                   *logging.NewOptions(),
		Tracing:                          true,
		MetricsPort:                      9090,
		GRPCHealthPort:                   9003,
		EnablePprof:                      true,
		SecureServing:                    true,
		MetricsEndpointAuth:              true,
		EppMode:                          EppModeClassic,
		StateAPIPort:                     DefaultStateAPIPort,
		StateAPIRemoteTimeout:            DefaultStateAPIRemoteTimeout,
		StateAccessModeInflight:          StateAccessModeFailOpen,
		StateAccessModePrefix:            StateAccessModeFailOpen,
		StateAccessModeFlowControl:       StateAccessModeLocalFallback,
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
	fs.StringVar(&opts.ConfigFile, "config-file", opts.ConfigFile, "The path to the configuration file.")
	fs.StringVar(&opts.ConfigText, "config-text", opts.ConfigText, "The configuration specified as text, in lieu of a file.")

	fs.StringVar(&opts.EppMode, "epp-mode", opts.EppMode,
		"EPP operating mode: classic (default, single active-passive leader serves ext-proc traffic), "+
			"stateful (holds shared scheduling state, exposes the internal State API, does not serve ext-proc "+
			"traffic), or stateless (serves ext-proc traffic from multiple concurrently-ready replicas, backed "+
			"by a stateful EPP via --stateful-epp-address). Feasibility spike for RFC #1593.")
	fs.IntVar(&opts.StateAPIPort, "state-api-port", opts.StateAPIPort,
		"gRPC port the stateful EPP exposes its internal State API on. Only used in stateful mode.")
	fs.StringVar(&opts.StatefulEPPAddress, "stateful-epp-address", opts.StatefulEPPAddress,
		"Address (host:port) of the stateful EPP's State API. Required in stateless mode.")
	fs.DurationVar(&opts.StateAPIRemoteTimeout, "state-api-remote-timeout", opts.StateAPIRemoteTimeout,
		"Per-call timeout for State API reads/writes made by a stateless replica.")
	fs.Int64Var(&opts.GlobalMaxConcurrency, "flow-control-global-max-concurrency", opts.GlobalMaxConcurrency,
		"Fleet-wide concurrency cap enforced by the stateful EPP's concurrency lease. Zero means unlimited. Only used in stateful mode.")
	fs.Int64Var(&opts.LocalMaxConcurrency, "flow-control-local-max-concurrency", opts.LocalMaxConcurrency,
		"Local fallback concurrency cap used by a stateless replica when the State API is unreachable. "+
			"Recommended: flow-control-global-max-concurrency divided by the expected stateless replica count.")
	fs.DurationVar(&opts.StateAPIArtificialDelay, "state-api-artificial-delay", opts.StateAPIArtificialDelay,
		"Artificial delay added before the stateful EPP responds to any State API call. For e2e performance-spike "+
			"RTT modeling only: on a single-node kind cluster, stateless<->stateful traffic is loopback and would "+
			"otherwise understate a real cross-pod network hop.")
	fs.StringVar(&opts.StateAccessModeInflight, "state-access-mode-inflight", opts.StateAccessModeInflight,
		"Access mode for the inflight-load capability in stateless mode: FailOpen (default, prefer a remote read, "+
			"fall back to local) or Local (never call remote, isolating the pure multi-replica benefit from remote "+
			"RPC cost). Only used in stateless mode.")
	fs.StringVar(&opts.StateAccessModePrefix, "state-access-mode-prefix", opts.StateAccessModePrefix,
		"Access mode for the approximate-prefix-cache capability in stateless mode: FailOpen (default) or Local. "+
			"Only used in stateless mode.")
	fs.StringVar(&opts.StateAccessModeFlowControl, "state-access-mode-flowcontrol", opts.StateAccessModeFlowControl,
		"Access mode for the concurrency-lease capability in stateless mode: LocalFallback (default, local queue "+
			"admits first, then a fleet-wide lease) or Local (never build a remote lease). Only used in stateless mode.")
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

	// Complete logging options.
	return opts.LoggingOptions.Complete()
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

	switch opts.EppMode {
	case EppModeClassic, EppModeStateful, EppModeStateless:
	default:
		return fmt.Errorf("unexpected %q value for %q flag, it can only be set to %q, %q, or %q",
			opts.EppMode, "epp-mode", EppModeClassic, EppModeStateful, EppModeStateless)
	}
	if opts.EppMode == EppModeStateless && opts.StatefulEPPAddress == "" {
		return fmt.Errorf("%q is required when %q is %q", "stateful-epp-address", "epp-mode", EppModeStateless)
	}

	switch opts.StateAccessModeInflight {
	case StateAccessModeLocal, StateAccessModeFailOpen:
	default:
		return fmt.Errorf("unexpected %q value for %q flag, it can only be set to %q or %q",
			opts.StateAccessModeInflight, "state-access-mode-inflight", StateAccessModeLocal, StateAccessModeFailOpen)
	}
	switch opts.StateAccessModePrefix {
	case StateAccessModeLocal, StateAccessModeFailOpen:
	default:
		return fmt.Errorf("unexpected %q value for %q flag, it can only be set to %q or %q",
			opts.StateAccessModePrefix, "state-access-mode-prefix", StateAccessModeLocal, StateAccessModeFailOpen)
	}
	switch opts.StateAccessModeFlowControl {
	case StateAccessModeLocal, StateAccessModeLocalFallback:
	default:
		return fmt.Errorf("unexpected %q value for %q flag, it can only be set to %q or %q",
			opts.StateAccessModeFlowControl, "state-access-mode-flowcontrol", StateAccessModeLocal, StateAccessModeLocalFallback)
	}

	if opts.GRPCMaxRecvMsgSize < 0 {
		return fmt.Errorf("grpc-max-recv-msg-size must be non-negative, got %d", opts.GRPCMaxRecvMsgSize)
	}
	if opts.GRPCMaxSendMsgSize < 0 {
		return fmt.Errorf("grpc-max-send-msg-size must be non-negative, got %d", opts.GRPCMaxSendMsgSize)
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
