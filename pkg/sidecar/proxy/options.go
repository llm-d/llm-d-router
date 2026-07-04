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

package proxy

import (
	"errors"
	"flag"
	"fmt"
	"net/url"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/go-logr/logr"
	"github.com/spf13/pflag"
	uberzap "go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"sigs.k8s.io/yaml"
)

const (
	// MoRIIOFeatureEnabled controls whether MoRI-IO WRITE-mode and Wide-EP
	// features are available. Set to false to keep the feature dormant until
	// full validation and CI integration is complete in a future RC.
	//
	// When false, any attempt to use --moriio-write-mode or related flags will
	// fail with a clear error message directing users to wait for the feature
	// to be officially released.
	//
	// TODO(AMD-MoRI-IO): Set to true once CI tests and production validation
	// are complete in a future release candidate.
	MoRIIOFeatureEnabled = false

	// Flags
	port                      = "port"
	vllmPort                  = "vllm-port"
	dataParallelSize          = "data-parallel-size"
	kvConnector               = "kv-connector"
	ecConnector               = "ec-connector"
	mooncakeBootstrapPortFlag = "mooncake-bootstrap-port"
	p2pConnectorPortFlag      = "p2p-connector-port"
	enableSSRFProtection      = "enable-ssrf-protection"
	enablePrefillerSampling   = "enable-prefiller-sampling"
	enableTLS                 = "enable-tls"
	tlsInsecureSkipVerify     = "tls-insecure-skip-verify"
	secureServing             = "secure-proxy"
	certPath                  = "cert-path"
	inferencePool             = "inference-pool"
	poolGroup                 = "pool-group"
	maxIdleConnsPerHost       = "max-idle-conns-per-host"
	prefillMaxRetries         = "prefill-max-retries"
	prefillRetryBackoff       = "prefill-retry-backoff"
	decodeChunkSize           = "decode-chunk-size"
	inlineConfiguration       = "configuration"
	configurationFile         = "configuration-file"
	tracingFlag               = "tracing"

	// Deprecated flags
	connector                      = "connector"
	prefillerUseTLS                = "prefiller-use-tls"
	decoderUseTLS                  = "decoder-use-tls"
	encoderUseTLS                  = "UseTLSForEncoder"
	prefillerTLSInsecureSkipVerify = "prefiller-tls-insecure-skip-verify"
	decoderTLSInsecureSkipVerify   = "decoder-tls-insecure-skip-verify"

	// Environment variables
	envInferencePool           = "INFERENCE_POOL"
	envEnablePrefillerSampling = "ENABLE_PREFILLER_SAMPLING"
	envMooncakeBootstrapPort   = "MOONCAKE_BOOTSTRAP_PORT"
	envP2PConnectorPort        = "P2P_CONNECTOR_PORT"

	// Defaults
	defaultPort                  = "8000"
	defaultVLLMPort              = "8200"
	defaultDataParallelSize      = 1
	defaultMooncakeBootstrapPort = 8998
	defaultP2PConnectorPort      = 7777

	// TLS stages
	prefillStage = "prefiller"
	decodeStage  = "decoder"
	encodeStage  = "encoder"
)

// yamlConfiguration represents structure of YAML configuration for sidecar proxy
type yamlConfiguration struct {
	Port                           int      `json:"port,omitempty"`
	VLLMPort                       int      `json:"vllm-port,omitempty"`
	MooncakeBootstrapPort          int      `json:"mooncake-bootstrap-port,omitempty"`
	P2PConnectorPort               int      `json:"p2p-connector-port,omitempty"`
	DataParallelSize               int      `json:"data-parallel-size,omitempty"`
	KVConnector                    string   `json:"kv-connector,omitempty"`
	Connector                      string   `json:"connector,omitempty"`
	ECConnector                    string   `json:"ec-connector,omitempty"`
	EnableSSRFProtection           *bool    `json:"enable-ssrf-protection,omitempty"`
	EnablePrefillerSampling        *bool    `json:"enable-prefiller-sampling,omitempty"`
	SecureServing                  *bool    `json:"secure-proxy,omitempty"`
	CertPath                       string   `json:"cert-path,omitempty"`
	EnableTLS                      []string `json:"enable-tls,omitempty"`
	TLSInsecureSkipVerify          []string `json:"tls-insecure-skip-verify,omitempty"`
	PrefillerUseTLS                *bool    `json:"prefiller-use-tls,omitempty"`
	DecoderUseTLS                  *bool    `json:"decoder-use-tls,omitempty"`
	PrefillerTLSInsecureSkipVerify *bool    `json:"prefiller-tls-insecure-skip-verify,omitempty"`
	DecoderTLSInsecureSkipVerify   *bool    `json:"decoder-tls-insecure-skip-verify,omitempty"`
	InferencePool                  string   `json:"inference-pool,omitempty"`
	PoolGroup                      string   `json:"pool-group,omitempty"`
	MaxIdleConnsPerHost            int      `json:"max-idle-conns-per-host,omitempty"`
	PrefillMaxRetries              *int     `json:"prefill-max-retries,omitempty"`
	PrefillRetryBackoff            string   `json:"prefill-retry-backoff,omitempty"`
	DecodeChunkSize                int      `json:"decode-chunk-size,omitempty"`
	Tracing                        *bool    `json:"tracing,omitempty"`
}

// Options holds the CLI-facing configuration for the pd-sidecar proxy.
// It embeds Config which represents the complete processed runtime configuration.
// After Options.Complete(), the embedded Config is fully populated and ready to
// pass directly to NewProxy.
type Options struct {
	// Config holds the processed runtime configuration (populated by Complete()).
	// Fields with direct CLI flags are bound here via embedding; derived fields are set in Complete().
	Config

	// vllmPort is the port vLLM is listening on; used to compute Config.DecoderURL in Complete().
	vllmPort string
	// enableTLS is the list of stages to enable TLS for; used to compute Config.UseTLSFor* in Complete().
	enableTLS []string
	// tlsInsecureSkipVerify is the list of stages to skip TLS verification for; used to compute Config.InsecureSkipVerifyFor* in Complete().
	tlsInsecureSkipVerify []string
	// inferencePool in namespace/name or name format; used to compute Config.InferencePoolNamespace/Name in Complete().
	inferencePool string

	// Deprecated flag fields - kept for backward compatibility; migrated in Complete()
	connector                   string // Deprecated: use --kv-connector instead
	prefillerUseTLS             bool   // Deprecated: use --enable-tls=prefiller instead
	decoderUseTLS               bool   // Deprecated: use --enable-tls=decoder instead
	prefillerInsecureSkipVerify bool   // Deprecated: use --tls-insecure-skip-verify=prefiller instead
	decoderInsecureSkipVerify   bool   // Deprecated: use --tls-insecure-skip-verify=decoder instead

	loggingOptions      zap.Options // loggingOptions holds the zap logging configuration
	pflagSet            *pflag.FlagSet
	inlineConfiguration string
	fileConfiguration   string
}

var (
	// supportedKVConnectors defines all valid P/D KV connector types
	supportedKVConnectors = map[string]struct{}{
		KVConnectorNIXLV2:        {},
		KVConnectorSharedStorage: {},
		KVConnectorSGLang:        {},
		KVConnectorMooncake:      {},
		KVConnectorOffloading:    {},
	}

	// supportedECConnectors defines all valid E/P EC connector types
	supportedECConnectors = map[string]struct{}{
		ECExampleConnector: {},
		ECConnectorNIXL:    {},
	}

	// supportedTLSStages defines all valid stages for TLS configuration
	supportedTLSStages = map[string]struct{}{
		prefillStage: {},
		decodeStage:  {},
		encodeStage:  {},
	}

	supportedKVConnectorNamesStr = strings.Join([]string{KVConnectorNIXLV2, KVConnectorSharedStorage, KVConnectorSGLang, KVConnectorMooncake, KVConnectorOffloading}, ", ")
	supportedECConnectorNamesStr = strings.Join([]string{ECExampleConnector, ECConnectorNIXL}, ", ")

	supportedTLSStageNamesStr = strings.Join([]string{prefillStage, decodeStage, encodeStage}, ", ")
)

// NewOptions returns a new Options struct initialized with default values.
func NewOptions() *Options {
	enablePrefillerSampling := false
	if val, err := strconv.ParseBool(os.Getenv(envEnablePrefillerSampling)); err == nil {
		enablePrefillerSampling = val
	}

	mooncakeBootstrapPort := defaultMooncakeBootstrapPort
	if portStr := os.Getenv(envMooncakeBootstrapPort); portStr != "" {
		if port, err := strconv.Atoi(portStr); err == nil {
			mooncakeBootstrapPort = port
		}
	}

	p2pConnectorPort := defaultP2PConnectorPort
	if portStr := os.Getenv(envP2PConnectorPort); portStr != "" {
		if port, err := strconv.Atoi(portStr); err == nil {
			p2pConnectorPort = port
		}
	}

	return &Options{
		Config: Config{
			Port:                    defaultPort,
			DataParallelSize:        defaultDataParallelSize,
			SecureServing:           true,
			EnablePrefillerSampling: enablePrefillerSampling,
			MaxIdleConnsPerHost:     defaultMaxIdleConnsPerHost,
			PrefillMaxRetries:       0,
			PrefillRetryBackoff:     200 * time.Millisecond,
			MooncakeBootstrapPort:   mooncakeBootstrapPort,
			P2PConnectorPort:        p2pConnectorPort,
			PoolGroup:               routing.InferencePoolAPIGroup,
			DecodeChunkSize:         0,
			Tracing:                 false,
			// MoRI-IO defaults: off, preserving existing NIXLv2 behaviour.
			// Port defaults match vLLM's MoRI-IO connector defaults.
			MoRIIOWriteMode:            false,
			MoRIIODecodeNotifyPort:     61005,
			MoRIIODecodeHandshakePort:  6301,
			MoRIIODecodePodIP:          os.Getenv("POD_IP"),
			MoRIIOParallelDispatch:     false,
			MoRIIOPrefillHandshakePort: 6301,
			MoRIIOPrefillNotifyPort:    61005,
			MoRIIOTPSize:               1,
			MoRIIODPSize:               1,
			// Wide-EP multi-pod: empty disables fan-out (single-pod fallback).
			MoRIIORemoteHosts: nil,
			MoRIIODPSizeLocal: 0,
			MoRIIODecodeHosts: nil,
		},
		vllmPort:      defaultVLLMPort,
		inferencePool: os.Getenv(envInferencePool),
		connector:     KVConnectorNIXLV2,
	}
}

// AddFlags binds the Options fields to command-line flags on the given FlagSet.
// It also sets up zap logging flags and integrates Go flags with pflag.
func (opts *Options) AddFlags(fs *pflag.FlagSet) {
	if opts.pflagSet == nil {
		opts.pflagSet = fs
	}
	goFlagSet := flag.NewFlagSet("goFlagSet", flag.ContinueOnError)
	// Add logging flags to the standard flag set
	opts.loggingOptions.BindFlags(goFlagSet)
	// Add Go flags to pflag (for zap options compatibility)
	fs.AddGoFlagSet(goFlagSet)
	fs.StringVar(&opts.Port, port, opts.Port, "the port the sidecar is listening on")
	fs.StringVar(&opts.vllmPort, vllmPort, opts.vllmPort, "the port vLLM is listening on")
	fs.IntVar(&opts.DataParallelSize, dataParallelSize, opts.DataParallelSize, "the vLLM DATA-PARALLEL-SIZE value")
	fs.StringVar(&opts.KVConnector, kvConnector, opts.KVConnector,
		"the KV protocol between prefiller and decoder. Supported: "+supportedKVConnectorNamesStr)
	fs.StringVar(&opts.ECConnector, ecConnector, opts.ECConnector,
		"the EC protocol between encoder and prefiller (for EPD mode). Supported: "+supportedECConnectorNamesStr+". Leave empty to skip encoder stage.")
	fs.IntVar(&opts.MooncakeBootstrapPort, mooncakeBootstrapPortFlag, opts.MooncakeBootstrapPort,
		"the port used to query the Mooncake bootstrap endpoint on prefill pods (only used with --kv-connector=mooncake)")
	fs.IntVar(&opts.P2PConnectorPort, p2pConnectorPortFlag, opts.P2PConnectorPort,
		"the prefiller's P2PConnector listening port, injected as remote_port on the decode leg (only used with --kv-connector=p2p)")
	fs.BoolVar(&opts.SecureServing, secureServing, opts.SecureServing, "Enables secure proxy. Defaults to true.")
	fs.StringVar(&opts.CertPath, certPath, opts.CertPath, "The path to the certificate for secure proxy. The certificate and private key files are assumed to be named tls.crt and tls.key, respectively. If not set, and secureProxy is enabled, then a self-signed certificate is used (for testing).")
	fs.BoolVar(&opts.EnableSSRFProtection, enableSSRFProtection, opts.EnableSSRFProtection, "enable SSRF protection using InferencePool allowlisting")
	fs.BoolVar(&opts.EnablePrefillerSampling, enablePrefillerSampling, opts.EnablePrefillerSampling, "if true, the target prefill instance will be selected randomly from among the provided prefill host values")
	fs.StringVar(&opts.PoolGroup, poolGroup, opts.PoolGroup, "group of the InferencePool this Endpoint Picker is associated with.")
	fs.IntVar(&opts.DecodeChunkSize, decodeChunkSize, opts.DecodeChunkSize, "enables chunked decode mode when > 0; value is the token budget per chunk. For best performance should be a multiple of the block size.")
	fs.BoolVar(&opts.Tracing, tracingFlag, opts.Tracing, "Enable OpenTelemetry tracing")

	// MoRI-IO WRITE-mode flags. Only meaningful with --kv-connector=nixlv2
	// against vLLM engines running MoRI-IO in WRITE mode.
	fs.BoolVar(&opts.MoRIIOWriteMode, "moriio-write-mode", opts.MoRIIOWriteMode,
		"Enable MoRI-IO WRITE-mode passthrough in kv_transfer_params. Requires --kv-connector=nixlv2.")
	fs.IntVar(&opts.MoRIIODecodeNotifyPort, "moriio-decode-notify-port", opts.MoRIIODecodeNotifyPort,
		"Base MoRI-IO notify port on the decode pod.")
	fs.StringVar(&opts.MoRIIODecodePodIP, "moriio-local-pod-ip", opts.MoRIIODecodePodIP,
		"Decode pod's routable IP, used as the prefill leg's remote_host. "+
			"Defaults to the POD_IP env var. Required with --moriio-write-mode.")
	fs.IntVar(&opts.MoRIIODecodeHandshakePort, "moriio-decode-handshake-port", opts.MoRIIODecodeHandshakePort,
		"Base MoRI-IO handshake port on the decode pod.")

	// Concurrent-dispatch flags: synthesise decode's kv_transfer_params from
	// config so prefill and decode dispatch can overlap.
	fs.BoolVar(&opts.MoRIIOParallelDispatch, "moriio-parallel-dispatch", opts.MoRIIOParallelDispatch,
		"Fire prefill and decode concurrently. Requires --moriio-write-mode.")
	fs.IntVar(&opts.MoRIIOPrefillHandshakePort, "moriio-prefill-handshake-port", opts.MoRIIOPrefillHandshakePort,
		"Prefill pod's base MoRI-IO handshake port, used to build the decode leg in parallel-dispatch mode.")
	fs.IntVar(&opts.MoRIIOPrefillNotifyPort, "moriio-prefill-notify-port", opts.MoRIIOPrefillNotifyPort,
		"Prefill pod's base MoRI-IO notify port.")
	fs.IntVar(&opts.MoRIIOTPSize, "moriio-tp-size", opts.MoRIIOTPSize,
		"Tensor-parallel size of the engines, echoed into kv_transfer_params[tp_size].")
	fs.IntVar(&opts.MoRIIODPSize, "moriio-dp-size", opts.MoRIIODPSize,
		"Data-parallel world size, emitted as kv_transfer_params[remote_dp_size] on both legs. "+
			"Set to the engine DP size for Wide-EP (TP=1, DP>1); default 1 leaves the wire unchanged.")

	// Wide-EP multi-pod fan-out. Optional: empty preserves single-pod behaviour.
	// remote_hosts carries the opposite side's pod IPs (decode IPs on the prefill
	// leg and vice versa); dp-size-local maps a global DP rank to a pod via
	// pod_idx = dp_rank / dp_size_local.
	fs.StringSliceVar(&opts.MoRIIORemoteHosts, "moriio-remote-hosts", opts.MoRIIORemoteHosts,
		"Wide-EP: comma-separated remote (prefill-side) pod IPs for per-DP-rank fan-out. "+
			"Pair with --moriio-dp-size-local.")
	fs.IntVar(&opts.MoRIIODPSizeLocal, "moriio-dp-size-local", opts.MoRIIODPSizeLocal,
		"Wide-EP: per-pod DP size used to map a global DP rank to a pod index. "+
			"Must satisfy --moriio-dp-size = dp-size-local * len(hosts).")
	fs.StringSliceVar(&opts.MoRIIODecodeHosts, "moriio-decode-hosts", opts.MoRIIODecodeHosts,
		"Wide-EP: comma-separated decode-side pod IPs, emitted as the prefill leg's "+
			"remote_hosts. Pair with --moriio-dp-size-local.")

	fs.StringSliceVar(&opts.enableTLS, enableTLS, opts.enableTLS, "stages to enable TLS for. Supported: "+supportedTLSStageNamesStr+". Can be specified multiple times or as comma-separated values.")
	fs.StringSliceVar(&opts.tlsInsecureSkipVerify, tlsInsecureSkipVerify, opts.tlsInsecureSkipVerify, "stages to skip TLS verification for. Supported: "+supportedTLSStageNamesStr+". Can be specified multiple times or as comma-separated values.")
	fs.StringVar(&opts.inferencePool, inferencePool, opts.inferencePool, "InferencePool in namespace/name or name format (e.g., default/my-pool or my-pool). A single name implies the 'default' namespace. Can also use INFERENCE_POOL env var.")

	// Deprecated flags - kept for backward compatibility
	fs.StringVar(&opts.connector, connector, opts.connector, "Deprecated: use --kv-connector instead. The P/D connector being used. Supported: "+supportedKVConnectorNamesStr)
	_ = fs.MarkDeprecated(connector, "use --kv-connector instead")

	fs.BoolVar(&opts.prefillerUseTLS, prefillerUseTLS, opts.prefillerUseTLS, "Deprecated: use --enable-tls=prefiller instead. Whether to use TLS when sending requests to prefillers.")
	_ = fs.MarkDeprecated(prefillerUseTLS, "use --enable-tls=prefiller instead")
	fs.BoolVar(&opts.decoderUseTLS, "decoder-use-tls", opts.decoderUseTLS, "Deprecated: use --enable-tls=decoder instead. Whether to use TLS when sending requests to the decoder.")
	_ = fs.MarkDeprecated(decoderUseTLS, "use --enable-tls=decoder instead")
	fs.BoolVar(&opts.prefillerInsecureSkipVerify, prefillerTLSInsecureSkipVerify, opts.prefillerInsecureSkipVerify, "Deprecated: use --tls-insecure-skip-verify=prefiller instead. Skip TLS verification for requests to prefiller.")
	_ = fs.MarkDeprecated(prefillerTLSInsecureSkipVerify, "use --tls-insecure-skip-verify=prefiller instead")
	fs.BoolVar(&opts.decoderInsecureSkipVerify, decoderTLSInsecureSkipVerify, opts.decoderInsecureSkipVerify, "Deprecated: use --tls-insecure-skip-verify=decoder instead. Skip TLS verification for requests to decoder.")
	_ = fs.MarkDeprecated(decoderTLSInsecureSkipVerify, "use --tls-insecure-skip-verify=decoder instead")

	fs.IntVar(&opts.MaxIdleConnsPerHost, "max-idle-conns-per-host", opts.MaxIdleConnsPerHost, "max idle keep-alive connections per host for reverse proxy transports; set to at least the expected concurrency")
	fs.IntVar(&opts.PrefillMaxRetries, prefillMaxRetries, opts.PrefillMaxRetries, "max retry attempts when a prefill request fails with a 5xx error; 0 means no retries (default)")
	fs.DurationVar(&opts.PrefillRetryBackoff, prefillRetryBackoff, opts.PrefillRetryBackoff, "delay between prefill retry attempts")
	fs.StringVar(&opts.inlineConfiguration, inlineConfiguration, "", "Sidecar configuration in YAML provided as inline specification. Example `--configuration={port: 8085, vllm-port: 8203}. Inline configuration and file configuration are mutually exclusive.`")
	fs.StringVar(&opts.fileConfiguration, configurationFile, "", "Path to file which contains sidecar configuration in YAML. Example `--configuration-file=/etc/config/sidecar-config.yaml`. Inline configuration and file configuration are mutually exclusive.")
}

// validateStages checks if all stages in the slice are valid according to the supportedStages map
func validateStages(stages []string, supportedStages map[string]struct{}, flagName string) error {
	for _, stage := range stages {
		if _, ok := supportedStages[stage]; !ok {
			return fmt.Errorf("%s stages must be one of: %s", flagName, supportedTLSStageNamesStr)
		}
	}
	return nil
}

// Complete performs post-processing of parsed command-line arguments.
// It extracts YAML configuration (if provided), handles migration from deprecated flags,
// parses the InferencePool field, computes boolean TLS fields, and builds Config.DecoderURL.
// After Complete(), opts.Config is fully populated.
func (opts *Options) Complete() error {
	if err := opts.extractYAMLConfiguration(); err != nil {
		return err
	}

	// Migrate deprecated connector flag to KVConnector
	if opts.connector != "" && opts.KVConnector == "" {
		opts.KVConnector = opts.connector
	}

	// Parse inferencePool field (namespace/name or just name) into Config.
	if opts.inferencePool != "" {
		parts := strings.SplitN(opts.inferencePool, "/", 2)
		if len(parts) == 2 {
			opts.InferencePoolNamespace = parts[0]
			opts.InferencePoolName = parts[1]
		} else {
			opts.InferencePoolNamespace = "default"
			opts.InferencePoolName = parts[0]
		}
	}

	// Migrate deprecated boolean TLS flags into enableTLS/tlsInsecureSkipVerify slices
	if opts.prefillerUseTLS && !slices.Contains(opts.enableTLS, prefillStage) {
		opts.enableTLS = append(opts.enableTLS, prefillStage)
	}
	if opts.decoderUseTLS && !slices.Contains(opts.enableTLS, decodeStage) {
		opts.enableTLS = append(opts.enableTLS, decodeStage)
	}
	if opts.prefillerInsecureSkipVerify && !slices.Contains(opts.tlsInsecureSkipVerify, prefillStage) {
		opts.tlsInsecureSkipVerify = append(opts.tlsInsecureSkipVerify, prefillStage)
	}
	if opts.decoderInsecureSkipVerify && !slices.Contains(opts.tlsInsecureSkipVerify, decodeStage) {
		opts.tlsInsecureSkipVerify = append(opts.tlsInsecureSkipVerify, decodeStage)
	}

	// Compute Config TLS fields from stage slices
	opts.UseTLSForPrefiller = slices.Contains(opts.enableTLS, prefillStage)
	opts.UseTLSForDecoder = slices.Contains(opts.enableTLS, decodeStage)
	opts.UseTLSForEncoder = slices.Contains(opts.enableTLS, encodeStage)
	opts.InsecureSkipVerifyForPrefiller = slices.Contains(opts.tlsInsecureSkipVerify, prefillStage)
	opts.InsecureSkipVerifyForEncoder = slices.Contains(opts.tlsInsecureSkipVerify, encodeStage)
	opts.InsecureSkipVerifyForDecoder = slices.Contains(opts.tlsInsecureSkipVerify, decodeStage)

	// Compute Config.DecoderURL from vllmPort and decoder TLS setting
	scheme := "http"
	if opts.UseTLSForDecoder {
		scheme = schemeHTTPS
	}
	var err error
	opts.DecoderURL, err = url.Parse(scheme + "://localhost:" + opts.vllmPort)
	if err != nil {
		return fmt.Errorf("failed to parse target URL: %w", err)
	}

	// MoRI-IO feature gate: when MoRIIOFeatureEnabled is false, reject ANY
	// --moriio-* flag at a non-default value. This keeps the feature fully
	// dormant until CI and production validation is complete. Even flags like
	// --moriio-dp-size > 1 can affect routing behavior (X-Data-Parallel-Rank
	// header) without WRITE mode, so all MoRI-IO flags must be blocked.
	if !MoRIIOFeatureEnabled && opts.hasMoRIIOFlagsSet() {
		return errors.New(
			"MoRI-IO WRITE-mode and Wide-EP features are not yet enabled in this release. " +
				"The --moriio-* flags are reserved for a future release candidate. " +
				"Please remove all --moriio-* flags, or wait for the " +
				"official feature release. See pkg/sidecar/proxy/MORIIO_README.md for details")
	}

	// WRITE mode without a routable decode pod IP makes prefill handshake with
	// itself and hang silently, so fail fast.
	if opts.MoRIIOWriteMode && opts.MoRIIODecodePodIP == "" {
		return errors.New(
			"--moriio-write-mode requires --moriio-local-pod-ip (or the POD_IP " +
				"env var) set to decode's routable pod IP")
	}

	// Concurrent dispatch needs the fields only WRITE mode synthesises.
	if opts.MoRIIOParallelDispatch && !opts.MoRIIOWriteMode {
		return errors.New("--moriio-parallel-dispatch requires --moriio-write-mode")
	}

	// Both legs share the same dp_size / dp_size_local contract; only the host
	// list differs.
	if err := validateWideEPHosts(
		"--moriio-remote-hosts", opts.MoRIIORemoteHosts,
		opts.MoRIIODPSize, opts.MoRIIODPSizeLocal); err != nil {
		return err
	}
	return validateWideEPHosts(
		"--moriio-decode-hosts", opts.MoRIIODecodeHosts,
		opts.MoRIIODPSize, opts.MoRIIODPSizeLocal)
}

// hasMoRIIOFlagsSet returns true if any --moriio-* flag is set to a non-default
// value. Used by the dormant feature gate to reject any MoRI-IO configuration
// when the feature is not yet enabled.
func (opts *Options) hasMoRIIOFlagsSet() bool {
	// Behavioral flags that affect routing or wire shape
	if opts.MoRIIOWriteMode || opts.MoRIIOParallelDispatch {
		return true
	}
	// DP-size > 1 triggers X-Data-Parallel-Rank header even without WRITE mode
	if opts.MoRIIODPSize > 1 {
		return true
	}
	// Wide-EP multi-pod configuration
	if len(opts.MoRIIORemoteHosts) > 0 || len(opts.MoRIIODecodeHosts) > 0 {
		return true
	}
	if opts.MoRIIODPSizeLocal > 0 {
		return true
	}
	return false
}

// validateWideEPHosts checks the multi-pod fan-out invariants for one host
// list: dpLocal must divide dpSize and the host count must equal the pod count
// (dpSize / dpLocal), since vLLM indexes hosts[dp_rank / dpLocal]. A misconfig
// otherwise surfaces only as a cross-pod handshake that hangs, so it fails fast.
// Lists of 0 or 1 host are the single-pod case and pass unchecked.
func validateWideEPHosts(flag string, hosts []string, dpSize, dpLocal int) error {
	if len(hosts) <= 1 {
		return nil
	}
	if dpLocal <= 0 {
		return fmt.Errorf(
			"%s has %d entries but --moriio-dp-size-local is %d; "+
				"Wide-EP multi-pod requires dp-size-local > 0 so vLLM "+
				"can compute pod_idx = dp_rank / dp_size_local",
			flag, len(hosts), dpLocal)
	}
	if dpSize%dpLocal != 0 {
		return fmt.Errorf(
			"--moriio-dp-size (%d) must be divisible by "+
				"--moriio-dp-size-local (%d) for Wide-EP multi-pod (%s)",
			dpSize, dpLocal, flag)
	}
	if dpSize/dpLocal != len(hosts) {
		return fmt.Errorf(
			"%s has %d entries but dp-size/dp-size-local = %d; the "+
				"count of hosts must match the number of pods so the "+
				"per-rank pod_idx mapping covers every DP rank",
			flag, len(hosts), dpSize/dpLocal)
	}
	return nil
}

// Validate checks the Options for invalid or conflicting values.
// Complete must be called before Validate.
func (opts *Options) Validate() error {
	// Validate KV connector
	if _, ok := supportedKVConnectors[opts.KVConnector]; !ok {
		return fmt.Errorf("--kv-connector must be one of: %s", supportedKVConnectorNamesStr)
	}

	// Validate EC connector if provided
	if opts.ECConnector != "" {
		if _, ok := supportedECConnectors[opts.ECConnector]; !ok {
			return fmt.Errorf("--ec-connector must be one of: %s", supportedECConnectorNamesStr)
		}
	}

	// Validate deprecated connector flag
	if opts.connector != "" && opts.connector != opts.KVConnector {
		if _, ok := supportedKVConnectors[opts.connector]; !ok {
			return fmt.Errorf("--connector must be one of: %s", supportedKVConnectorNamesStr)
		}
	}

	// Validate TLS stages
	if err := validateStages(opts.enableTLS, supportedTLSStages, "--enable-tls"); err != nil {
		return err
	}
	if err := validateStages(opts.tlsInsecureSkipVerify, supportedTLSStages, "--tls-insecure-skip-verify"); err != nil {
		return err
	}

	// Validate inferencePool format if provided
	if opts.inferencePool != "" {
		if strings.Count(opts.inferencePool, "/") > 1 {
			return errors.New("--inference-pool must be in format 'namespace/name' or 'name', not multiple slashes")
		}
		parts := strings.Split(opts.inferencePool, "/")
		for _, part := range parts {
			if part == "" {
				return errors.New("--inference-pool cannot have empty namespace or name")
			}
		}
	}

	// Validate prefill retry configuration
	if opts.PrefillMaxRetries > 0 && opts.PrefillRetryBackoff <= 0 {
		return fmt.Errorf("--prefill-retry-backoff must be positive when --prefill-max-retries > 0, got %v", opts.PrefillRetryBackoff)
	}

	// Validate chunked decode
	if opts.DecodeChunkSize < 0 {
		return fmt.Errorf("--decode-chunk-size must be a non-negative integer (0 disables chunked decode), got %d", opts.DecodeChunkSize)
	}

	// Validate mooncake bootstrap port
	if opts.MooncakeBootstrapPort < 1 || opts.MooncakeBootstrapPort > 65535 {
		return fmt.Errorf("--mooncake-bootstrap-port must be between 1 and 65535, got %d", opts.MooncakeBootstrapPort)
	}

	// Validate P2P connector port
	if opts.P2PConnectorPort < 1 || opts.P2PConnectorPort > 65535 {
		return fmt.Errorf("--p2p-connector-port must be between 1 and 65535, got %d", opts.P2PConnectorPort)
	}

	// Validate SSRF protection requirements
	if opts.EnableSSRFProtection {
		if opts.InferencePoolNamespace == "" || opts.InferencePoolName == "" {
			return errors.New("--inference-pool flag or INFERENCE_POOL environment variable is required when --enable-ssrf-protection is true")
		}
	}

	return nil
}

// customLevelEncoder maps negative Zap levels to human-readable names that
// match the project's verbosity constants (VERBOSE=3, DEBUG=4, TRACE=5).
// Without this, controller-runtime's zap bridge emits all V(n) calls as
// "debug" in JSON output, which is misleading for V(1)–V(3) (verbose info).
func customLevelEncoder(l zapcore.Level, enc zapcore.PrimitiveArrayEncoder) {
	if l >= 0 {
		zapcore.LowercaseLevelEncoder(l, enc)
		return
	}
	switch l {
	case zapcore.Level(-1 * logutil.DEBUG): // V(4) → "debug"
		enc.AppendString("debug")
	case zapcore.Level(-1 * logutil.TRACE): // V(5) → "trace"
		enc.AppendString("trace")
	default:
		if l >= zapcore.Level(-1*logutil.VERBOSE) { // V(1)–V(3) → "info"
			enc.AppendString("info")
		} else { // V(6+) → "trace"
			enc.AppendString("trace")
		}
	}
}

// NewLogger returns a logger configured from the Options logging flags,
// with a custom level encoder that maps verbosity levels to their semantic
// names instead of always rendering V(n) as "debug".
func (opts *Options) NewLogger() logr.Logger {
	config := uberzap.NewProductionEncoderConfig()
	config.EncodeLevel = customLevelEncoder
	return zap.New(
		zap.UseFlagOptions(&opts.loggingOptions),
		zap.Encoder(zapcore.NewJSONEncoder(config)),
	)
}

// extractYAMLConfiguration extracts sidecar configuration (if provided)
// from `--configuration` and `--configuration-file` parameters
func (opts *Options) extractYAMLConfiguration() error {
	var yamlConfiguration yamlConfiguration
	var yamlData []byte
	var err error

	switch {
	case opts.inlineConfiguration != "" && opts.fileConfiguration != "":
		return fmt.Errorf("flags --%s and --%s are mutually exclusive", inlineConfiguration, configurationFile)

	case opts.inlineConfiguration != "":
		yamlData = []byte(opts.inlineConfiguration)

	case opts.fileConfiguration != "":
		yamlData, err = os.ReadFile(opts.fileConfiguration)
		if err != nil {
			return fmt.Errorf("failed to read sidecar configuration from file: %w", err)
		}
	}

	if yamlData == nil {
		return nil
	}

	// fail on unknown YAML fields
	if err := yaml.UnmarshalStrict(yamlData, &yamlConfiguration); err != nil {
		return fmt.Errorf("failed to unmarshal sidecar configuration: %w", err)
	}

	opts.mergeYAMLConfiguration(yamlConfiguration)
	return nil
}

// mergeYAMLConfiguration merges provided yamlConfiguration into Options struct,
// respecting precedence of command-line flags (i.e., YAML values are only applied if corresponding flag was not explicitly set by user)
func (opts *Options) mergeYAMLConfiguration(cfg yamlConfiguration) {
	if cfg.Port != 0 && !opts.isFlagSet(port) {
		opts.Port = strconv.Itoa(cfg.Port)
	}
	if cfg.VLLMPort != 0 && !opts.isFlagSet(vllmPort) {
		opts.vllmPort = strconv.Itoa(cfg.VLLMPort)
	}
	if cfg.MooncakeBootstrapPort != 0 && !opts.isFlagSet(mooncakeBootstrapPortFlag) {
		opts.MooncakeBootstrapPort = cfg.MooncakeBootstrapPort
	}
	if cfg.P2PConnectorPort != 0 && !opts.isFlagSet(p2pConnectorPortFlag) {
		opts.P2PConnectorPort = cfg.P2PConnectorPort
	}
	if cfg.DataParallelSize != 0 && !opts.isFlagSet(dataParallelSize) {
		opts.DataParallelSize = cfg.DataParallelSize
	}
	if cfg.MaxIdleConnsPerHost != 0 && !opts.isFlagSet(maxIdleConnsPerHost) {
		opts.MaxIdleConnsPerHost = cfg.MaxIdleConnsPerHost
	}

	if cfg.KVConnector != "" && !opts.isFlagSet(kvConnector) {
		opts.KVConnector = cfg.KVConnector
	}
	if cfg.Connector != "" && !opts.isFlagSet(connector) {
		opts.connector = cfg.Connector
	}
	if cfg.ECConnector != "" && !opts.isFlagSet(ecConnector) {
		opts.ECConnector = cfg.ECConnector
	}

	if cfg.EnableSSRFProtection != nil && !opts.isFlagSet(enableSSRFProtection) {
		opts.EnableSSRFProtection = *cfg.EnableSSRFProtection
	}
	if cfg.EnablePrefillerSampling != nil && !opts.isFlagSet(enablePrefillerSampling) {
		opts.EnablePrefillerSampling = *cfg.EnablePrefillerSampling
	}

	if cfg.SecureServing != nil && !opts.isFlagSet(secureServing) {
		opts.SecureServing = *cfg.SecureServing
	}
	if cfg.CertPath != "" && !opts.isFlagSet(certPath) {
		opts.CertPath = cfg.CertPath
	}

	if len(cfg.EnableTLS) > 0 && !opts.isFlagSet(enableTLS) {
		opts.enableTLS = cfg.EnableTLS
	}
	if len(cfg.TLSInsecureSkipVerify) > 0 && !opts.isFlagSet(tlsInsecureSkipVerify) {
		opts.tlsInsecureSkipVerify = cfg.TLSInsecureSkipVerify
	}

	// update prefiller/decoder TLS settings from deprecated YAML fields if corresponding new fields are not set by user via flags
	// (i.e., prefillerUseTLS only applies if --enable-tls and --prefiller-use-tls are not set,
	// and decoderUseTLS only applies if --enable-tls and --decoder-use-tls are not set)
	if cfg.PrefillerUseTLS != nil && !opts.isFlagSet(enableTLS) && !opts.isFlagSet(prefillerUseTLS) {
		opts.prefillerUseTLS = *cfg.PrefillerUseTLS
	}
	if cfg.DecoderUseTLS != nil && !opts.isFlagSet(enableTLS) && !opts.isFlagSet(decoderUseTLS) {
		opts.decoderUseTLS = *cfg.DecoderUseTLS
	}

	// update prefiller/decoder TLS insecure skip verify settings from deprecated YAML fields if corresponding new fields are not set by user via flags
	// (i.e., prefillerTLSInsecureSkipVerify only applies if --tls-insecure-skip-verify and --prefiller-tls-insecure-skip-verify are not set,
	// and decoderTLSInsecureSkipVerify only applies if --tls-insecure-skip-verify and --decoder-tls-insecure-skip-verify are not set)
	if cfg.PrefillerTLSInsecureSkipVerify != nil && !opts.isFlagSet(prefillerTLSInsecureSkipVerify) && !opts.isFlagSet(tlsInsecureSkipVerify) {
		opts.prefillerInsecureSkipVerify = *cfg.PrefillerTLSInsecureSkipVerify
	}
	if cfg.DecoderTLSInsecureSkipVerify != nil && !opts.isFlagSet(decoderTLSInsecureSkipVerify) && !opts.isFlagSet(tlsInsecureSkipVerify) {
		opts.decoderInsecureSkipVerify = *cfg.DecoderTLSInsecureSkipVerify
	}

	if cfg.InferencePool != "" && !opts.isFlagSet(inferencePool) {
		opts.inferencePool = cfg.InferencePool
	}
	if cfg.PoolGroup != "" && !opts.isFlagSet(poolGroup) {
		opts.PoolGroup = cfg.PoolGroup
	}
	if cfg.PrefillMaxRetries != nil && !opts.isFlagSet(prefillMaxRetries) {
		opts.PrefillMaxRetries = *cfg.PrefillMaxRetries
	}
	if cfg.PrefillRetryBackoff != "" && !opts.isFlagSet(prefillRetryBackoff) {
		d, err := time.ParseDuration(cfg.PrefillRetryBackoff)
		if err != nil {
			fmt.Fprintf(os.Stderr, "WARNING: ignoring invalid %s value %q: %v; using default %v\n",
				prefillRetryBackoff, cfg.PrefillRetryBackoff, err, opts.PrefillRetryBackoff)
		} else {
			opts.PrefillRetryBackoff = d
		}
	}
	if cfg.DecodeChunkSize != 0 && !opts.isFlagSet(decodeChunkSize) {
		opts.DecodeChunkSize = cfg.DecodeChunkSize
	}
	if cfg.Tracing != nil && !opts.isFlagSet(tracingFlag) {
		opts.Tracing = *cfg.Tracing
	}
}

// isFlagSet returns true if flag was set by user
func (opts *Options) isFlagSet(parameter string) bool {
	if opts.pflagSet != nil {
		flag := opts.pflagSet.Lookup(parameter)
		if flag != nil && flag.Changed {
			return true
		}
	}
	return false
}
