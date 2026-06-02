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
	// Flags
	port                    = "port"
	vllmPort                = "vllm-port"
	dataParallelSize        = "data-parallel-size"
	kvConnector             = "kv-connector"
	ecConnector             = "ec-connector"
	enableSSRFProtection    = "enable-ssrf-protection"
	enablePrefillerSampling = "enable-prefiller-sampling"
	enableTLS               = "enable-tls"
	tlsInsecureSkipVerify   = "tls-insecure-skip-verify"
	secureServing           = "secure-proxy"
	certPath                = "cert-path"
	inferencePool           = "inference-pool"
	poolGroup               = "pool-group"
	maxIdleConnsPerHost     = "max-idle-conns-per-host"
	decodeChunkSize         = "decode-chunk-size"
	inlineConfiguration     = "configuration"
	configurationFile       = "configuration-file"

	// Deprecated flags
	connector                      = "connector"
	prefillerUseTLS                = "prefiller-use-tls"
	decoderUseTLS                  = "decoder-use-tls"
	encoderUseTLS                  = "UseTLSForEncoder"
	prefillerTLSInsecureSkipVerify = "prefiller-tls-insecure-skip-verify"
	decoderTLSInsecureSkipVerify   = "decoder-tls-insecure-skip-verify"
	inferencePoolNamespace         = "inference-pool-namespace"
	inferencePoolName              = "inference-pool-name"

	// Environment variables
	envInferencePool           = "INFERENCE_POOL"
	envInferencePoolNamespace  = "INFERENCE_POOL_NAMESPACE"
	envInferencePoolName       = "INFERENCE_POOL_NAME"
	envEnablePrefillerSampling = "ENABLE_PREFILLER_SAMPLING"

	// Defaults
	defaultPort             = "8000"
	defaultVLLMPort         = "8001"
	defaultDataParallelSize = 1

	// TLS stages
	prefillStage = "prefiller"
	decodeStage  = "decoder"
	encodeStage  = "encoder"
)

// yamlConfiguration represents structure of YAML configuration for sidecar proxy
type yamlConfiguration struct {
	Port                           int      `json:"port,omitempty"`
	VLLMPort                       int      `json:"vllm-port,omitempty"`
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
	DecodeChunkSize                int      `json:"decode-chunk-size,omitempty"`
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
	}

	// supportedECConnectors defines all valid E/P EC connector types
	supportedECConnectors = map[string]struct{}{
		ECExampleConnector: {},
	}

	// supportedTLSStages defines all valid stages for TLS configuration
	supportedTLSStages = map[string]struct{}{
		prefillStage: {},
		decodeStage:  {},
		encodeStage:  {},
	}

	supportedKVConnectorNamesStr = strings.Join([]string{KVConnectorNIXLV2, KVConnectorSharedStorage, KVConnectorSGLang}, ", ")
	supportedECConnectorNamesStr = strings.Join([]string{ECExampleConnector}, ", ")
	supportedTLSStageNamesStr    = strings.Join([]string{prefillStage, decodeStage, encodeStage}, ", ")
)

// NewOptions returns a new Options struct initialized with default values.
func NewOptions() *Options {
	enablePrefillerSampling := false
	if val, err := strconv.ParseBool(os.Getenv(envEnablePrefillerSampling)); err == nil {
		enablePrefillerSampling = val
	}

	return &Options{
		Config: Config{
			Port:                    defaultPort,
			DataParallelSize:        defaultDataParallelSize,
			SecureServing:           true,
			EnablePrefillerSampling: enablePrefillerSampling,
			MaxIdleConnsPerHost:     defaultMaxIdleConnsPerHost,
			PoolGroup:               routing.InferencePoolAPIGroup,
			InferencePoolNamespace:  os.Getenv(envInferencePoolNamespace),
			InferencePoolName:       os.Getenv(envInferencePoolName),
			DecodeChunkSize:         0,
			// MoRI-IO defaults: OFF by default to preserve existing NIXLv2 behaviour.
			// 61005 matches MoRIIOConstants.DEFAULT_NOTIFY_PORT in vLLM.
			MoRIIOWriteMode:           false,
			MoRIIODecodeNotifyPort:    61005,
			MoRIIODecodeDPRank:        0,
			MoRIIODecodeHandshakePort: 6301,
			// MoRIIODecodePodIP defaults to the POD_IP env var which Helm
			// templates from K8s downward API.  Empty here preserves NIXLv2
			// behaviour for non-MoRI deployments.
			MoRIIODecodePodIP: os.Getenv("POD_IP"),
			// Concurrent-dispatch defaults.  OFF by default so existing
			// strictly-serial deployments stay bit-identical.  When
			// MoRIIOParallelDispatch is true the proxy synthesises the
			// decode-leg kv_transfer_params from these MoRIIOPrefill*
			// values + prefillPodHostPort and fires prefill/decode in
			// parallel goroutines.
			MoRIIOParallelDispatch:     false,
			MoRIIOPrefillHandshakePort: 6301,
			MoRIIOPrefillNotifyPort:    61005,
			MoRIIOPrefillDPRank:        0,
			MoRIIOTPSize:               1,
			// Single-DP default keeps remote_dp_size: 1 in
			// runNIXLProtocolV2WriteParallel.  Set to the engine's DP
			// world size for Wide-EP.
			MoRIIODPSize: 1,
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
	fs.BoolVar(&opts.SecureServing, secureServing, opts.SecureServing, "Enables secure proxy. Defaults to true.")
	fs.StringVar(&opts.CertPath, certPath, opts.CertPath, "The path to the certificate for secure proxy. The certificate and private key files are assumed to be named tls.crt and tls.key, respectively. If not set, and secureProxy is enabled, then a self-signed certificate is used (for testing).")
	fs.BoolVar(&opts.EnableSSRFProtection, enableSSRFProtection, opts.EnableSSRFProtection, "enable SSRF protection using InferencePool allowlisting")
	fs.BoolVar(&opts.EnablePrefillerSampling, enablePrefillerSampling, opts.EnablePrefillerSampling, "if true, the target prefill instance will be selected randomly from among the provided prefill host values")
	fs.StringVar(&opts.PoolGroup, poolGroup, opts.PoolGroup, "group of the InferencePool this Endpoint Picker is associated with.")
	fs.IntVar(&opts.DecodeChunkSize, decodeChunkSize, opts.DecodeChunkSize, "enables chunked decode mode when > 0; value is the token budget per chunk. For best performance should be a multiple of the block size.")

	// MoRI-IO WRITE-mode flags.  Only meaningful with --kv-connector=nixlv2 when
	// the vLLM engines run MoRIIOConnector with VLLM_MORIIO_CONNECTOR_READ_MODE=0.
	// Otherwise these are silent no-ops.
	fs.BoolVar(&opts.MoRIIOWriteMode, "moriio-write-mode", opts.MoRIIOWriteMode,
		"Enable MoRI-IO WRITE-mode passthrough of remote_host, remote_notify_port and "+
			"transfer_id in kv_transfer_params to the prefill engine.  Requires "+
			"--kv-connector=nixlv2 with vLLM engines running MoRI-IO in WRITE mode "+
			"(VLLM_MORIIO_CONNECTOR_READ_MODE unset or 0).  Default false.")
	fs.IntVar(&opts.MoRIIODecodeNotifyPort, "moriio-decode-notify-port", opts.MoRIIODecodeNotifyPort,
		"Base MoRI-IO notify port on the decode pod (matches "+
			"MoRIIOConstants.DEFAULT_NOTIFY_PORT in vLLM, 61005).")
	fs.IntVar(&opts.MoRIIODecodeDPRank, "moriio-decode-dp-rank", opts.MoRIIODecodeDPRank,
		"Data-parallel rank of the decode engine, used by the prefill connector to "+
			"compute per-TP target_port offsets in send_notify_block.")
	fs.StringVar(&opts.MoRIIODecodePodIP, "moriio-local-pod-ip", opts.MoRIIODecodePodIP,
		"Decode pod's routable IP address as seen from other pods.  Used in "+
			"WRITE mode as kv_transfer_params remote_host on the prefill "+
			"leg so the prefill engine can handshake with decode for RDMA "+
			"WRITE.  Defaults to the POD_IP environment variable (templated "+
			"by Helm from K8s downward API).  Required when --moriio-write-mode "+
			"is set; otherwise prefill handshakes with itself at localhost:6301.")
	fs.IntVar(&opts.MoRIIODecodeHandshakePort, "moriio-decode-handshake-port", opts.MoRIIODecodeHandshakePort,
		"Base MoRI-IO handshake port on the decode pod (matches "+
			"MoRIIOConstants.DEFAULT_HANDSHAKE_PORT in vLLM, 6301).")

	// Concurrent-dispatch flags.  These let the proxy synthesise the
	// decode-leg kv_transfer_params without waiting for the prefill
	// response, so prefill+decode HTTP dispatch overlap with the decode
	// engine's block allocation.  See Config.MoRIIOParallelDispatch for the
	// TTFT win this enables and the failure mode if --moriio-write-mode is
	// not also set.
	fs.BoolVar(&opts.MoRIIOParallelDispatch, "moriio-parallel-dispatch", opts.MoRIIOParallelDispatch,
		"Enable concurrent prefill/decode dispatch.  Requires "+
			"--moriio-write-mode.  Trades the strictly-serial flow for "+
			"parallel goroutines where decode's kv_transfer_params is synthesised "+
			"from the --moriio-prefill-* flags + the EPP-supplied prefill "+
			"host:port instead of being read from the prefill response.  Default false.")
	fs.IntVar(&opts.MoRIIOPrefillHandshakePort, "moriio-prefill-handshake-port", opts.MoRIIOPrefillHandshakePort,
		"Prefill pod's base MoRI-IO handshake port (matches "+
			"MoRIIOConstants.DEFAULT_HANDSHAKE_PORT in vLLM, 6301).  Used in "+
			"concurrent-dispatch mode to populate the decode-leg kv_transfer_params remote_handshake_port "+
			"so decode can send notify-block to prefill in parallel with prefill's "+
			"forward pass.")
	fs.IntVar(&opts.MoRIIOPrefillNotifyPort, "moriio-prefill-notify-port", opts.MoRIIOPrefillNotifyPort,
		"Prefill pod's base MoRI-IO notify port (matches "+
			"MoRIIOConstants.DEFAULT_NOTIFY_PORT in vLLM, 61005).")
	fs.IntVar(&opts.MoRIIOPrefillDPRank, "moriio-prefill-dp-rank", opts.MoRIIOPrefillDPRank,
		"Data-parallel rank of the prefill engine, used by the decode connector "+
			"to compute per-TP target_port offsets in send_notify_block.")
	fs.IntVar(&opts.MoRIIOTPSize, "moriio-tp-size", opts.MoRIIOTPSize,
		"Tensor-parallel size of the prefill/decode engines.  Echoed into the "+
			"decode-leg kv_transfer_params[tp_size] when --moriio-parallel-dispatch "+
			"is set, so vLLM's MoRIIOConnector can size its per-TP notify-port stride "+
			"without waiting for prefill's response.")
	fs.IntVar(&opts.MoRIIODPSize, "moriio-dp-size", opts.MoRIIODPSize,
		"Data-parallel world size of the prefill/decode engines.  Emitted "+
			"as kv_transfer_params[remote_dp_size] on BOTH legs in "+
			"concurrent-dispatch (WriteParallel) mode.  Required for "+
			"Wide-EP deployments (TP=1, DP>1) where vLLM's "+
			"MoRIIOConnector iterates a per-DP-rank handshake loop "+
			"sized by this value; the default of 1 keeps the wire "+
			"shape identical for single-DP deployments.")

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

	fs.StringVar(&opts.InferencePoolNamespace, inferencePoolNamespace, opts.InferencePoolNamespace, "Deprecated: use --inference-pool instead. The Kubernetes namespace for the InferencePool (defaults to INFERENCE_POOL_NAMESPACE env var)")
	_ = fs.MarkDeprecated(inferencePoolNamespace, "use --inference-pool instead")
	fs.StringVar(&opts.InferencePoolName, inferencePoolName, opts.InferencePoolName, "Deprecated: use --inference-pool instead. The specific InferencePool name (defaults to INFERENCE_POOL_NAME env var)")
	_ = fs.MarkDeprecated(inferencePoolName, "use --inference-pool instead")
	fs.IntVar(&opts.MaxIdleConnsPerHost, "max-idle-conns-per-host", opts.MaxIdleConnsPerHost, "max idle keep-alive connections per host for reverse proxy transports; set to at least the expected concurrency")
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

	// Parse inferencePool field (namespace/name or just name), overriding deprecated separate flags
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

	// Surface misconfiguration up front: WRITE mode without a routable
	// decode pod IP causes prefill to handshake with itself (localhost),
	// which hangs silently with no error logs (see proxy.go MoRIIODecodePodIP
	// comment for the full failure mode).
	if opts.MoRIIOWriteMode && opts.MoRIIODecodePodIP == "" {
		return errors.New(
			"--moriio-write-mode requires --moriio-local-pod-ip (or " +
				"the POD_IP env var) to be set to decode's routable pod IP; " +
				"a falsy value would make prefill handshake with itself")
	}

	// Concurrent dispatch only makes sense in WRITE mode -- the READ-mode
	// path needs remote_block_ids and remote_engine_id from the prefill
	// response and cannot proceed in parallel.  Surface the
	// misconfiguration up front rather than silently fall back to the
	// strictly-serial path.
	if opts.MoRIIOParallelDispatch && !opts.MoRIIOWriteMode {
		return errors.New(
			"--moriio-parallel-dispatch requires --moriio-write-mode " +
				"(concurrent dispatch is a WRITE-mode-only optimisation; the " +
				"READ-mode path depends on prefill-supplied kv_transfer_params " +
				"fields that the sidecar cannot synthesise)")
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

	// Validate chunked decode
	if opts.DecodeChunkSize < 0 {
		return fmt.Errorf("--decode-chunk-size must be a non-negative integer (0 disables chunked decode), got %d", opts.DecodeChunkSize)
	}

	// Validate SSRF protection requirements
	if opts.EnableSSRFProtection {
		if opts.InferencePoolNamespace == "" {
			return errors.New("--inference-pool, --inference-pool-namespace, INFERENCE_POOL, or INFERENCE_POOL_NAMESPACE environment variable is required when --enable-ssrf-protection is true")
		}
		if opts.InferencePoolName == "" {
			return errors.New("--inference-pool, --inference-pool-name, INFERENCE_POOL, or INFERENCE_POOL_NAME environment variable is required when --enable-ssrf-protection is true")
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
	if cfg.DecodeChunkSize != 0 && !opts.isFlagSet(decodeChunkSize) {
		opts.DecodeChunkSize = cfg.DecodeChunkSize
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
