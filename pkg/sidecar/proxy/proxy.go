/*
Copyright 2025 The llm-d Authors.

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
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"math/rand/v2"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/go-logr/logr"
	lru "github.com/hashicorp/golang-lru/v2"
	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
	"golang.org/x/sync/errgroup"
	"sigs.k8s.io/controller-runtime/pkg/log"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
)

const (
	schemeHTTPS = "https"

	defaultMaxIdleConnsPerHost = 1024

	requestHeaderRequestID = reqcommon.RequestIDHeaderKey

	requestFieldKVTransferParams     = reqcommon.FieldKVTransferParams
	requestFieldECTransferParams     = reqcommon.FieldECTransferParams
	requestFieldMaxTokens            = reqcommon.FieldMaxTokens
	requestFieldMaxCompletionTokens  = reqcommon.FieldMaxCompletionTokens
	requestFieldMaxOutputTokens      = reqcommon.FieldMaxOutputTokens
	requestFieldMinTokens            = reqcommon.FieldMinTokens
	requestFieldSamplingParams       = reqcommon.FieldSamplingParams
	requestFieldDoRemotePrefill      = reqcommon.FieldDoRemotePrefill
	requestFieldDoRemoteDecode       = reqcommon.FieldDoRemoteDecode
	requestFieldRemoteBlockIDs       = reqcommon.FieldRemoteBlockIDs
	requestFieldRemoteEngineID       = reqcommon.FieldRemoteEngineID
	requestFieldRemoteHost           = reqcommon.FieldRemoteHost
	requestFieldRemotePort           = reqcommon.FieldRemotePort
	requestFieldStream               = reqcommon.FieldStream
	requestFieldStreamOptions        = reqcommon.FieldStreamOptions
	requestFieldCacheHitThreshold    = reqcommon.FieldCacheHitThreshold
	requestFieldContinueFinalMessage = reqcommon.FieldContinueFinalMessage
	requestFieldAddGenerationPrompt  = reqcommon.FieldAddGenerationPrompt

	// requestHeaderDataParallelRank pins a request to a specific vLLM
	// data-parallel rank, set on both legs of a disagg pair (see pickDPRank).
	requestHeaderDataParallelRank = "x-data-parallel-rank"

	// MoRI-IO WRITE-mode kv_transfer_params fields, populated by the sidecar
	// so the prefill engine can push KV to decode via RDMA Write.
	requestFieldRemoteNotifyPort = "remote_notify_port"
	requestFieldRemoteDPRank     = "remote_dp_rank"
	// requestFieldRemoteDPRankOverride tells the decode-side connector to use
	// the sidecar's remote_dp_rank verbatim rather than recomputing its own hash.
	requestFieldRemoteDPRankOverride = "remote_dp_rank_override"
	requestFieldRemoteHandshakePort  = "remote_handshake_port"
	requestFieldTransferID           = "transfer_id"

	responseFieldChoices      = "choices"
	responseFieldFinishReason = "finish_reason"

	finishReasonCacheThreshold = "cache_threshold"

	// SGLang bootstrap fields
	requestFieldBootstrapHost = "bootstrap_host"
	requestFieldBootstrapPort = "bootstrap_port"
	requestFieldBootstrapRoom = "bootstrap_room"
	// Mooncake transfer fields
	requestFieldRemoteBootstrapAddr = "remote_bootstrap_addr"

	// OffloadingConnector kv_transfer_params fields. The role is encoded by the
	// nesting key, named for the remote party it describes: "remote_decoder" on
	// the prefiller leg, "remote_prefiller" on the decoder leg, "remote_kv_source"
	// for a symmetric cached-prefix pull.
	requestFieldRemoteDecoder   = "remote_decoder"
	requestFieldRemotePrefiller = "remote_prefiller"
	requestFieldRemoteKVSource  = "remote_kv_source"
	requestFieldKVRequestID     = "kv_request_id"

	KVConnectorNIXLV2        = constants.KVConnectorNIXLV2
	KVConnectorSharedStorage = constants.KVConnectorSharedStorage
	KVConnectorSGLang        = constants.KVConnectorSGLang
	KVConnectorMooncake      = constants.KVConnectorMooncake
	KVConnectorOffloading    = constants.KVConnectorOffloading
	ECExampleConnector       = constants.ECExampleConnector
	ECConnectorNIXL          = constants.ECConnectorNIXL
)

// APIType represents the type of OpenAI API being used.
type APIType int

const (
	// APITypeChatCompletions is the Chat Completions API (/v1/chat/completions, /v1/completions)
	APITypeChatCompletions APIType = iota
	// APITypeResponses is the Responses API (/v1/responses)
	APITypeResponses
	// APITypeGenerate is vLLM's token-in generate API (/inference/v1/generate)
	APITypeGenerate
)

// String implements fmt.Stringer so structured logs show readable API names.
func (a APIType) String() string {
	switch a {
	case APITypeChatCompletions:
		return "chat_completions"
	case APITypeResponses:
		return "responses"
	case APITypeGenerate:
		return "generate"
	default:
		return fmt.Sprintf("APIType(%d)", int(a))
	}
}

// JSON request field names used for token limits in prefill/decode staging.
// Do not mutate these slices.
var (
	chatCompletionTokenLimitFields = []string{requestFieldMaxTokens, requestFieldMaxCompletionTokens, requestFieldMinTokens}
	responsesStyleTokenLimitFields = []string{requestFieldMaxOutputTokens}
	generateStyleTokenLimitFields  = []string{requestFieldMaxTokens, requestFieldMinTokens}
)

// tokenLimitFieldsForAPIType returns token limit field names for the given API.
// Returned slices are shared package-level vars; callers must not mutate them.
func tokenLimitFieldsForAPIType(api APIType) []string {
	switch api {
	case APITypeResponses:
		return responsesStyleTokenLimitFields
	case APITypeGenerate:
		return generateStyleTokenLimitFields
	default:
		return chatCompletionTokenLimitFields
	}
}

// Config represents the complete runtime configuration for the proxy server.
type Config struct {
	// Port is the port the sidecar is listening on.
	Port string
	// DecoderURL is the URL of the local decoder (vLLM) instance.
	DecoderURL *url.URL

	// KVConnector is the name of the KV protocol between prefiller and decoder.
	KVConnector string
	// ECConnector is the name of the EC protocol between encoder and prefiller (for EPD mode).
	// If empty, encoder stage is skipped.
	ECConnector string
	// DataParallelSize is the value passed to the vLLM server's --DATA_PARALLEL-SIZE argument.
	DataParallelSize int

	// MaxIdleConnsPerHost controls how many idle keep-alive connections are
	// maintained per host for the reverse proxy transports. Set this to at
	// least the expected concurrency level to avoid connection churn.
	MaxIdleConnsPerHost int

	// EnablePrefillerSampling configures the proxy to randomly choose from the set
	// of provided prefill hosts instead of always using the first one.
	EnablePrefillerSampling bool

	// PrefillMaxRetries is the number of additional attempts when a prefill
	// request fails with a 5xx error (e.g. connection reset → 502).
	// 0 means no retries (original behavior).
	PrefillMaxRetries int
	// PrefillRetryBackoff is the delay between prefill retry attempts.
	PrefillRetryBackoff time.Duration

	// UseTLSForPrefiller indicates whether to use TLS when sending requests to prefillers.
	UseTLSForPrefiller bool
	// UseTLSForDecoder indicates whether to use TLS when sending requests to the decoder.
	UseTLSForDecoder bool
	// UseTLSForEncoder indicates whether to use TLS when sending requests to encoders.
	UseTLSForEncoder bool
	// InsecureSkipVerifyForPrefiller configures the proxy to skip TLS verification for requests to the prefiller.
	InsecureSkipVerifyForPrefiller bool
	// InsecureSkipVerifyForEncoder configures the proxy to skip TLS verification for requests to the encoder.
	InsecureSkipVerifyForEncoder bool
	// InsecureSkipVerifyForDecoder configures the proxy to skip TLS verification for requests to the decoder.
	InsecureSkipVerifyForDecoder bool

	// SecureServing enables TLS for the sidecar server itself.
	SecureServing bool
	// CertPath is the path to TLS certificates for the sidecar server.
	CertPath string

	// MetricsPort is the port for the Prometheus /metrics endpoint. 0 (the
	// default) disables it; when > 0 the sidecar serves the shared metrics
	// registry (carrying the moriio_dns_* counters) at /metrics on that port,
	// on a separate address from the data-plane proxy port. Takes precedence
	// over the MORIIO_METRICS_ADDR env var (kept for backward compatibility).
	MetricsPort int

	// MooncakeBootstrapPort is the port used to query the Mooncake bootstrap endpoint on prefill pods.
	MooncakeBootstrapPort int

	// P2PConnectorPort is the prefiller's OffloadingConnector P2P tier listening port,
	// injected as remote_port on the decode leg so the decoder can pull KV from it.
	// With data parallelism it is the rank-0 port: rank r's tier listens on
	// P2PConnectorPort+r and the injected port is offset by the target's rank.
	// Meaningful with --kv-connector=offloading or --enable-p2p-pull.
	P2PConnectorPort int

	// EnableP2PPull declares that the OffloadingConnector P2P tier is available
	// for cached-prefix pulls even when the PD connector is not offloading, i.e.
	// the engines run MultiConnector(NixlConnector + OffloadingConnector). It has
	// no effect with --kv-connector=offloading, where the tier is always present.
	EnableP2PPull bool

	// EnableSSRFProtection enables SSRF protection using InferencePool allowlisting.
	EnableSSRFProtection bool
	// InferencePoolNamespace is the Kubernetes namespace of the InferencePool to watch.
	InferencePoolNamespace string
	// InferencePoolName is the name of the InferencePool to watch.
	InferencePoolName string
	// PoolGroup is the API group of the InferencePool resource.
	PoolGroup string

	// DecodeChunkSize is the token budget per decode chunk.
	// Chunked decode is enabled when this value is > 0.
	DecodeChunkSize int

	// Tracing enables OpenTelemetry tracing.
	Tracing bool
	// MoRIIOWriteMode enables MoRI-IO WRITE-mode: the sidecar populates the
	// prefill leg's kv_transfer_params so the prefill engine pushes KV to decode
	// via RDMA Write. Only meaningful with --kv-connector=nixlv2.
	MoRIIOWriteMode bool
	// MoRIIODecodeNotifyPort is the decode pod's base MoRI-IO notify port.
	MoRIIODecodeNotifyPort int
	// MoRIIODecodeHandshakePort is the decode pod's base MoRI-IO handshake port.
	MoRIIODecodeHandshakePort int
	// MoRIIODecodePodIP is decode's routable address, used as the prefill leg's
	// remote_host so prefill handshakes with decode (not itself). Must not be
	// localhost; typically the POD_IP downward-API value. May be set to a
	// Kubernetes DNS name (e.g., an LWS pod name), which is resolved to an IP at
	// startup in Complete(); raw IPs are passed through unchanged.
	MoRIIODecodePodIP string

	// MoRIIOParallelDispatch fires the prefill and decode legs concurrently,
	// synthesising decode's kv_transfer_params from config instead of reading
	// them from the prefill response. Requires MoRIIOWriteMode.
	MoRIIOParallelDispatch bool
	// MoRIIOParallelDecodeWaitTimeout bounds how long the parallel WRITE
	// dispatch waits for the prefill outcome before cancelling decode, so a
	// hung or failed prefill cannot make decode wait for KV that never
	// arrives. Zero falls back to defaultMoRIIOParallelDecodeWaitTimeout.
	MoRIIOParallelDecodeWaitTimeout time.Duration
	// MoRIIOPrefillHandshakePort is the prefill pod's base MoRI-IO handshake port.
	MoRIIOPrefillHandshakePort int
	// MoRIIOPrefillNotifyPort is the prefill pod's base MoRI-IO notify port.
	MoRIIOPrefillNotifyPort int
	// MoRIIOTPSize is the tensor-parallel size of the engines, echoed into
	// kv_transfer_params[tp_size] in parallel-dispatch mode.
	MoRIIOTPSize int
	// MoRIIODPSize is the data-parallel world size, emitted as remote_dp_size on
	// both legs. Wide-EP (TP=1, DP>1) must set this so the decode connector
	// registers RDMA notifies against every DP rank; 1 leaves the wire unchanged.
	MoRIIODPSize int

	// MoRIIORemoteHosts is the ordered list of prefill-side pod IPs across which
	// vLLM fans out its per-DP-rank handshake, emitted as the decode leg's
	// remote_hosts. host[i] serves DP ranks [i*MoRIIODPSizeLocal, (i+1)*...).
	// Empty disables fan-out (single-host fallback).
	MoRIIORemoteHosts []string
	// MoRIIODPSizeLocal is the per-pod DP size, mapping a global DP rank to a pod
	// via pod_idx = dp_rank / MoRIIODPSizeLocal. 0 means single-pod.
	MoRIIODPSizeLocal int
	// MoRIIODecodeHosts is the decode-side counterpart of MoRIIORemoteHosts,
	// emitted as the prefill leg's remote_hosts. A multi-pod deployment sets
	// both; the lists must use opposite sides or every cross-pod handshake hangs.
	// DNS names (e.g., LWS pod names) are automatically resolved to IPs at startup.
	MoRIIODecodeHosts []string

	// MoRIIORemoteHostSpecs / MoRIIODecodeHostSpecs / MoRIIODecodePodIPSpec hold
	// the ORIGINAL host specs (DNS names or raw IPs) exactly as supplied on the
	// CLI, captured in Complete() before one-shot resolution rewrites the
	// resolved fields above. The request path re-resolves these specs through a
	// short-TTL hostResolver so a peer pod that restarts with a new IP is picked
	// up without a router restart. Raw-IP specs pass through unchanged.
	MoRIIORemoteHostSpecs []string
	MoRIIODecodeHostSpecs []string
	MoRIIODecodePodIPSpec string
}

// MarshalJSON implements json.Marshaler for Config.
// It overrides the default marshaling of DecoderURL (*url.URL) to serialize it as a string.
func (c Config) MarshalJSON() ([]byte, error) {
	// alias avoids infinite recursion when calling json.Marshal below
	type alias Config
	decoderURL := ""
	if c.DecoderURL != nil {
		decoderURL = c.DecoderURL.String()
	}
	return json.Marshal(struct {
		alias
		DecoderURL string
	}{
		alias:      alias(c),
		DecoderURL: decoderURL,
		// Tracing is serialized automatically as it is part of alias
	})
}

// String returns a JSON representation of Config for logging and debugging.
// It implements fmt.Stringer.
func (c Config) String() string {
	b, _ := json.Marshal(c)
	return string(b)
}

// pdConnectorHandler handles a P/D KV connector request. kvCacheSource is the
// validated x-kv-cache-source-host-port peer to pull cached prefix from ("" when
// absent); the APIType lets each connector decide internally which JSON fields
// (if any) need special handling.
type pdConnectorHandler func(http.ResponseWriter, *http.Request, string, string, APIType)

type ecConnectorHandler func(http.ResponseWriter, *http.Request, string, []string)

// Server is the reverse proxy server
type Server struct {
	logger             logr.Logger
	addr               net.Addr      // the proxy TCP address
	readyCh            chan struct{} // closed once addr is set and server is listening
	handler            http.Handler  // the handler function. either a Mux or a proxy
	allowlistValidator *AllowlistValidator
	handlePDConnector  pdConnectorHandler // handles the Prefiller-Decoder connector request
	handleECConnector  ecConnectorHandler // handles the Encoder disaggregation connector request.
	prefillerURLPrefix string
	encoderURLPrefix   string

	decoderProxy        http.Handler                          // decoder proxy handler
	prefillerProxies    *lru.Cache[string, http.Handler]      // cached prefiller proxy handlers
	encoderProxies      *lru.Cache[string, http.Handler]      // cached encoder proxy handlers
	mooncakeEngineIDs   *lru.Cache[string, map[string]string] // cached mooncake dp_rank->engine_id per prefill host:port
	dataParallelProxies map[string]http.Handler               // Proxies to other vLLM servers
	forwardDataParallel bool                                  // Use special Data Parallel work around

	prefillSamplerFn func(n int) int // allow test override

	// dpBasePort is the rank-0 proxy port. Rank clones override config.Port
	// (data_parallel.go), so rank derivation from a routed endpoint's port
	// needs the pre-clone base. 0 disables derivation.
	dpBasePort int

	// hostResolver re-resolves MoRI-IO peer DNS specs to IPs on a short TTL so
	// peer pod restarts are picked up on the request path. Lazily initialized
	// via resolverOnce (clone-safe; each Server gets its own).
	resolverOnce sync.Once
	hostResolver *hostResolver

	config Config
}

// resolver lazily initializes and returns the request-path host resolver.
// Initialization is deferred so it can use s.logger (populated in Start) and
// so Clone'd servers each get their own instance.
func (s *Server) resolver() *hostResolver {
	s.resolverOnce.Do(func() {
		s.hostResolver = newHostResolver(s.logger, resolveTTLFromEnv())
		s.seedResolver(s.hostResolver)
	})
	return s.hostResolver
}

// seedResolver primes the request-path resolver with the spec->IP mappings that
// Complete() already resolved at startup, so the first request does not repeat
// those lookups and a request-time DNS failure still serves the startup IP
// instead of the raw hostname (which would hang the MoRI-IO handshake). Specs
// and their resolved IPs are captured positionally in Complete(); a length
// mismatch (e.g. a Config built directly in tests) skips seeding for that list.
func (s *Server) seedResolver(r *hostResolver) {
	seedPairs := func(specs, ips []string) {
		if len(specs) != len(ips) {
			return
		}
		for i := range specs {
			r.seed(specs[i], ips[i])
		}
	}
	seedPairs(s.config.MoRIIODecodeHostSpecs, s.config.MoRIIODecodeHosts)
	seedPairs(s.config.MoRIIORemoteHostSpecs, s.config.MoRIIORemoteHosts)
	r.seed(s.config.MoRIIODecodePodIPSpec, s.config.MoRIIODecodePodIP)
}

// currentDecodeHosts returns the decode-side peer IPs for the current request,
// re-resolving the original specs through the short-TTL resolver. The request
// context bounds any cold-start lookup so a cancelled/timed-out request cancels
// the DNS lookup. Falls back to the boot-resolved config field when no specs
// were captured (e.g. tests that build Config directly).
func (s *Server) currentDecodeHosts(ctx context.Context) []string {
	if len(s.config.MoRIIODecodeHostSpecs) == 0 {
		return s.config.MoRIIODecodeHosts
	}
	return s.resolver().resolve(ctx, s.config.MoRIIODecodeHostSpecs)
}

// currentRemoteHosts is the prefill-side counterpart of currentDecodeHosts.
func (s *Server) currentRemoteHosts(ctx context.Context) []string {
	if len(s.config.MoRIIORemoteHostSpecs) == 0 {
		return s.config.MoRIIORemoteHosts
	}
	return s.resolver().resolve(ctx, s.config.MoRIIORemoteHostSpecs)
}

// currentDecodePodIP returns decode's advertised remote_host for the current
// request, re-resolving the original spec through the short-TTL resolver. The
// request context bounds any cold-start lookup.
func (s *Server) currentDecodePodIP(ctx context.Context) string {
	if s.config.MoRIIODecodePodIPSpec == "" {
		return s.config.MoRIIODecodePodIP
	}
	return s.resolver().resolveOne(ctx, s.config.MoRIIODecodePodIPSpec)
}

// NewProxy creates a new routing reverse proxy from the given Config.
func NewProxy(config Config) *Server {
	prefillerCache, _ := lru.New[string, http.Handler](1024)         // nolint:errcheck
	encoderCache, _ := lru.New[string, http.Handler](1024)           // nolint:errcheck
	mooncakeEngineIDs, _ := lru.New[string, map[string]string](1024) // nolint:errcheck

	server := &Server{
		readyCh:             make(chan struct{}),
		prefillerProxies:    prefillerCache,
		encoderProxies:      encoderCache,
		mooncakeEngineIDs:   mooncakeEngineIDs,
		prefillerURLPrefix:  "http://",
		encoderURLPrefix:    "http://",
		config:              config,
		dataParallelProxies: map[string]http.Handler{},
		forwardDataParallel: true,
		prefillSamplerFn:    rand.IntN,
	}
	if basePort, err := strconv.Atoi(config.Port); err == nil {
		server.dpBasePort = basePort
	}

	server.setKVConnector()
	if config.UseTLSForPrefiller {
		server.prefillerURLPrefix = "https://"
	}

	if config.ECConnector != "" {
		server.setECConnector()
		if config.UseTLSForEncoder {
			server.encoderURLPrefix = "https://"
		}
	}

	return server
}

// Start the HTTP reverse proxy.
// allowlistValidator is constructed from s.config on first call; inject an alternative before calling Start to override.
func (s *Server) Start(ctx context.Context) error {
	s.logger = log.FromContext(ctx).WithName("proxy server on port " + s.config.Port)

	if s.allowlistValidator == nil {
		var err error
		s.allowlistValidator, err = NewAllowlistValidator(
			s.config.EnableSSRFProtection,
			s.config.PoolGroup,
			s.config.InferencePoolNamespace,
			s.config.InferencePoolName,
		)
		if err != nil {
			return err
		}
	}

	// Configure handlers
	s.handler = s.createRoutes()

	grp, ctx := errgroup.WithContext(ctx)
	if err := s.startDataParallel(ctx, grp); err != nil {
		return err
	}

	grp.Go(func() error {
		return s.startHTTP(ctx)
	})

	// Opt-in Prometheus /metrics endpoint (MORIIO_METRICS_ADDR); no-op when unset.
	s.maybeStartMetrics(ctx, grp)

	return grp.Wait()
}

// Clone returns a clone of the current Server struct.
// Note: decoderURL and decoderProxy are intentionally not copied — callers (e.g. startDataParallel)
// always set them explicitly after cloning.
func (s *Server) Clone() *Server {
	return &Server{
		addr:                s.addr,
		readyCh:             make(chan struct{}),
		handler:             s.handler,
		allowlistValidator:  s.allowlistValidator,
		handlePDConnector:   s.handlePDConnector,
		handleECConnector:   s.handleECConnector,
		prefillerURLPrefix:  s.prefillerURLPrefix,
		encoderURLPrefix:    s.encoderURLPrefix,
		prefillerProxies:    s.prefillerProxies,
		encoderProxies:      s.encoderProxies,
		mooncakeEngineIDs:   s.mooncakeEngineIDs,
		dataParallelProxies: s.dataParallelProxies,
		forwardDataParallel: s.forwardDataParallel,
		prefillSamplerFn:    s.prefillSamplerFn,
		dpBasePort:          s.dpBasePort,
		config:              s.config,
	}
}

// newProxyTransport returns an http.RoundTripper backed by an http.Transport
// cloned from the default with connection-pool settings applied. If scheme is
// schemeHTTPS the transport's TLSClientConfig is set accordingly. The transport
// is wrapped with otelhttp so outbound requests carry W3C trace context,
// keeping EPP, routing-proxy, and vLLM spans in a single trace.
func (s *Server) newProxyTransport(scheme string, insecureSkipVerify bool) http.RoundTripper {
	maxIdle := s.config.MaxIdleConnsPerHost
	if maxIdle <= 0 {
		maxIdle = defaultMaxIdleConnsPerHost
	}
	t := http.DefaultTransport.(*http.Transport).Clone() //nolint:errcheck
	t.MaxIdleConns = 0                                   // unlimited
	t.MaxIdleConnsPerHost = maxIdle
	t.MaxConnsPerHost = 0 // unlimited
	t.IdleConnTimeout = 90 * time.Second
	if scheme == schemeHTTPS {
		t.TLSClientConfig = &tls.Config{
			InsecureSkipVerify: insecureSkipVerify, //nolint:gosec
			MinVersion:         tls.VersionTLS12,
			CipherSuites: []uint16{
				tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
				tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
				tls.TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
				tls.TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
				tls.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305,
				tls.TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305,
			},
		}
	}
	return otelhttp.NewTransport(t)
}

func (s *Server) setKVConnector() {

	switch s.config.KVConnector {
	case KVConnectorSharedStorage:
		s.handlePDConnector = func(w http.ResponseWriter, r *http.Request, host string, _ string, _ APIType) {
			s.handleSharedStorage(w, r, host)
		}
	case KVConnectorSGLang:
		s.handlePDConnector = func(w http.ResponseWriter, r *http.Request, host string, _ string, _ APIType) {
			s.handleSGLang(w, r, host)
		}
	case KVConnectorMooncake:
		s.handlePDConnector = func(w http.ResponseWriter, r *http.Request, host string, _ string, _ APIType) {
			s.handleMooncake(w, r, host)
		}
	case KVConnectorOffloading:
		s.handlePDConnector = func(w http.ResponseWriter, r *http.Request, host string, kvCacheSource string, _ APIType) {
			s.handleP2P(w, r, host, kvCacheSource)
		}
	case KVConnectorNIXLV2:
		fallthrough
	default:
		s.handlePDConnector = func(w http.ResponseWriter, r *http.Request, host string, kvCacheSource string, apiType APIType) {
			s.handleNIXLV2(w, r, host, kvCacheSource, apiType)
		}
	}
}

func (s *Server) setECConnector() {
	ecConnector := s.config.ECConnector

	if ecConnector == "" {
		// No encoder connector specified, encoder stage will be skipped
		return
	}

	switch ecConnector {
	case ECExampleConnector:
		s.handleECConnector = s.handleECSharedStorage
	case ECConnectorNIXL:
		s.handleECConnector = s.handleECNIXL
	default:
		// Unknown EC connector value, skip encoder stage. Validate() should
		// have rejected this earlier; reaching here means the validation was
		// bypassed (e.g., programmatic config) and the binary degrades.
		s.logger.Info("warning: unknown ec-connector; encoder stage will be skipped",
			"ecConnector", ecConnector, "supported", supportedECConnectorNamesStr)
		return
	}
}

func (s *Server) createRoutes() *http.ServeMux {
	// Configure handlers
	mux := http.NewServeMux()

	// Intercept chat requests
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("POST "+ChatCompletionsPath, s.disaggregatedPrefillHandler(APITypeChatCompletions))
	mux.HandleFunc("POST "+CompletionsPath, s.disaggregatedPrefillHandler(APITypeChatCompletions))
	mux.HandleFunc("POST "+MessagesPath, s.disaggregatedPrefillHandler(APITypeChatCompletions))
	mux.HandleFunc("POST "+ResponsesPath, s.disaggregatedPrefillHandler(APITypeResponses))
	mux.HandleFunc("POST "+GeneratePath, s.disaggregatedPrefillHandler(APITypeGenerate))

	s.decoderProxy = s.createDecoderProxyHandler(s.config.DecoderURL, s.config.InsecureSkipVerifyForDecoder)

	mux.Handle("/", s.decoderProxy)

	return mux
}

// createProxyHandler creates a reverse proxy handler for the given host:port.
// It uses the provided cache, URL prefix, and TLS settings.
func (s *Server) createProxyHandler(
	hostPort string,
	cache *lru.Cache[string, http.Handler],
	urlPrefix string,
	insecureSkipVerify bool,
) (http.Handler, error) {
	// Check cache first
	proxy, exists := cache.Get(hostPort)
	if exists {
		return proxy, nil
	}

	// Backward compatible behavior: trim `http:` prefix
	hostPort, _ = strings.CutPrefix(hostPort, "http://")

	u, err := url.Parse(urlPrefix + hostPort)
	if err != nil {
		s.logger.Error(err, "failed to parse URL", "hostPort", hostPort)
		return nil, err
	}

	newProxy := httputil.NewSingleHostReverseProxy(u)
	newProxy.Transport = s.newProxyTransport(u.Scheme, insecureSkipVerify)
	cache.Add(hostPort, newProxy)

	return newProxy, nil
}

func (s *Server) prefillerProxyHandler(hostPort string) (http.Handler, error) {
	return s.createProxyHandler(
		hostPort,
		s.prefillerProxies,
		s.prefillerURLPrefix,
		s.config.InsecureSkipVerifyForPrefiller,
	)
}

func (s *Server) encoderProxyHandler(hostPort string) (http.Handler, error) {
	return s.createProxyHandler(
		hostPort,
		s.encoderProxies,
		s.encoderURLPrefix,
		s.config.InsecureSkipVerifyForEncoder,
	)
}
