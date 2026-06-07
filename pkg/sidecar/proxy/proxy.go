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
	"strings"
	"time"

	"github.com/go-logr/logr"
	lru "github.com/hashicorp/golang-lru/v2"
	"golang.org/x/sync/errgroup"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
)

const (
	schemeHTTPS = "https"

	defaultMaxIdleConnsPerHost = 1024

	requestHeaderRequestID = "x-request-id"

	// requestHeaderDataParallelRank is the HTTP header vLLM honours
	// in AsyncLLM.add_request when the caller wants to pin a request
	// to a specific data-parallel rank, bypassing the round-robin
	// DPLBAsyncMPClient load balancer.
	//
	// We set this on BOTH the prefill and the decode legs of a disagg
	// pair to a stable per-request hash (pickDPRank(uuid, dpSize)) so
	// the SO_REUSEPORT shuffle that --api-server-count>1 induces does
	// not split the two legs across different DP ranks. Without this
	// the prefill engine handshakes a peer (rank H) but decode is
	// listening on a different rank, the notify never lands, and the
	// request hangs until VLLM_MORIIO_DEFERRED_TIMEOUT_S (default
	// 300 s) trips. See pkg/sidecar/proxy/dp_rank.go for the helper.
	requestHeaderDataParallelRank = "x-data-parallel-rank"

	requestFieldKVTransferParams    = "kv_transfer_params"
	requestFieldMaxTokens           = "max_tokens"
	requestFieldMaxCompletionTokens = "max_completion_tokens"
	requestFieldMaxOutputTokens     = "max_output_tokens" // Used by Responses API
	requestFieldDoRemotePrefill     = "do_remote_prefill"
	requestFieldDoRemoteDecode      = "do_remote_decode"
	requestFieldRemoteBlockIDs      = "remote_block_ids"
	requestFieldRemoteEngineID      = "remote_engine_id"
	requestFieldRemoteHost          = "remote_host"
	requestFieldRemotePort          = "remote_port"
	// MoRI-IO WRITE-mode kv_transfer_params fields.
	// When --moriio-write-mode is enabled (only meaningful with --kv-connector=nixlv2
	// against engines running vLLM's MoRIIOConnector in WRITE mode), the sidecar
	// populates these on the prefill leg so the prefill engine knows where to
	// push KV blocks via RDMA Write.
	requestFieldRemoteNotifyPort = "remote_notify_port"
	requestFieldRemoteDPRank     = "remote_dp_rank"
	// requestFieldRemoteDPRankOverride is a sentinel field the sidecar
	// sets to true whenever it stamps a per-request remote_dp_rank
	// (computed via pickDPRank). The decode-side
	// MoRIIOConnectorScheduler.request_finished hash-route fix in
	// vLLM checks for this flag and, when present, uses
	// kv_transfer_params["remote_dp_rank"] verbatim instead of
	// recomputing its own blake2s(rid, digest_size=8) hash. Setting
	// it here lets the sidecar be the single source of truth for the
	// per-request DP rank on both legs of the pair, and avoids any
	// accidental cross-language hash divergence (Go blake2s-256 vs
	// Python blake2s-8).
	requestFieldRemoteDPRankOverride = "remote_dp_rank_override"
	// remote_handshake_port is the decode pod's MoRI-IO handshake port (default
	// 6301, matches MoRIIOConstants.DEFAULT_HANDSHAKE_PORT in vLLM).
	requestFieldRemoteHandshakePort  = "remote_handshake_port"
	requestFieldTransferID           = "transfer_id"
	requestFieldStream               = "stream"
	requestFieldStreamOptions        = "stream_options"
	requestFieldCacheHitThreshold    = "cache_hit_threshold"
	requestFieldContinueFinalMessage = "continue_final_message"
	requestFieldAddGenerationPrompt  = "add_generation_prompt"

	responseFieldChoices      = "choices"
	responseFieldFinishReason = "finish_reason"

	finishReasonCacheThreshold = "cache_threshold"

	// SGLang bootstrap fields
	requestFieldBootstrapHost = "bootstrap_host"
	requestFieldBootstrapPort = "bootstrap_port"
	requestFieldBootstrapRoom = "bootstrap_room"

	KVConnectorNIXLV2        = constants.KVConnectorNIXLV2
	KVConnectorSharedStorage = constants.KVConnectorSharedStorage
	KVConnectorSGLang        = constants.KVConnectorSGLang
	ECExampleConnector       = constants.ECExampleConnector
)

// APIType represents the type of OpenAI API being used.
type APIType int

const (
	// APITypeChatCompletions is the Chat Completions API (/v1/chat/completions, /v1/completions)
	APITypeChatCompletions APIType = iota
	// APITypeResponses is the Responses API (/v1/responses)
	APITypeResponses
)

// String implements fmt.Stringer so structured logs show readable API names.
func (a APIType) String() string {
	switch a {
	case APITypeChatCompletions:
		return "chat_completions"
	case APITypeResponses:
		return "responses"
	default:
		return fmt.Sprintf("APIType(%d)", int(a))
	}
}

// JSON request field names used for token limits in prefill/decode staging.
// Do not mutate these slices.
var (
	chatCompletionTokenLimitFields = []string{requestFieldMaxTokens, requestFieldMaxCompletionTokens}
	responsesStyleTokenLimitFields = []string{requestFieldMaxOutputTokens}
)

// tokenLimitFieldsForAPIType returns token limit field names for the given API.
// Returned slices are shared package-level vars; callers must not mutate them.
func tokenLimitFieldsForAPIType(api APIType) []string {
	switch api {
	case APITypeResponses:
		return responsesStyleTokenLimitFields
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

	// MoRIIOWriteMode enables MoRI-IO WRITE-mode passthrough on the NIXLv2 path.
	// When true, the sidecar populates remote_host/remote_notify_port/transfer_id
	// in the prefill leg's kv_transfer_params so the prefill MoRIIOConnector can
	// push KV blocks to the decode engine via RDMA Write rather than have decode
	// pull them via RDMA Read.  Only meaningful with --kv-connector=nixlv2 against
	// engines running vLLM's MoRIIOConnector with VLLM_MORIIO_CONNECTOR_READ_MODE=0.
	MoRIIOWriteMode bool
	// MoRIIODecodeNotifyPort is the decode pod's base MoRI-IO notify port.
	// Matches MoRIIOConstants.DEFAULT_NOTIFY_PORT (61005) in vLLM's MoRI-IO
	// connector.  The prefill engine adds get_port_offset(dp_rank, tp_index) to
	// this base when iterating over TP ranks during send_notify_block.
	MoRIIODecodeNotifyPort int
	// MoRIIODecodeDPRank is the decode engine's data-parallel rank, used by the
	// prefill connector to compute the per-TP target_port offset.
	// Defaults to 0 which is correct for single-DP and DP-rank-0 deployments.
	MoRIIODecodeDPRank int
	// MoRIIODecodeHandshakePort is the decode pod's MoRI-IO handshake port.
	// Matches MoRIIOConstants.DEFAULT_HANDSHAKE_PORT (6301) in vLLM.  vLLM's
	// MoRIIOConnectorWorker listens here (on TP rank 0) for the prefill
	// engine's RDMA handshake before WRITE-mode KV transfers.
	// Defaults to 6301.
	MoRIIODecodeHandshakePort int
	// MoRIIODecodePodIP is the decode pod's routable IP address as seen from
	// other pods in the cluster.  In WRITE mode the prefill engine derives
	// its destination engine_id from this value (moriio_connector.py:1330):
	//
	//   remote_engine_id = str(meta.remote_host) + ":" + str(meta.remote_handshake_port)
	//
	// and then handshakes with decode at (remote_host, remote_handshake_port)
	// to exchange RDMA metadata before WRITE-mode KV transfers.
	//
	// This must be decode's routable pod IP, not localhost: the sidecar
	// is co-located with decode, so a localhost value would cause prefill
	// to handshake with itself.  Typically templated by Helm from the K8s
	// downward API into the routing-proxy initContainer as the POD_IP env
	// var, surfaced through --moriio-local-pod-ip.
	MoRIIODecodePodIP string

	// MoRIIOParallelDispatch enables concurrent prefill/decode HTTP
	// dispatch from the sidecar.  When false (the default) the proxy is
	// strictly serial: it sends the prefill request, blocks until prefill
	// returns a body, then sends the decode request using fields
	// extracted from prefill's response.  When true the proxy synthesises
	// decode's kv_transfer_params up-front (from prefillPodHostPort + the
	// MoRIIOPrefill* config below) and fires the prefill and decode
	// requests in parallel goroutines, so decode's
	// MoRIIOConnector.update_state_after_alloc / send_notify_block /
	// block allocation overlaps with prefill's forward pass -- TTFT
	// becomes max(P_compute + RDMA_write, D_queue) instead of
	// P_compute + D_queue + RDMA_burst.
	//
	// Requires --moriio-write-mode (concurrent dispatch is a WRITE-mode-only
	// optimisation; the READ-mode path needs prefill's remote_block_ids and
	// remote_engine_id from the prefill response and cannot proceed without it).
	MoRIIOParallelDispatch bool
	// MoRIIOPrefillHandshakePort is the prefill pod's base MoRI-IO handshake
	// port (matches MoRIIOConstants.DEFAULT_HANDSHAKE_PORT in vLLM, 6301).  The
	// decode-side MoRIIOConnector connects here during send_notify_block to
	// register block addresses with the prefill engine.  In concurrent-
	// dispatch mode the proxy synthesises kv_transfer_params["remote_handshake_port"] on the decode
	// leg from this value (without it the decode side cannot establish the
	// handshake socket before prefill starts writing).  In uniform-port
	// deployments this equals MoRIIODecodeHandshakePort; they are kept
	// separate to make non-uniform deployments tractable.
	MoRIIOPrefillHandshakePort int
	// MoRIIOPrefillNotifyPort is the prefill pod's base MoRI-IO notify port
	// (matches MoRIIOConstants.DEFAULT_NOTIFY_PORT in vLLM, 61005).  Used by
	// the decode-side connector to send block-allocation notifications back
	// to prefill so prefill knows where to RDMA-Write each KV layer.
	MoRIIOPrefillNotifyPort int
	// MoRIIOPrefillDPRank is the prefill engine's data-parallel rank, used by
	// the decode connector to compute the per-TP target_port offset when
	// sending block-allocation notifications.  Defaults to 0 (single-DP
	// prefill).
	MoRIIOPrefillDPRank int
	// MoRIIOTPSize is the tensor-parallel size of the prefill/decode engines.
	// Echoed into kv_transfer_params["tp_size"] in concurrent-dispatch mode
	// so vLLM's MoRIIOConnector can size the per-TP notify-port stride in
	// send_notify_block.  Defaults to 1; should be set to match the
	// engine's --tensor-parallel-size.
	MoRIIOTPSize int
	// MoRIIODPSize is the data-parallel world size of the prefill/decode
	// engines.  Emitted as `remote_dp_size` on BOTH legs of the
	// kv_transfer_params payload in concurrent-dispatch mode (see
	// runNIXLProtocolV2WriteParallel).
	//
	// Wide-EP deployments (TP=1, DP>1) MUST set this to the actual DP size
	// (e.g. 16 for Wide-EP-16) so vLLM's MoRIIOConnector iterates the
	// per-DP-rank handshake loop in update_state_after_alloc and registers
	// RDMA notifies against every remote DP rank.  At the default of 1 the
	// connector targets remote_dp_rank only, the other DP ranks' WriteTasks
	// never receive their notify, and the request hangs until
	// --distributed-timeout-seconds expires (-> bench 502s).
	//
	// Single-DP deployments leave this at 1 to keep the wire shape
	// identical to upstream's hardcoded value.
	MoRIIODPSize int
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
	})
}

// String returns a JSON representation of Config for logging and debugging.
// It implements fmt.Stringer.
func (c Config) String() string {
	b, _ := json.Marshal(c)
	return string(b)
}

// pdConnectorHandler handles a P/D KV connector request. The APIType lets each
// connector decide internally which JSON fields (if any) need special handling.
type pdConnectorHandler func(http.ResponseWriter, *http.Request, string, APIType)

type epdConnectorHandler func(http.ResponseWriter, *http.Request, string, []string)

// Server is the reverse proxy server
type Server struct {
	logger             logr.Logger
	addr               net.Addr      // the proxy TCP address
	readyCh            chan struct{} // closed once addr is set and server is listening
	handler            http.Handler  // the handler function. either a Mux or a proxy
	allowlistValidator *AllowlistValidator
	handlePDConnector  pdConnectorHandler  // handles the Prefiller-Decoder connector request
	handleEPDConnector epdConnectorHandler // handles the Encoder-Prefiller-Decoder connector request
	prefillerURLPrefix string
	encoderURLPrefix   string

	decoderProxy        http.Handler                     // decoder proxy handler
	prefillerProxies    *lru.Cache[string, http.Handler] // cached prefiller proxy handlers
	encoderProxies      *lru.Cache[string, http.Handler] // cached encoder proxy handlers
	dataParallelProxies map[string]http.Handler          // Proxies to other vLLM servers
	forwardDataParallel bool                             // Use special Data Parallel work around

	prefillSamplerFn func(n int) int // allow test override

	config Config
}

// NewProxy creates a new routing reverse proxy from the given Config.
func NewProxy(config Config) *Server {
	prefillerCache, _ := lru.New[string, http.Handler](1024) // nolint:errcheck
	encoderCache, _ := lru.New[string, http.Handler](1024)   // nolint:errcheck

	server := &Server{
		readyCh:             make(chan struct{}),
		prefillerProxies:    prefillerCache,
		encoderProxies:      encoderCache,
		prefillerURLPrefix:  "http://",
		encoderURLPrefix:    "http://",
		config:              config,
		dataParallelProxies: map[string]http.Handler{},
		forwardDataParallel: true,
		prefillSamplerFn:    rand.IntN,
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
		handleEPDConnector:  s.handleEPDConnector,
		prefillerURLPrefix:  s.prefillerURLPrefix,
		encoderURLPrefix:    s.encoderURLPrefix,
		prefillerProxies:    s.prefillerProxies,
		encoderProxies:      s.encoderProxies,
		dataParallelProxies: s.dataParallelProxies,
		forwardDataParallel: s.forwardDataParallel,
		prefillSamplerFn:    s.prefillSamplerFn,
		config:              s.config,
	}
}

// newProxyTransport returns an http.Transport cloned from the default with
// connection-pool settings applied. If scheme is schemeHTTPS the transport's
// TLSClientConfig is set accordingly.
func (s *Server) newProxyTransport(scheme string, insecureSkipVerify bool) *http.Transport {
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
	return t
}

func (s *Server) setKVConnector() {

	switch s.config.KVConnector {
	case KVConnectorSharedStorage:
		s.handlePDConnector = func(w http.ResponseWriter, r *http.Request, host string, _ APIType) {
			s.handleSharedStorage(w, r, host)
		}
	case KVConnectorSGLang:
		s.handlePDConnector = func(w http.ResponseWriter, r *http.Request, host string, _ APIType) {
			s.handleSGLang(w, r, host)
		}
	case KVConnectorNIXLV2:
		fallthrough
	default:
		s.handlePDConnector = s.handleNIXLV2
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
		s.handleEPDConnector = s.handleEPD
	default:
		// Unknown EC connector value, skip encoder stage
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
	mux.HandleFunc("POST "+ResponsesPath, s.disaggregatedPrefillHandler(APITypeResponses))

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
