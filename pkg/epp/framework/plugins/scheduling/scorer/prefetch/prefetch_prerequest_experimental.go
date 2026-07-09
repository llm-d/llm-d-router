package prefetch

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	preciseproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/preciseprefixcache"
)

const (
	// PrefetchPrerequestHandlerType is the type of the PrefetchPrerequestHandler.
	PrefetchPrerequestHandlerType = "prefetch-prerequest-handler"
)

// digestToFilenamePathSuffix converts a full-width block-hash digest to the
// path suffix used by llm-d-fs-connector: <hhh>/<hh>_g<groupIdx>/<hex>.bin.
// The digest width matches the upstream prefix-caching hash algo: 32 bytes
// for sha256_cbor (vLLM ≥ v0.10.2 → 64 hex chars on disk) or 8 bytes for
// fnv64a (16 hex chars). The hash bytes name the leaf and the <hhh>/<hh>
// subfolders; group_idx lives in the parent folder name.
func digestToFilenamePathSuffix(digest []byte, groupIdx int) string {
	hashHex := hex.EncodeToString(digest)
	sub1, sub2 := hashHex[:3], hashHex[3:5]
	return fmt.Sprintf("%s/%s_g%d/%s.bin", sub1, sub2, groupIdx, hashHex)
}

// KVFilePathBaseParams holds operator-supplied parameters for KV-cache file
// paths. All fields are deserialized from the plugin's JSON config; this
// struct holds no runtime state.
//
// On-disk layout written by vLLM (one identical filename per rank, with
// different byte contents — each rank stores only its own KV shard):
//
//	<rootDir>/<safeModelName>_<digest>_r<rank>/<hhh>/<hh>_g<groupIdx>/<hash>.bin
//
// Every component except the per-block <hash> is operator-supplied:
//   - <safeModelName> is ModelName with '/' replaced by '_'.
//   - <digest> is the 12-hex fs-connector fingerprint over the vLLM cache
//     config (Digest). The router cannot reliably recompute it, so the
//     operator reads it from the vLLM pod and supplies it directly.
//   - <groupIdx> is the KV-cache group index (GroupIdx, typically 0).
//   - <rank> ranges over [0, TpSize*PpSize*PcpSize*DcpSize); every worker
//     shares the same base prefix (file_mapper.py:117), so each rank folder
//     is the base with "_r<rank>" appended.
type KVFilePathBaseParams struct {
	RootDir          string `json:"rootDir"`
	ModelName        string `json:"modelName"`
	Digest           string `json:"digest"`
	GroupIdx         int    `json:"groupIdx"`
	GpuBlocksPerFile int    `json:"gpuBlocksPerFile"`
	TpSize           int    `json:"tpSize"`
	PpSize           int    `json:"ppSize"`
	PcpSize          int    `json:"pcpSize"`
	DcpSize          int    `json:"dcpSize"`
}

// IsSet returns true if a base path can be built. RootDir, ModelName, and the
// fs-connector Digest are all required.
func (b *KVFilePathBaseParams) IsSet() bool {
	return b != nil && b.RootDir != "" && b.ModelName != "" && b.Digest != ""
}

// SetDefaults applies default values to unset KVFilePathBaseParams fields.
func (b *KVFilePathBaseParams) SetDefaults() {
	if b.GpuBlocksPerFile < 1 {
		b.GpuBlocksPerFile = 1
	}
	if b.TpSize < 1 {
		b.TpSize = 1
	}
	if b.PpSize < 1 {
		b.PpSize = 1
	}
	if b.PcpSize < 1 {
		b.PcpSize = 1
	}
	if b.DcpSize < 1 {
		b.DcpSize = 1
	}
}

// basePath returns the rank-shared base prefix
// "<rootDir>/<safeModelName>_<digest>", where safeModelName is ModelName with
// '/' replaced by '_'. Per-rank folders append "_r<rank>".
func (b *KVFilePathBaseParams) basePath() string {
	safeModelName := strings.ReplaceAll(b.ModelName, "/", "_")
	return filepath.Join(b.RootDir, safeModelName+"_"+b.Digest)
}

// digestToFullPath returns the complete on-disk file path for a block-hash
// digest on the given rank, built entirely from operator config:
// "<base>_r<rank>/<hhh>/<hh>_g<groupIdx>/<hash>.bin".
func (b *KVFilePathBaseParams) digestToFullPath(rank int, digest []byte) string {
	suffix := digestToFilenamePathSuffix(digest, b.GroupIdx)
	return fmt.Sprintf("%s_r%d/%s", b.basePath(), rank, filepath.FromSlash(suffix))
}

// digestsToFilePaths returns the file paths to prefetch for the given
// block-hash digests on the given rank. The fs-connector aggregates
// GpuBlocksPerFile vLLM blocks into one on-disk file, naming it after the
// last block's hash — so when n>1, only digests at indices n-1, 2n-1, 3n-1,
// ... correspond to files on disk. When fewer than n digests are present, no
// aggregated file has been written for this request yet and nil is returned.
func digestsToFilePaths(base *KVFilePathBaseParams, rank int, digests [][]byte) []string {
	if len(digests) == 0 {
		return nil
	}
	n := base.GpuBlocksPerFile
	if n < 1 {
		n = 1
	}
	if len(digests) < n {
		// No aggregated file has been written for this request yet.
		return nil
	}

	if n == 1 {
		paths := make([]string, 0, len(digests))
		for _, d := range digests {
			paths = append(paths, base.digestToFullPath(rank, d))
		}
		return paths
	}

	paths := make([]string, 0, len(digests)/n)
	for i := n - 1; i < len(digests); i += n {
		paths = append(paths, base.digestToFullPath(rank, digests[i]))
	}
	return paths
}

func prefetchFile(ctx context.Context, filePath string, buffer []byte) error {
	file, err := os.Open(filePath)
	if err != nil {
		// A missing file is expected: vLLM may not have written this block
		// yet, or the operator-supplied digest/sizes don't match what's on
		// disk. Treat it as a benign skip rather than a prefetch failure.
		if errors.Is(err, os.ErrNotExist) {
			log.FromContext(ctx).V(logging.DEFAULT).Info("prefetchFile: file not present, skipping", "path", filePath)
			return nil
		}
		return fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	finfo, err := file.Stat()
	if err != nil {
		return fmt.Errorf("failed to stat file: %w", err)
	}
	// finfo.Size() is the logical object size as known from the remote object
	// store; st_blocks*512 is how many bytes are actually resident in the local
	// storage cache. A not-yet-fetched stub reports the full logical size but
	// near-zero allocated blocks.
	var allocated int64
	if st, ok := finfo.Sys().(*syscall.Stat_t); ok {
		allocated = st.Blocks * 512 // st_blocks is always counted in 512-byte units
	}
	log.FromContext(ctx).V(logging.VERBOSE).Info("prefetchFile: file stat",
		"path", filePath, "size", finfo.Size(), "allocated", allocated, "bufferSize", len(buffer), "modTime", finfo.ModTime())

	// A prefetch is a single buffer-sized read whose only purpose is to fault
	// the object in from the remote object store into the local storage cache.
	// This experimental plugin is tested against IBM Storage Scale with AFM
	// (Active File Management): reading the first buffer (the configured number
	// of blocks) is enough to trigger AFM to fetch the whole object from the
	// remote store, so a single read warms the entire file regardless of size.
	//
	// Skip the read when the object is already cached. The test depends on size
	// because a resident object has the whole file faulted in by AFM:
	//   - size <= bufferSize: the object fits within one read, so it is cached
	//     iff fully resident, i.e. allocated covers the logical size.
	//   - size  > bufferSize: a resident object has far more than a buffer's
	//     worth allocated, so allocated > bufferSize means AFM already pulled it.
	var cached bool
	if finfo.Size() <= int64(len(buffer)) {
		cached = allocated >= finfo.Size()
	} else {
		cached = allocated > int64(len(buffer))
	}
	if cached {
		log.FromContext(ctx).V(logging.DEBUG).Info("prefetchFile: object already resident in storage cache, skipping",
			"path", filePath, "size", finfo.Size(), "allocated", allocated, "bufferSize", len(buffer))
		return nil
	}

	n, err := file.Read(buffer)
	if err != nil && err != io.EOF {
		return fmt.Errorf("failed to read file: %w", err)
	}

	if err == io.EOF || n < len(buffer) {
		log.FromContext(ctx).V(logging.DEBUG).Info("prefetchFile: read partial file",
			"path", filePath, "requestedBytes", len(buffer), "actualBytes", n)
	} else {
		log.FromContext(ctx).V(logging.DEBUG).Info("prefetchFile: read complete",
			"path", filePath, "bytes", n)
	}

	return nil
}

func initializeWorkerPool(ctx context.Context, handler *PrefetchPrerequestHandler) error {
	if handler.prefetchConfig == nil || !handler.prefetchConfig.Enabled {
		log.FromContext(ctx).Info("initializeWorkerPool: prefetching disabled")
		return nil
	}

	config := handler.prefetchConfig
	log.FromContext(ctx).Info("initializeWorkerPool: initializing worker pool",
		"maxConcurrentFiles", config.MaxConcurrentFiles,
		"workQueueSize", config.WorkQueueSize,
		"blockSize", config.BlockSize,
		"blockCount", config.BlockCount)

	pool := &PrefetchWorkerPool{
		workQueue:   make(chan string, config.WorkQueueSize),
		workersDone: make(chan struct{}),
	}
	pool.shutdownCtx, pool.shutdownFn = context.WithCancel(context.Background())
	handler.workerPool = pool

	bufferSize := int(config.BlockSize * int64(config.BlockCount))

	var wg sync.WaitGroup
	for i := 0; i < config.MaxConcurrentFiles; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()

			buffer := make([]byte, bufferSize)
			log.FromContext(ctx).V(2).Info("initializeWorkerPool: worker started",
				"workerID", workerID, "bufferSize", bufferSize)

			for {
				select {
				case filePath, ok := <-pool.workQueue:
					if !ok {
						log.FromContext(ctx).V(2).Info("initializeWorkerPool: worker exiting (channel closed)",
							"workerID", workerID)
						return
					}

					startTime := time.Now()
					err := prefetchFile(pool.shutdownCtx, filePath, buffer)
					duration := time.Since(startTime)

					if err != nil {
						log.FromContext(ctx).Error(err, "initializeWorkerPool: worker failed to prefetch file",
							"workerID", workerID, "path", filePath, "duration", duration)
					} else {
						log.FromContext(ctx).V(1).Info("initializeWorkerPool: worker successfully prefetched file",
							"workerID", workerID, "path", filePath, "duration", duration)
					}

				case <-pool.shutdownCtx.Done():
					log.FromContext(ctx).V(2).Info("initializeWorkerPool: worker exiting (shutdown signal)",
						"workerID", workerID)
					return
				}
			}
		}(i)
	}

	go func() {
		wg.Wait()
		close(pool.workersDone)
		log.FromContext(ctx).Info("initializeWorkerPool: all workers exited")
	}()

	log.FromContext(ctx).Info("initializeWorkerPool: worker pool initialized",
		"workerCount", config.MaxConcurrentFiles)
	return nil
}

// PrefetchConfig holds configuration for file prefetching behavior.
type PrefetchConfig struct {
	Enabled            bool  `json:"enabled"`
	BlockSize          int64 `json:"blockSize,omitempty"`
	BlockCount         int   `json:"blockCount,omitempty"`
	MaxConcurrentFiles int   `json:"maxConcurrentFiles,omitempty"`
	WorkQueueSize      int   `json:"workQueueSize,omitempty"`
}

// SetDefaultsForFilePrefetching applies default values to unset prefetch configuration fields.
func (p *PrefetchConfig) SetDefaultsForFilePrefetching() {
	if p.BlockSize == 0 {
		p.BlockSize = 4 * 1024 * 1024
	}
	if p.BlockCount == 0 {
		p.BlockCount = 3
	}
	if p.MaxConcurrentFiles == 0 {
		p.MaxConcurrentFiles = 16
	}
	if p.WorkQueueSize == 0 {
		p.WorkQueueSize = 256
	}
}

// PrefetchWorkerPool holds the runtime state for the prefetch worker pool.
type PrefetchWorkerPool struct {
	workQueue    chan string
	workersDone  chan struct{}
	shutdownCtx  context.Context
	shutdownFn   context.CancelFunc
	shutdownOnce sync.Once
}

type prefetchPrerequestHandlerParameters struct {
	EngineKeysProviderPluginName string                `json:"engineKeysProviderPluginName"`
	KVFilePathBase               *KVFilePathBaseParams `json:"kvFilePathBase,omitempty"`
	PrefetchConfig               *PrefetchConfig       `json:"prefetchConfig,omitempty"`
}

var _ requestcontrol.PreRequest = &PrefetchPrerequestHandler{}

// PluginFactory defines the factory function for the PrefetchPrerequestHandler.
func PluginFactory(name string, rawParameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	parameters := prefetchPrerequestHandlerParameters{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' pre-request plugin - %w", PrefetchPrerequestHandlerType, err)
		}
	}

	if parameters.PrefetchConfig != nil {
		parameters.PrefetchConfig.SetDefaultsForFilePrefetching()
	}
	if parameters.KVFilePathBase != nil {
		parameters.KVFilePathBase.SetDefaults()
	}

	handler := NewPrefetchPrerequestHandler(handle, parameters.EngineKeysProviderPluginName, parameters.KVFilePathBase, parameters.PrefetchConfig).WithName(name)

	if err := initializeWorkerPool(handle.Context(), handler); err != nil {
		return nil, fmt.Errorf("failed to initialize worker pool for '%s' plugin - %w", name, err)
	}

	return handler, nil
}

// NewPrefetchPrerequestHandler initializes a new PrefetchPrerequestHandler.
func NewPrefetchPrerequestHandler(handle plugin.Handle, engineKeysProviderPluginName string, kvFilePathBase *KVFilePathBaseParams, prefetchConfig *PrefetchConfig) *PrefetchPrerequestHandler {
	return &PrefetchPrerequestHandler{
		typedName:                    plugin.TypedName{Type: PrefetchPrerequestHandlerType},
		handle:                       handle,
		engineKeysProviderPluginName: engineKeysProviderPluginName,
		kvFilePathBase:               kvFilePathBase,
		prefetchConfig:               prefetchConfig,
	}
}

// PrefetchPrerequestHandler is a PreRequest plugin.
type PrefetchPrerequestHandler struct {
	typedName                    plugin.TypedName
	handle                       plugin.Handle
	engineKeysProviderPluginName string
	kvFilePathBase               *KVFilePathBaseParams
	prefetchConfig               *PrefetchConfig
	workerPool                   *PrefetchWorkerPool
}

// TypedName returns the typed name of the plugin.
func (p *PrefetchPrerequestHandler) TypedName() plugin.TypedName {
	return p.typedName
}

// WithName sets the name of the plugin.
func (p *PrefetchPrerequestHandler) WithName(name string) *PrefetchPrerequestHandler {
	p.typedName.Name = name
	return p
}

// submitForPrefetch enqueues each path onto the worker queue and returns the
// number of paths submitted and skipped. It runs on the request's critical
// path, so it never blocks waiting on the worker pool: a full queue drops the
// path immediately. A skipped path is a benign warm-up miss (the backend reads
// it cold later), never a request failure, so skipped is the queue-saturation
// signal.
func (p *PrefetchPrerequestHandler) submitForPrefetch(ctx context.Context, requestID string, paths []string) (submitted, skipped int) {
	for _, path := range paths {
		select {
		case p.workerPool.workQueue <- path:
			submitted++
		default:
			skipped++
			log.FromContext(ctx).V(logging.DEBUG).Info("PreRequest: queue full, skipping file",
				"requestId", requestID, "path", path)
		}
	}
	return submitted, skipped
}

// PreRequest logs engine keys and submits matching KV cache files for best-effort prefetching.
func (p *PrefetchPrerequestHandler) PreRequest(ctx context.Context, request *scheduling.InferenceRequest, schedulingResult *scheduling.SchedulingResult) {
	_ = schedulingResult

	if request == nil {
		return
	}

	log.FromContext(ctx).Info("prefetch-prerequest-handler PreRequest triggered", "requestId", request.RequestID,
		"engineKeysProviderPluginName", p.engineKeysProviderPluginName, "handleNil", p.handle == nil)

	if p.engineKeysProviderPluginName != "" && p.handle != nil {
		rawPlugin := p.handle.Plugin(p.engineKeysProviderPluginName)
		if rawPlugin != nil {
			if keysProvider, ok := rawPlugin.(*preciseproducer.Producer); ok {
				log.FromContext(ctx).Info("PreRequest: accessing engine-keys from provider",
					"requestId", request.RequestID, "provider", p.engineKeysProviderPluginName)

				hashStart := time.Now()
				engineKeys, digests, err := keysProvider.GetEngineKeysAndDigestsForRequest(ctx, request)
				hashDuration := time.Since(hashStart)
				if err != nil {
					log.FromContext(ctx).Error(err, "PreRequest: GetEngineKeysAndDigestsForRequest failed",
						"requestId", request.RequestID, "provider", p.engineKeysProviderPluginName)
					return
				}

				log.FromContext(ctx).V(logging.DEBUG).Info("PreRequest: engine-keys/digests computed",
					"requestId", request.RequestID, "digestCount", len(digests), "hashDuration", hashDuration)

				if len(digests) == 0 {
					log.FromContext(ctx).Info("PreRequest: no block digests for request, skipping prefetch (prompt likely shorter than one KV block)",
						"requestId", request.RequestID, "provider", p.engineKeysProviderPluginName,
						"engineKeyCount", len(engineKeys), "digestCount", 0)
					return
				}

				log.FromContext(ctx).Info("PreRequest: digests for request",
					"requestId", request.RequestID, "provider", p.engineKeysProviderPluginName,
					"engineKeyCount", len(engineKeys), "digestCount", len(digests))

				if p.kvFilePathBase != nil && p.kvFilePathBase.IsSet() {
					base := p.kvFilePathBase
					totalRanks := base.TpSize * base.PpSize * base.PcpSize * base.DcpSize
					filesPerRank := (len(digests) + base.GpuBlocksPerFile - 1) / base.GpuBlocksPerFile
					allFilePaths := make([]string, 0, filesPerRank*totalRanks)

					for rank := 0; rank < totalRanks; rank++ {
						fullPaths := digestsToFilePaths(base, rank, digests)
						if fullPaths == nil {
							log.FromContext(ctx).Info("PreRequest: not enough block digests to form an aggregated KV cache file yet, skipping prefetch",
								"requestId", request.RequestID, "rank", rank,
								"digestCount", len(digests), "gpuBlocksPerFile", base.GpuBlocksPerFile)
							return
						}
						allFilePaths = append(allFilePaths, fullPaths...)
						log.FromContext(ctx).Info("PreRequest: KV-cache file paths for rank",
							"requestId", request.RequestID, "rank", rank,
							"gpuBlocksPerFile", base.GpuBlocksPerFile, "paths", fullPaths)
					}

					if p.workerPool != nil && p.workerPool.workQueue != nil && p.prefetchConfig != nil && p.prefetchConfig.Enabled {
						log.FromContext(ctx).Info("PreRequest: submitting files for prefetch",
							"requestId", request.RequestID, "fileCount", len(allFilePaths))

						submitted, skipped := p.submitForPrefetch(ctx, request.RequestID, allFilePaths)

						log.FromContext(ctx).Info("PreRequest: prefetch submission complete",
							"requestId", request.RequestID, "submitted", submitted, "skipped", skipped)
					}
				} else {
					log.FromContext(ctx).Info("PreRequest: kvFilePathBase not set (needs rootDir+modelName+digest), skipping prefetch",
						"requestId", request.RequestID, "provider", p.engineKeysProviderPluginName,
						"kvFilePathBaseNil", p.kvFilePathBase == nil,
						"isSet", p.kvFilePathBase != nil && p.kvFilePathBase.IsSet())
				}
			} else {
				log.FromContext(ctx).Info("PreRequest: plugin found but is not a precise-prefix-cache producer",
					"requestId", request.RequestID, "plugin", p.engineKeysProviderPluginName)
			}
		} else {
			registeredNames := make([]string, 0, len(p.handle.GetAllPluginsWithNames()))
			for name := range p.handle.GetAllPluginsWithNames() {
				registeredNames = append(registeredNames, name)
			}
			log.FromContext(ctx).Info("PreRequest: engine-keys provider plugin not found",
				"requestId", request.RequestID, "provider", p.engineKeysProviderPluginName, "registeredPluginNames", registeredNames)
		}
	} else if p.engineKeysProviderPluginName != "" {
		log.FromContext(ctx).Info("PreRequest: engine-keys provider configured but handle unavailable",
			"requestId", request.RequestID, "provider", p.engineKeysProviderPluginName)
	}
}
