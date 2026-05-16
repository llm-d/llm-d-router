package prefetch

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/scheduling/scorer/preciseprefixcache"
)

const (
	// PrefetchPrerequestHandlerType is the type of the PrefetchPrerequestHandler.
	PrefetchPrerequestHandlerType = "prefetch-prerequest-handler"
)

// EngineKeyToFilenamePathSuffix converts an engine-key (uint64 block hash) to
// the path suffix used by llm-d-fs-connector: hhh/hh/<16hex>.bin.
func EngineKeyToFilenamePathSuffix(engineKey uint64) string {
	const mask64 = (1 << 64) - 1
	blockHashHex := fmt.Sprintf("%016x", engineKey&mask64)
	sub1, sub2 := blockHashHex[:3], blockHashHex[3:5]
	return sub1 + "/" + sub2 + "/" + blockHashHex + ".bin"
}

// KVFilePathBaseParams holds parameters to build the base path for KV-cache files.
type KVFilePathBaseParams struct {
	RootDir          string `json:"rootDir"`
	ModelParentDir   string `json:"modelParentDir,omitempty"`
	ModelName        string `json:"modelName"`
	GpuBlockSize     int    `json:"gpuBlockSize"`
	GpuBlocksPerFile int    `json:"gpuBlocksPerFile"`
	TpSize           int    `json:"tpSize"`
	PpSize           int    `json:"ppSize"`
	PcpSize          int    `json:"pcpSize"`
	Rank             int    `json:"rank"`
	Dtype            string `json:"dtype"`
}

// IsSet returns true if base path can be built.
func (b *KVFilePathBaseParams) IsSet() bool {
	return b != nil && b.RootDir != "" && b.ModelName != ""
}

// SetDefaults applies default values to unset KVFilePathBaseParams fields.
func (b *KVFilePathBaseParams) SetDefaults() {
	if b.GpuBlockSize < 1 {
		b.GpuBlockSize = 64
	}
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
}

// BasePath returns the base directory path for KV-cache files.
func (b *KVFilePathBaseParams) BasePath() string {
	modelSegments := []string{b.RootDir}
	if b.ModelParentDir != "" {
		modelSegments = append(modelSegments, b.ModelParentDir)
	}
	modelSegments = append(modelSegments, b.ModelName,
		fmt.Sprintf("block_size_%d_blocks_per_file_%d", b.GpuBlockSize, b.GpuBlocksPerFile),
		fmt.Sprintf("tp_%d_pp_size_%d_pcp_size_%d", b.TpSize, b.PpSize, b.PcpSize),
		fmt.Sprintf("rank_%d", b.Rank),
		b.Dtype)
	return filepath.Join(modelSegments...)
}

// EngineKeyToFullPath returns the complete file path for an engine key.
func EngineKeyToFullPath(base *KVFilePathBaseParams, engineKey uint64) string {
	suffix := EngineKeyToFilenamePathSuffix(engineKey)
	return filepath.Join(base.BasePath(), filepath.FromSlash(suffix))
}

// EngineKeysToFilePaths returns the file paths to prefetch for the given engine keys.
func EngineKeysToFilePaths(base *KVFilePathBaseParams, engineKeys []uint64) []string {
	n := base.GpuBlocksPerFile
	if n <= 1 {
		paths := make([]string, len(engineKeys))
		for i, ek := range engineKeys {
			paths[i] = EngineKeyToFullPath(base, ek)
		}
		return paths
	}

	paths := make([]string, 0, (len(engineKeys)+n-1)/n)
	for i := n - 1; i < len(engineKeys); i += n {
		paths = append(paths, EngineKeyToFullPath(base, engineKeys[i]))
	}
	return paths
}

func prefetchFile(ctx context.Context, filePath string, buffer []byte) error {
	file, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	n, err := file.Read(buffer)
	if err != nil && err != io.EOF {
		return fmt.Errorf("failed to read file: %w", err)
	}

	if err == io.EOF || n < len(buffer) {
		log.FromContext(ctx).V(4).Info("prefetchFile: read partial file",
			"path", filePath, "requestedBytes", len(buffer), "actualBytes", n)
	} else {
		log.FromContext(ctx).V(4).Info("prefetchFile: read complete",
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
	QueueTimeout       int   `json:"queueTimeout,omitempty"`
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
func PluginFactory(name string, rawParameters json.RawMessage, handle plugin.Handle) (plugin.Plugin, error) {
	parameters := prefetchPrerequestHandlerParameters{}
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &parameters); err != nil {
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
			if keysProvider, ok := rawPlugin.(*preciseprefixcache.Scorer); ok {
				log.FromContext(ctx).Info("PreRequest: accessing engine-keys from provider",
					"requestId", request.RequestID, "provider", p.engineKeysProviderPluginName)

				engineKeys, err := keysProvider.GetEngineKeysForRequest(ctx, request)
				if err != nil {
					log.FromContext(ctx).Error(err, "PreRequest: GetEngineKeysForRequest failed",
						"requestId", request.RequestID, "provider", p.engineKeysProviderPluginName)
					return
				}

				if len(engineKeys) == 0 {
					return
				}

				validEngineKeys := make([]uint64, 0, len(engineKeys))
				for _, ek := range engineKeys {
					if ek != 0 {
						validEngineKeys = append(validEngineKeys, ek)
					}
				}

				if emptyCount := len(engineKeys) - len(validEngineKeys); emptyCount > 0 {
					log.FromContext(ctx).Info("PreRequest: empty engine keys (0) received",
						"requestId", request.RequestID, "provider", p.engineKeysProviderPluginName,
						"emptyCount", emptyCount, "totalEngineKeys", len(engineKeys))
				}
				if len(validEngineKeys) == 0 {
					log.FromContext(ctx).Info("PreRequest: all engine-keys are 0 (no requestKey→engineKey mapping); skipping paths",
						"requestId", request.RequestID, "provider", p.engineKeysProviderPluginName)
					return
				}

				filenames := make([]string, len(validEngineKeys))
				for i, ek := range validEngineKeys {
					filenames[i] = EngineKeyToFilenamePathSuffix(ek)
				}
				log.FromContext(ctx).Info("PreRequest: engine-keys and filenames for request",
					"requestId", request.RequestID, "provider", p.engineKeysProviderPluginName,
					"engineKeys", validEngineKeys, "filenames", filenames)

				if p.kvFilePathBase != nil && p.kvFilePathBase.IsSet() {
					base := p.kvFilePathBase
					totalRanks := base.TpSize * base.PpSize * base.PcpSize
					filesPerRank := (len(validEngineKeys) + base.GpuBlocksPerFile - 1) / base.GpuBlocksPerFile
					allFilePaths := make([]string, 0, filesPerRank*totalRanks)

					for rank := 0; rank < totalRanks; rank++ {
						rankParams := *base
						rankParams.Rank = rank
						fullPaths := EngineKeysToFilePaths(&rankParams, validEngineKeys)
						allFilePaths = append(allFilePaths, fullPaths...)
						log.FromContext(ctx).Info("PreRequest: KV-cache file paths for rank",
							"requestId", request.RequestID, "rank", rank,
							"gpuBlocksPerFile", base.GpuBlocksPerFile, "paths", fullPaths)
					}

					if p.workerPool != nil && p.workerPool.workQueue != nil && p.prefetchConfig != nil && p.prefetchConfig.Enabled {
						log.FromContext(ctx).Info("PreRequest: submitting files for prefetch",
							"requestId", request.RequestID, "fileCount", len(allFilePaths),
							"queueTimeout", p.prefetchConfig.QueueTimeout)

						submitted := 0
						skipped := 0

						for _, path := range allFilePaths {
							if p.prefetchConfig.QueueTimeout > 0 {
								select {
								case p.workerPool.workQueue <- path:
									submitted++
								case <-time.After(time.Duration(p.prefetchConfig.QueueTimeout) * time.Millisecond):
									skipped++
									log.FromContext(ctx).V(1).Info("PreRequest: queue timeout, skipping file",
										"requestId", request.RequestID, "path", path, "timeout", p.prefetchConfig.QueueTimeout)
								}
							} else {
								p.workerPool.workQueue <- path
								submitted++
							}
						}

						log.FromContext(ctx).Info("PreRequest: prefetch submission complete",
							"requestId", request.RequestID, "submitted", submitted, "skipped", skipped)
					}
				}
			} else {
				log.FromContext(ctx).Info("PreRequest: plugin found but is not a precise-prefix-cache scorer",
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
