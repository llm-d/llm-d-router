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

package tokenization

import (
	"context"
	"fmt"
	"sync"

	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"

	"github.com/llm-d/llm-d-kv-cache-manager/pkg/tokenization/prefixstore"
)

const (
	defaultWorkers               = 5
	defaultMinPrefixOverlapRatio = 0.8
)

// Config holds the configuration for the TokenizationPool.
type Config struct {
	// Number of worker goroutines for processing tokenization tasks.
	WorkersCount int `json:"workersCount"`
	// Minimum overlap ratio to skip full tokenization and use cached prefix tokens.
	MinPrefixOverlapRatio float64 `json:"minPrefixOverlapRatio"`

	LocalTokenizerConfig *LocalTokenizerConfig `json:"local,omitempty"`
	UdsTokenizerConfig   *UdsTokenizerConfig   `json:"uds,omitempty"`
	HFTokenizerConfig    *HFTokenizerConfig    `json:"hf,omitempty"`
}

// DefaultConfig returns a default configuration for the TokenizationPool.
func DefaultConfig() (*Config, error) {
	localTokenizerConfig, err := DefaultLocalTokenizerConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to create default local tokenizer config: %w", err)
	}

	return &Config{
		WorkersCount:          defaultWorkers,
		MinPrefixOverlapRatio: defaultMinPrefixOverlapRatio,
		HFTokenizerConfig:     DefaultHFTokenizerConfig(),
		LocalTokenizerConfig:  localTokenizerConfig,
	}, nil
}

// tokenizationResponse holds the result of a tokenization operation.
type tokenizationResponse struct {
	Tokens []uint32
}

// Task represents a unit of work for tokenizing a prompt.
type Task struct {
	Prompt    string
	ModelName string
	ResultCh  chan<- tokenizationResponse // nil => fire-and-forget
}

// Pool encapsulates the queue, worker pool, and token indexer.
type Pool struct {
	workers int
	queue   workqueue.TypedRateLimitingInterface[Task]
	wg      sync.WaitGroup
	indexer prefixstore.Indexer

	// Tokenizer caches multiple tokenizers in memory.
	// The cache is shared between all pool workers. Since each tokenizer
	// is immutable, Encode calls are safe for concurrent use without locks.
	tokenizer Tokenizer

	// Minimum overlap ratio to skip full tokenization and use cached prefix tokens.
	minPrefixOverlapRatio float64
}

// NewTokenizationPool initializes a TokenizationPool with the specified number
// of workers and the provided Indexer.
func NewTokenizationPool(config *Config, store prefixstore.Indexer) (*Pool, error) {
	if config == nil {
		var err error
		config, err = DefaultConfig()
		if err != nil {
			return nil, fmt.Errorf("failed to create default config: %w", err)
		}
	}

	tokenizers := make([]Tokenizer, 0, 2)

	if config.LocalTokenizerConfig.IsEnabled() {
		localTokenizer, err := NewCachedLocalTokenizer(*config.LocalTokenizerConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to create local tokenizer: %w", err)
		}
		tokenizers = append(tokenizers, localTokenizer)
	}

	if config.UdsTokenizerConfig.IsEnabled() {
		udsTokenizer, err := NewUdsTokenizer(config.UdsTokenizerConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to create UDS tokenizer: %w", err)
		}
		tokenizers = append(tokenizers, udsTokenizer)
	}

	if config.HFTokenizerConfig.IsEnabled() {
		hfTokenizer, err := NewCachedHFTokenizer(config.HFTokenizerConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to create HuggingFace tokenizer: %w", err)
		}
		tokenizers = append(tokenizers, hfTokenizer)
	}

	return &Pool{
		workers:               config.WorkersCount,
		queue:                 workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[Task]()),
		indexer:               store,
		tokenizer:             &CompositeTokenizer{Tokenizers: tokenizers},
		minPrefixOverlapRatio: config.MinPrefixOverlapRatio,
	}, nil
}

// EnqueueTokenization enqueues a new tokenization task.
// This method only enqueues the task and does not start processing it.
func (pool *Pool) EnqueueTokenization(prompt, modelName string) {
	task := Task{
		Prompt:    prompt,
		ModelName: modelName,
	}
	pool.queue.Add(task)
}

// Tokenize queues a task and blocks until the final result is available.
func (pool *Pool) Tokenize(prompt, modelName string) []uint32 {
	resultCh := make(chan tokenizationResponse, 1)
	pool.queue.Add(Task{
		Prompt:    prompt,
		ModelName: modelName,
		ResultCh:  resultCh,
	})

	res := <-resultCh
	tokens := res.Tokens
	return tokens
}

// Run launches worker goroutines that process tasks until the context is
// cancelled.
func (pool *Pool) Run(ctx context.Context) {
	for i := 0; i < pool.workers; i++ {
		pool.wg.Add(1)
		go pool.workerLoop(i)
	}

	<-ctx.Done()

	pool.queue.ShutDown()
	pool.wg.Wait()
}

// workerLoop is the main processing loop for each worker.
func (pool *Pool) workerLoop(_ int) {
	defer pool.wg.Done()
	for {
		task, shutdown := pool.queue.Get()
		if shutdown {
			return
		}

		// Process the task.
		if err := pool.processTask(task); err == nil {
			pool.queue.Forget(task)
		} else {
			pool.queue.AddRateLimited(task)
		}
		pool.queue.Done(task)
	}
}

// processTask tokenizes the prompt and updates the indexer.
// It sends exactly one response (success or error) if ResultCh is provided.
func (pool *Pool) processTask(task Task) error {
	tokenIDs, overlapRatio := pool.indexer.FindLongestContainedTokens(task.Prompt, task.ModelName)

	// if the overlap ratio is low, get the full tokenization
	if overlapRatio < pool.minPrefixOverlapRatio {
		tokens, offsets, err := pool.tokenizer.Encode(task.Prompt, task.ModelName)
		if err != nil {
			klog.Error(err, "failed to encode tokens", "prompt", task.Prompt, "modelName", task.ModelName)
			return err
		}

		// update the indexer with the new tokenization
		if e := pool.indexer.AddTokenization(task.ModelName, task.Prompt, tokens, offsets); e != nil {
			err = fmt.Errorf("tokenization failed for model %s: %w", task.ModelName, e)
			return err
		}

		tokenIDs = tokens
	}

	// On success, send the response if a channel is provided and close the channel.
	if task.ResultCh != nil {
		resp := tokenizationResponse{
			Tokens: tokenIDs,
		}
		task.ResultCh <- resp
		close(task.ResultCh)
	}

	return nil
}
