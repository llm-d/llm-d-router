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

// Package outlensketch is an online, CPU-only estimator of a request's output
// length, used as a more precise alternative to the request-time signal rules.
// Before generation it maps a prompt to a representative output-token magnitude;
// at end-of-stream it learns from the observed completion length. It needs no
// training and keeps adapting to the live workload.
//
// Pipeline: window the prompt to a bounded set of structure-bearing turns, embed
// the window with a static table-lookup embedding (no neural forward pass), assign
// the embedding to one of K online k-means centroids (the AdaptiveKey), fold that
// cluster with a signature of the request-time signals to form the key, and read
// that key's count-decaying output-length histogram at a high quantile. A key that
// has not yet seen enough traffic abstains, so the estimator self-heals from a cold
// start as traffic accumulates.
package outlensketch

import (
	"encoding/binary"
	"errors"
	"fmt"

	"github.com/cespare/xxhash/v2"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

// Config carries the sketch tunables. The defaults match the offline study's
// shipped configuration; ModelDir is required and has no default.
type Config struct {
	// ModelDir is the mounted directory holding the static-embedding model
	// (vocab.txt + model.safetensors). Required.
	ModelDir string
	// Centroids is the number of k-means centroids (the AdaptiveKey resolution).
	Centroids int
	// DecayHalfLifeRequests is the per-key count-decay half-life N: a bin's weight
	// halves after N observations of that key.
	DecayHalfLifeRequests int
	// Quantile is the over-provision quantile read from each key's histogram.
	Quantile float64
	// WarmupSamples is the minimum effective count before a key is trusted; below
	// it the key abstains.
	WarmupSamples float64
	// MaxKeys bounds the histogram store with an LRU safety cap.
	MaxKeys int
	// WarmupBatch is the observation count that triggers a centroid fit attempt.
	WarmupBatch int
	// LloydIterations is the number of Lloyd iterations per centroid fit.
	LloydIterations int
	// MinCentroidMembers is the per-centroid member floor for the readiness check.
	MinCentroidMembers int
	// MinCentroidCoverage is the fraction of centroids that must meet the floor
	// before the AdaptiveKey goes online.
	MinCentroidCoverage float64
	// Seed makes the Lloyd initialization deterministic.
	Seed int64
}

// DefaultConfig returns the shipped sketch configuration (ModelDir unset).
func DefaultConfig() Config {
	return Config{
		Centroids:             64,
		DecayHalfLifeRequests: 200,
		Quantile:              0.85,
		WarmupSamples:         5,
		MaxKeys:               100_000,
		WarmupBatch:           500,
		LloydIterations:       25,
		MinCentroidMembers:    5,
		MinCentroidCoverage:   0.9,
		Seed:                  5,
	}
}

// Sketch is the assembled estimator: embedder, AdaptiveKey, and histogram store.
// It is safe for concurrent use.
type Sketch struct {
	embedder *embedder
	key      *adaptiveKMeans
	hist     *histogramStore
}

// validate rejects a configuration that would make the sketch misbehave silently.
func (cfg Config) validate() error {
	switch {
	case cfg.ModelDir == "":
		return errors.New("modelDir is required")
	case cfg.Quantile <= 0 || cfg.Quantile > 1:
		return fmt.Errorf("quantile must be in (0, 1], got %v", cfg.Quantile)
	case cfg.Centroids < 1:
		return fmt.Errorf("centroids must be positive, got %d", cfg.Centroids)
	case cfg.DecayHalfLifeRequests < 1:
		return fmt.Errorf("decayHalfLifeRequests must be positive, got %d", cfg.DecayHalfLifeRequests)
	case cfg.WarmupSamples < 1:
		return fmt.Errorf("warmupSamples must be positive, got %v", cfg.WarmupSamples)
	case cfg.WarmupBatch < 1:
		return fmt.Errorf("warmupBatch must be positive, got %d", cfg.WarmupBatch)
	case cfg.MaxKeys < 1:
		return fmt.Errorf("maxKeys must be positive, got %d", cfg.MaxKeys)
	case cfg.LloydIterations < 1:
		return fmt.Errorf("lloydIterations must be positive, got %d", cfg.LloydIterations)
	case cfg.MinCentroidMembers < 1:
		return fmt.Errorf("minCentroidMembers must be positive, got %d", cfg.MinCentroidMembers)
	case cfg.MinCentroidCoverage <= 0 || cfg.MinCentroidCoverage > 1:
		return fmt.Errorf("minCentroidCoverage must be in (0, 1], got %v", cfg.MinCentroidCoverage)
	}
	return nil
}

// New builds a Sketch, loading the embedding model from cfg.ModelDir.
func New(cfg Config) (*Sketch, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	emb, err := newEmbedder(cfg.ModelDir)
	if err != nil {
		return nil, err
	}
	key := newAdaptiveKMeans(adaptiveKMeansConfig{
		k:           cfg.Centroids,
		dim:         emb.dim,
		iters:       cfg.LloydIterations,
		warmupN:     cfg.WarmupBatch,
		minMembers:  cfg.MinCentroidMembers,
		minCoverage: cfg.MinCentroidCoverage,
		seed:        cfg.Seed,
	})
	hist := newHistogramStore(cfg.DecayHalfLifeRequests, cfg.Quantile, cfg.WarmupSamples, cfg.MaxKeys)
	return &Sketch{embedder: emb, key: key, hist: hist}, nil
}

// Prediction is a request's output-length estimate together with the featurization
// it was derived from (the prompt embedding and the histogram key). Holding the
// embedding lets the caller feed the same features back to Learn at end-of-stream
// without re-windowing and re-embedding the prompt, and keeps the learned key
// identical to the predicted one even if the centroids drift in between.
type Prediction struct {
	// Magnitude is the estimated output-token count; meaningful only when OK.
	Magnitude int64
	// OK is false when the key is cold (abstain -- the caller applies its own
	// fallback).
	OK bool

	valid     bool // set for a prediction derived from a non-nil request body
	emb       []float32
	key       uint64
	hasPrompt bool
}

// Predict featurizes the request and reads its output-length estimate. The returned
// Prediction also carries the features, to be passed back to Learn at end-of-stream.
func (s *Sketch) Predict(body *fwkrh.InferenceRequestBody) Prediction {
	if body == nil {
		return Prediction{}
	}
	emb, key, hasPrompt := s.features(body)
	magnitude, ok := s.hist.predict(key)
	return Prediction{Magnitude: magnitude, OK: ok, valid: true, emb: emb, key: key, hasPrompt: hasPrompt}
}

// Learn folds a completed request's output length into the sketch using the
// features captured at Predict time: it nudges the AdaptiveKey centroids toward the
// prompt embedding and updates the key's output-length histogram. A zero-value
// Prediction (nil request body) or a non-positive length is ignored.
func (s *Sketch) Learn(p Prediction, completionTokens int64) {
	if !p.valid || completionTokens <= 0 {
		return
	}
	if p.hasPrompt {
		s.key.observe(p.emb)
	}
	s.hist.observe(p.key, completionTokens)
}

// features computes the prompt embedding and the histogram key for a request. When
// the request carries no chat prompt, or while the AdaptiveKey is still warming up,
// the key is the signal signature alone; once the centroids are ready it is the
// cluster folded with the signal signature.
func (s *Sketch) features(body *fwkrh.InferenceRequestBody) (emb []float32, key uint64, hasPrompt bool) {
	windowed := windowedPrompt(body)
	sig := signalSig(body)
	if windowed == "" {
		return nil, sig, false
	}
	emb = s.embedder.encode(windowed)
	if cluster, ready := s.key.cluster(emb); ready {
		return emb, combineKey(cluster, sig), true
	}
	return emb, sig, true
}

// combineKey folds a cluster id and a signal signature into the histogram key.
func combineKey(cluster int, sig uint64) uint64 {
	var buf [16]byte
	binary.LittleEndian.PutUint64(buf[0:8], uint64(cluster))
	binary.LittleEndian.PutUint64(buf[8:16], sig)
	return xxhash.Sum64(buf[:])
}
