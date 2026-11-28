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

package datalayer

import (
	"context"
	"sync"
	"time"

	"github.com/go-logr/logr"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

// maxImportRetryBackoff caps the delay before a failed import is retried.
const maxImportRetryBackoff = 2 * time.Second

// endpointGen is one registration of an endpoint. registerEndpoint creates it;
// beginDelete cancels it and waits for its running imports to return.
type endpointGen struct {
	ctx     context.Context
	cancel  context.CancelFunc
	mu      sync.Mutex
	drain   bool           // set by beginDelete; no import of this generation starts afterwards
	running sync.WaitGroup // imports in flight for this generation
	drained chan struct{}  // closed after the delete event reached the plugins
}

func newEndpointGen() *endpointGen {
	ctx, cancel := context.WithCancel(context.Background())
	return &endpointGen{ctx: ctx, cancel: cancel, drained: make(chan struct{})}
}

// startImport registers an import of this generation, or reports false once
// the generation is draining.
func (g *endpointGen) startImport() bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.drain {
		return false
	}
	g.running.Add(1)
	return true
}

type importJob struct {
	gen       *endpointGen
	peer      any
	version   uint64
	notBefore time.Time
	backoff   time.Duration
}

type importedVersion struct {
	gen     *endpointGen
	version uint64
}

// importWorker folds peer snapshots of one contributor into its plugin, one
// endpoint at a time, off the request path.
type importWorker struct {
	spec          fwkdl.CrossReplicaSpec
	importer      fwkdl.CrossReplicaImporter
	reader        fwkdl.CrossReplicaPeerReader
	retryInterval time.Duration

	mu       sync.Mutex
	pending  map[string]*importJob
	running  map[string]importedVersion // job taken by the worker, not finished yet
	imported map[string]importedVersion
	wake     chan struct{}
}

func newImportWorker(spec fwkdl.CrossReplicaSpec, importer fwkdl.CrossReplicaImporter, reader fwkdl.CrossReplicaPeerReader, retryInterval time.Duration) *importWorker {
	return &importWorker{
		spec:          spec,
		importer:      importer,
		reader:        reader,
		retryInterval: retryInterval,
		pending:       make(map[string]*importJob),
		running:       make(map[string]importedVersion),
		imported:      make(map[string]importedVersion),
		wake:          make(chan struct{}, 1),
	}
}

// enqueue queues the peer snapshot prepared by a successful Set of endpointID
// in generation gen. A job of the same generation that the worker has not
// taken yet is kept: it may hold peer fields that have expired since, and the
// newer version is enqueued by a later publish because it differs from the
// imported one.
func (w *importWorker) enqueue(endpointID string, gen *endpointGen) {
	peer, version, ok := w.reader.PeerSnapshot(w.spec, endpointID)
	if !ok {
		return
	}
	w.mu.Lock()
	current := importedVersion{gen: gen, version: version}
	if w.imported[endpointID] == current || w.running[endpointID] == current {
		w.mu.Unlock()
		return
	}
	// Imports only add, so an untaken job of this generation is kept and the
	// newer version is enqueued by a publish after the job is taken.
	if job, ok := w.pending[endpointID]; ok && job.gen == gen {
		w.mu.Unlock()
		return
	}
	w.pending[endpointID] = &importJob{gen: gen, peer: peer, version: version}
	w.mu.Unlock()
	select {
	case w.wake <- struct{}{}:
	default:
	}
}

// forget drops the bookkeeping of endpointID. Called once its generation has
// no import running.
func (w *importWorker) forget(endpointID string) {
	w.mu.Lock()
	delete(w.pending, endpointID)
	delete(w.running, endpointID)
	delete(w.imported, endpointID)
	w.mu.Unlock()
}

// take removes and returns a job whose retry time has passed. Otherwise it
// returns how long until the earliest retry, or zero when nothing is pending.
func (w *importWorker) take(now time.Time) (string, *importJob, time.Duration) {
	w.mu.Lock()
	defer w.mu.Unlock()
	var wait time.Duration
	for endpointID, job := range w.pending {
		if !job.notBefore.After(now) {
			delete(w.pending, endpointID)
			w.running[endpointID] = importedVersion{gen: job.gen, version: job.version}
			return endpointID, job, 0
		}
		if d := job.notBefore.Sub(now); wait == 0 || d < wait {
			wait = d
		}
	}
	return "", nil, wait
}

func (w *importWorker) run(ctx context.Context, logger logr.Logger) {
	for ctx.Err() == nil {
		endpointID, job, wait := w.take(time.Now())
		if job != nil {
			w.runJob(logger, endpointID, job)
			continue
		}
		var retry <-chan time.Time
		var timer *time.Timer
		if wait > 0 {
			timer = time.NewTimer(wait)
			retry = timer.C
		}
		select {
		case <-ctx.Done():
			if timer != nil {
				timer.Stop()
			}
			return
		case <-w.wake:
		case <-retry:
		}
		if timer != nil {
			timer.Stop()
		}
	}
}

func (w *importWorker) runJob(logger logr.Logger, endpointID string, job *importJob) {
	gen := job.gen
	if !gen.startImport() {
		w.mu.Lock()
		delete(w.running, endpointID)
		w.mu.Unlock()
		return
	}
	err := w.importer.ImportCrossReplica(gen.ctx, endpointID, job.peer)
	gen.running.Done()

	w.mu.Lock()
	defer w.mu.Unlock()
	delete(w.running, endpointID)
	switch {
	case gen.ctx.Err() != nil:
		// The generation was deleted; beginDelete clears its bookkeeping.
	case err != nil:
		// Retry this snapshot ahead of any newer one: it may hold peer fields
		// that have expired in the syncer since.
		job.backoff = min(max(2*job.backoff, w.retryInterval), maxImportRetryBackoff)
		job.notBefore = time.Now().Add(job.backoff)
		w.pending[endpointID] = job
		if v := logger.V(logging.DEBUG); v.Enabled() {
			v.Info("cross-replica import failed, retrying", "key", w.spec.StateKey, "endpoint", endpointID, "error", err, "retryIn", job.backoff)
		}
	default:
		w.imported[endpointID] = importedVersion{gen: gen, version: job.version}
	}
}
