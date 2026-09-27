/*
Copyright 2026 The Kubernetes Authors.

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

package config

import (
	"context"
	"crypto/sha256"
	"os"
	"path/filepath"
	"time"

	"github.com/fsnotify/fsnotify"
	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
)

const (
	configWatchDebounce = 250 * time.Millisecond
	configWatchPoll     = 30 * time.Second
)

// FileAttempt contains a file read or watch error.
type FileAttempt struct {
	Content    []byte
	Hash       [sha256.Size]byte
	Err        error
	WatchError bool
}

// FileWatcher watches a path while reopening it for every read.
type FileWatcher struct {
	path        string
	watcher     *fsnotify.Watcher
	watchEvents <-chan fsnotify.Event
	watchErrors <-chan error
	handler     func(FileAttempt)
	lastHash    [sha256.Size]byte
	debounce    time.Duration
	poll        time.Duration
}

// NewFileWatcher watches the path's parent directory. initial is the startup file content.
func NewFileWatcher(path string, initial []byte, handler func(FileAttempt)) (*FileWatcher, error) {
	w, err := fsnotify.NewWatcher()
	if err != nil {
		return nil, err
	}
	if err := w.Add(filepath.Dir(path)); err != nil {
		_ = w.Close()
		return nil, err
	}
	return &FileWatcher{
		path: path, watcher: w, watchEvents: w.Events, watchErrors: w.Errors, handler: handler, lastHash: sha256.Sum256(initial),
		debounce: configWatchDebounce, poll: configWatchPoll,
	}, nil
}

// Run blocks until ctx is cancelled.
func (w *FileWatcher) Run(ctx context.Context) {
	defer w.watcher.Close()
	logger := log.FromContext(ctx).WithValues("path", w.path)
	poll := time.NewTicker(w.poll)
	defer poll.Stop()
	if ctx.Err() != nil {
		return
	}
	w.read(logger)

	var timer *time.Timer
	var timerC <-chan time.Time
	debounce := func() {
		if timer == nil {
			timer = time.NewTimer(w.debounce)
		} else {
			if !timer.Stop() {
				select {
				case <-timer.C:
				default:
				}
			}
			timer.Reset(w.debounce)
		}
		timerC = timer.C
	}
	defer func() {
		if timer != nil {
			timer.Stop()
		}
	}()

	events := w.watchEvents
	errors := w.watchErrors
	for {
		select {
		case <-ctx.Done():
			return
		case event, ok := <-events:
			if !ok {
				events = nil
				continue
			}
			logger.V(logging.DEBUG).Info("config file watch event", "event", event.String())
			debounce()
		case err, ok := <-errors:
			if !ok {
				errors = nil
				continue
			}
			w.handler(FileAttempt{Err: err, WatchError: true})
		case <-poll.C:
			if ctx.Err() != nil {
				return
			}
			w.read(logger)
		case <-timerC:
			timerC = nil
			if ctx.Err() != nil {
				return
			}
			w.read(logger)
		}
	}
}

func (w *FileWatcher) read(logger logr.Logger) {
	content, err := os.ReadFile(w.path)
	if err != nil {
		w.handler(FileAttempt{Err: err})
		return
	}
	hash := sha256.Sum256(content)
	if hash == w.lastHash {
		logger.V(logging.DEBUG).Info("config file content is unchanged")
		return
	}
	w.lastHash = hash
	w.handler(FileAttempt{Content: content, Hash: hash})
}
