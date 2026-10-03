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
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/fsnotify/fsnotify"
	"github.com/stretchr/testify/require"
)

func newTestFileWatcher(t *testing.T, path, initial string) (*FileWatcher, <-chan FileAttempt) {
	t.Helper()
	attempts := make(chan FileAttempt, 8)
	watcher, err := NewFileWatcher(path, []byte(initial), func(attempt FileAttempt) { attempts <- attempt })
	require.NoError(t, err)
	return watcher, attempts
}

func runTestFileWatcher(t *testing.T, watcher *FileWatcher) func() {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		watcher.Run(ctx)
		close(done)
	}()
	stop := func() {
		cancel()
		select {
		case <-done:
		case <-time.After(time.Second):
			t.Error("watcher did not stop")
		}
	}
	t.Cleanup(stop)
	return stop
}

func requireFileContent(t *testing.T, attempts <-chan FileAttempt, want string, timeout time.Duration) {
	t.Helper()
	select {
	case attempt := <-attempts:
		require.NoError(t, attempt.Err)
		require.Equal(t, want, string(attempt.Content))
	case <-time.After(timeout):
		t.Fatalf("timed out waiting for config content %q", want)
	}
}

func TestFileWatcherDetectsWriteAndStops(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	require.NoError(t, os.WriteFile(path, []byte("one"), 0o600))
	watcher, attempts := newTestFileWatcher(t, path, "startup")
	stop := runTestFileWatcher(t, watcher)
	requireFileContent(t, attempts, "one", time.Second)

	require.NoError(t, os.WriteFile(path, []byte("two"), 0o600))
	requireFileContent(t, attempts, "two", 3*time.Second)
	stop()
}

func TestFileWatcherDetectsChangeBeforeRun(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	require.NoError(t, os.WriteFile(path, []byte("one"), 0o600))
	watcher, attempts := newTestFileWatcher(t, path, "one")
	watcher.poll = time.Hour
	watcher.debounce = time.Hour
	require.NoError(t, os.WriteFile(path, []byte("two"), 0o600))

	runTestFileWatcher(t, watcher)
	requireFileContent(t, attempts, "two", time.Second)
}

func TestFileWatcherDetectsConfigMapSymlinkSwap(t *testing.T) {
	for _, tc := range []struct {
		name      string
		poll      time.Duration
		eventOnly bool
	}{
		{name: "projected update", poll: 100 * time.Millisecond},
		{name: "watch event", poll: time.Hour, eventOnly: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			for _, version := range []string{"..data-1", "..data-2"} {
				require.NoError(t, os.Mkdir(filepath.Join(dir, version), 0o700))
			}
			require.NoError(t, os.WriteFile(filepath.Join(dir, "..data-1", "config.yaml"), []byte("one"), 0o600))
			require.NoError(t, os.WriteFile(filepath.Join(dir, "..data-2", "config.yaml"), []byte("two"), 0o600))
			require.NoError(t, os.Symlink("..data-1", filepath.Join(dir, "..data")))
			require.NoError(t, os.Symlink(filepath.Join("..data", "config.yaml"), filepath.Join(dir, "config.yaml")))

			watcher, attempts := newTestFileWatcher(t, filepath.Join(dir, "config.yaml"), "startup")
			watcher.poll = tc.poll
			var watchEvents chan fsnotify.Event
			if tc.eventOnly {
				// Inject the event to test handling independently of platform notification delivery.
				watchEvents = make(chan fsnotify.Event, 1)
				watcher.watchEvents = watchEvents
			}
			runTestFileWatcher(t, watcher)
			requireFileContent(t, attempts, "one", time.Second)
			require.NoError(t, os.Symlink("..data-2", filepath.Join(dir, "..data-new")))
			require.NoError(t, os.Rename(filepath.Join(dir, "..data-new"), filepath.Join(dir, "..data")))
			if tc.eventOnly {
				watchEvents <- fsnotify.Event{Name: filepath.Join(dir, "..data"), Op: fsnotify.Rename}
			}
			requireFileContent(t, attempts, "two", 3*time.Second)
		})
	}
}

func TestFileWatcherIgnoresDuplicateContent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	require.NoError(t, os.WriteFile(path, []byte("same"), 0o600))
	watcher, attempts := newTestFileWatcher(t, path, "startup")
	watchEvents := make(chan fsnotify.Event, 1)
	watcher.watchEvents = watchEvents
	runTestFileWatcher(t, watcher)
	requireFileContent(t, attempts, "same", time.Second)

	watchEvents <- fsnotify.Event{Name: path, Op: fsnotify.Write}
	select {
	case <-attempts:
		t.Fatal("duplicate content triggered a reload")
	case <-time.After(2 * configWatchDebounce):
	}
}

func TestFileWatcherPollsAfterEventChannelsClose(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	require.NoError(t, os.WriteFile(path, []byte("one"), 0o600))
	watcher, attempts := newTestFileWatcher(t, path, "startup")
	watcher.poll = 20 * time.Millisecond
	watcher.debounce = time.Hour
	runTestFileWatcher(t, watcher)
	requireFileContent(t, attempts, "one", time.Second)

	require.NoError(t, watcher.watcher.Close())
	require.NoError(t, os.WriteFile(path, []byte("two"), 0o600))
	requireFileContent(t, attempts, "two", time.Second)
}

func TestFileWatcherReportsWatchError(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	require.NoError(t, os.WriteFile(path, []byte("one"), 0o600))
	watcher, attempts := newTestFileWatcher(t, path, "one")
	watchErrors := make(chan error, 1)
	watcher.watchErrors = watchErrors
	runTestFileWatcher(t, watcher)

	watchErr := errors.New("watch failed")
	watchErrors <- watchErr
	select {
	case attempt := <-attempts:
		require.ErrorIs(t, attempt.Err, watchErr)
		require.True(t, attempt.WatchError)
	case <-time.After(time.Second):
		t.Fatal("watcher did not report error")
	}
}
