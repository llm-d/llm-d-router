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

package main

import (
	"io"
	"os"
	"path/filepath"
	"syscall"
	"testing"
)

const markerName = "test_running"

func TestWriteTestRunningMarker(t *testing.T) {
	t.Run("creates marker", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), markerName)
		if err := writeTestRunningMarker(path); err != nil {
			t.Fatal(err)
		}
		assertRunningMarker(t, path)
	})

	t.Run("replaces stale file", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), markerName)
		if err := os.WriteFile(path, []byte("stale"), 0600); err != nil {
			t.Fatal(err)
		}
		if err := writeTestRunningMarker(path); err != nil {
			t.Fatal(err)
		}
		assertRunningMarker(t, path)
	})

	t.Run("does not follow symlink", func(t *testing.T) {
		dir := t.TempDir()
		sentinel, err := os.CreateTemp(dir, "sentinel")
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = sentinel.Close() })
		if _, err := sentinel.Write([]byte("secret")); err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(dir, markerName)
		if err := os.Symlink(sentinel.Name(), path); err != nil {
			t.Fatal(err)
		}

		if err := writeTestRunningMarker(path); err != nil {
			t.Fatal(err)
		}

		if _, err := sentinel.Seek(0, io.SeekStart); err != nil {
			t.Fatal(err)
		}
		got, err := io.ReadAll(sentinel)
		if err != nil {
			t.Fatal(err)
		}
		if string(got) != "secret" {
			t.Fatalf("sentinel = %q, want %q", got, "secret")
		}
		assertRunningMarker(t, path)
	})
}

func assertRunningMarker(t *testing.T, path string) {
	t.Helper()
	info, err := os.Lstat(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode()&os.ModeSymlink != 0 {
		t.Fatalf("%s is a symlink", path)
	}
	if !info.Mode().IsRegular() {
		t.Fatalf("%s mode = %s, want a regular file", path, info.Mode())
	}
	// OpenFile applies the process umask to the requested mode.
	if got, want := info.Mode().Perm(), os.FileMode(0644)&^processUmask(); got != want {
		t.Fatalf("%s perm = %o, want %o", path, got, want)
	}
	got, err := os.ReadFile(path) //nolint:gosec // G304: path is a temp file created in this test
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "running" {
		t.Fatalf("marker = %q, want %q", got, "running")
	}
}

// processUmask reads the process file-mode mask. Umask(0) replaces it, so the
// previous value is restored before returning.
func processUmask() os.FileMode {
	mask := syscall.Umask(0)
	syscall.Umask(mask)
	return os.FileMode(mask)
}
