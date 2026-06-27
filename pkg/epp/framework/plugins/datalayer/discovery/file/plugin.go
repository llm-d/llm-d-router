/*
Copyright 2025 The Kubernetes Authors.

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

// Package file provides a file-based EndpointDiscovery implementation that reads
// a YAML (or JSON) file listing inference endpoints.
package file

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"sort"
	"strconv"
	"sync"

	"github.com/fsnotify/fsnotify"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/yaml"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

const PluginType = "file-discovery"

// EndpointEntry is the YAML/JSON representation of a single endpoint.
type EndpointEntry struct {
	Name      string            `json:"name"                yaml:"name"`
	Namespace string            `json:"namespace,omitempty" yaml:"namespace,omitempty"`
	Address   string            `json:"address"             yaml:"address"`
	Port      string            `json:"port"                yaml:"port"`
	Labels    map[string]string `json:"labels,omitempty"    yaml:"labels,omitempty"`
}

// EndpointsFile is the top-level structure of the endpoints YAML/JSON file.
type EndpointsFile struct {
	Endpoints []EndpointEntry `json:"endpoints" yaml:"endpoints"`
}

// params is the user-facing configuration for the file-discovery plugin.
// It is unmarshalled from the plugin's "parameters" block in the EPP config.
type params struct {
	// Path is the absolute path to the YAML/JSON file listing endpoints.
	// Required.
	Path string `json:"path"`
	// WatchFile enables hot-reload via fsnotify: edits, atomic renames, and
	// ConfigMap-style symlink swaps trigger a reload of the file. When
	// false (default), the file is read once at startup and never re-read.
	WatchFile bool `json:"watchFile"`
}

// FileDiscovery implements EndpointDiscovery by reading a static endpoints file.
type FileDiscovery struct {
	typedName fwkplugin.TypedName
	path      string
	watchFile bool
	// mu guards endpoints, which DumpState reads concurrently with load.
	mu sync.RWMutex
	// endpoints is the set of endpoint identities applied to the datastore
	// from the last successful load. Used as a key set only -- values are
	// zero-byte structs. Compared against the entries parsed during a
	// reload to compute which endpoints to delete from the datastore.
	endpoints map[types.NamespacedName]struct{}

	ready     chan struct{}
	readyOnce sync.Once
}

var (
	_ fwkdl.EndpointDiscovery = (*FileDiscovery)(nil)
	_ fwkplugin.StateDumper   = (*FileDiscovery)(nil)
)

// Factory is the plugin factory for file-discovery.
func Factory(name string, parameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	p := &params{WatchFile: false}
	if parameters != nil {
		if err := parameters.Decode(p); err != nil {
			return nil, fmt.Errorf("file-discovery: failed to parse parameters: %w", err)
		}
	}
	if p.Path == "" {
		return nil, errors.New("file-discovery: 'path' parameter is required")
	}
	if name == "" {
		name = PluginType
	}
	return &FileDiscovery{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: name},
		path:      p.Path,
		watchFile: p.WatchFile,
		endpoints: make(map[types.NamespacedName]struct{}),
		ready:     make(chan struct{}),
	}, nil
}

func (f *FileDiscovery) TypedName() fwkplugin.TypedName { return f.typedName }

const maxDebugDumpEndpoints = 100

// discoveryState is the sanitized snapshot returned by DumpState: discovered
// endpoint identities only, never their addresses or labels.
type discoveryState struct {
	Endpoints      []string `json:"endpoints"`
	TotalEndpoints int      `json:"totalEndpoints"`
	MaxEndpoints   int      `json:"maxEndpoints"`
	Truncated      bool     `json:"truncated"`
}

// DumpState reports the endpoint identities currently loaded from the file,
// sorted and capped to maxDebugDumpEndpoints so the payload stays bounded. The
// set is snapshotted under a read lock, so a concurrent reload may not yet be
// reflected; best-effort visibility is enough for a debug endpoint.
func (f *FileDiscovery) DumpState() (json.RawMessage, error) {
	f.mu.RLock()
	names := make([]string, 0, len(f.endpoints))
	for id := range f.endpoints {
		names = append(names, id.String())
	}
	f.mu.RUnlock()

	total := len(names)
	sort.Strings(names)

	state := discoveryState{TotalEndpoints: total, MaxEndpoints: maxDebugDumpEndpoints}
	if len(names) > maxDebugDumpEndpoints {
		names = names[:maxDebugDumpEndpoints]
		state.Truncated = true
	}
	state.Endpoints = names
	return json.Marshal(state)
}

// Ready returns a channel closed after the first successful load of the
// endpoints file. See EndpointDiscovery.Ready for the contract.
func (f *FileDiscovery) Ready() <-chan struct{} { return f.ready }

// Start loads the endpoints file, notifies the datastore, then optionally watches
// for changes. Blocks until ctx is cancelled or a fatal error occurs.
func (f *FileDiscovery) Start(ctx context.Context, notifier fwkdl.DiscoveryNotifier) error {
	logger := log.FromContext(ctx).WithValues("plugin", PluginType, "path", f.path)

	if err := f.load(notifier); err != nil {
		return fmt.Errorf("file-discovery: initial load failed: %w", err)
	}
	f.readyOnce.Do(func() { close(f.ready) })

	if !f.watchFile {
		<-ctx.Done()
		return nil
	}

	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return fmt.Errorf("file-discovery: failed to create watcher: %w", err)
	}
	defer watcher.Close()

	if err := watcher.Add(f.path); err != nil {
		return fmt.Errorf("file-discovery: failed to watch %s: %w", f.path, err)
	}

	logger.Info("watching endpoints file for changes")
	for {
		select {
		case <-ctx.Done():
			return nil
		case event, ok := <-watcher.Events:
			if !ok {
				return nil
			}
			if event.Has(fsnotify.Remove) || event.Has(fsnotify.Create) {
				// Re-attach to the new inode at f.path after atomic rename
				// (editor safe-write) or ConfigMap symlink swap. Safe to
				// ignore error: if the file isn't present yet the subsequent
				// Create event will re-add it.
				_ = watcher.Add(f.path)
			}
			if event.Has(fsnotify.Write) || event.Has(fsnotify.Create) || event.Has(fsnotify.Remove) {
				logger.Info("endpoints file changed, reloading")
				if err := f.load(notifier); err != nil {
					logger.Error(err, "failed to reload endpoints file")
				}
			}
		case err, ok := <-watcher.Errors:
			if !ok {
				return nil
			}
			logger.Error(err, "watcher error")
		}
	}
}

const maxEndpointsFileSize = 1 << 20 // 1 MiB

func (f *FileDiscovery) load(notifier fwkdl.DiscoveryNotifier) error {
	fh, err := os.Open(f.path)
	if err != nil {
		return fmt.Errorf("reading %s: %w", f.path, err)
	}
	defer fh.Close()

	data, err := io.ReadAll(io.LimitReader(fh, maxEndpointsFileSize+1))
	if err != nil {
		return fmt.Errorf("reading %s: %w", f.path, err)
	}
	if len(data) > maxEndpointsFileSize {
		return fmt.Errorf("endpoints file %s exceeds 1 MiB limit", f.path)
	}

	jsonData, err := yaml.YAMLToJSON(data)
	if err != nil {
		return fmt.Errorf("parsing %s: %w", f.path, err)
	}

	var ef EndpointsFile
	if err := json.Unmarshal(jsonData, &ef); err != nil {
		return fmt.Errorf("unmarshalling %s: %w", f.path, err)
	}

	incoming := make(map[types.NamespacedName]struct{}, len(ef.Endpoints))
	var errs []error
	for _, e := range ef.Endpoints {
		if ip := net.ParseIP(e.Address); ip == nil || ip.To4() == nil {
			errs = append(errs, fmt.Errorf("endpoint %q: invalid IPv4 address %q", e.Name, e.Address))
			continue
		}
		if portNum, err := strconv.Atoi(e.Port); err != nil || portNum < 1 || portNum > 65535 {
			errs = append(errs, fmt.Errorf("endpoint %q: invalid port %q", e.Name, e.Port))
			continue
		}
		ns := e.Namespace
		if ns == "" {
			ns = "default"
		}
		meta := &fwkdl.EndpointMetadata{
			NamespacedName: types.NamespacedName{Name: e.Name, Namespace: ns},
			PodName:        e.Name,
			Address:        e.Address,
			Port:           e.Port,
			MetricsHost:    net.JoinHostPort(e.Address, e.Port),
			Labels:         e.Labels,
		}
		incoming[meta.NamespacedName] = struct{}{}
		notifier.Upsert(meta)
	}

	f.mu.Lock()
	old := f.endpoints
	f.endpoints = incoming
	f.mu.Unlock()

	for id := range old {
		if _, ok := incoming[id]; !ok {
			notifier.Delete(id)
		}
	}
	return errors.Join(errs...)
}
