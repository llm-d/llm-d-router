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

package datalayer

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/go-logr/logr"
	"k8s.io/apimachinery/pkg/runtime/schema"
	ctrl "sigs.k8s.io/controller-runtime"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

var (
	ErrSourceTypeCollision    = errors.New("source type registered across variants")
	ErrDuplicateExtractorType = errors.New("duplicate extractor type configured for the same source")
)

type sourceVariant string

const (
	variantPolling      sourceVariant = "polling"
	variantNotification sourceVariant = "notification"
	variantEndpoint     sourceVariant = "endpoint"
)

// Runtime manages data sources, extractors, their mapping, and endpoint lifecycle.
type Runtime struct {
	pollingInterval time.Duration // used for polling sources

	dispatchers  *pollingDispatchers
	notification *notificationManager
	endpoint     *endpointManager
	extractors   *extractorMap

	pendingMu            sync.Mutex
	pendingRegistrations []fwkdl.PendingRegistration // code-registered (source-type, extractor) pairs, resolved by Configure()

	collectors *collectorManager // per-endpoint poller, keyed by namespaced name
	logger     logr.Logger       // Set in Configure; used where no context is available (e.g. ReleaseEndpoint).
}

const (
	defaultRefreshInterval = 50 * time.Millisecond
)

// NewRuntime creates a new Runtime with the given polling interval.
// If duration is <= 0, uses the defaultRefreshInterval.
func NewRuntime(pollingInterval time.Duration) *Runtime {
	interval := defaultRefreshInterval
	if pollingInterval > 0 {
		interval = pollingInterval
	}
	return &Runtime{
		pollingInterval: interval,
		dispatchers:     newPollingDispatchers(),
		notification:    newNotificationManager(),
		endpoint:        newEndpointManager(),
		extractors:      newExtractorMap(),
		collectors:      newCollectorManager(),
		logger:          logr.Discard(),
	}
}

// Configure is called to transform the configuration information into the Runtime's
// internal fields.
func (r *Runtime) Configure(cfg *Config, enableNewMetrics bool, disallowedExtractorType string, logger logr.Logger) error {
	hasPending := len(r.pendingRegistrations) > 0
	if (cfg == nil || len(cfg.Sources) == 0) && !hasPending {
		if enableNewMetrics {
			return errors.New("data layer enabled but no data sources configured")
		}
		return nil
	}

	r.logger = logger
	numSources := 0
	if cfg != nil {
		numSources = len(cfg.Sources)
	}
	logger.Info("Configuring datalayer runtime", "numSources", numSources)

	// boundTypes tracks (srcName, extractor type) pairs that already have a bound
	// extractor. Pending registrations consult this so code-registered defaults
	// yield to user-config extractors of the same Type.
	boundTypes := make(map[string]map[string]struct{})
	markBound := func(srcName, extType string) {
		if boundTypes[srcName] == nil {
			boundTypes[srcName] = make(map[string]struct{})
		}
		boundTypes[srcName][extType] = struct{}{}
	}

	gvk := newGvk()
	if cfg != nil {
		for _, srcCfg := range cfg.Sources {
			src := srcCfg.Plugin
			srcName := src.TypedName().Name

			logger.V(logging.DEFAULT).Info("Processing source", "source", srcName, "numExtractors", len(srcCfg.Extractors))
			if err := r.validateSourceExtractors(src, srcCfg.Extractors, disallowedExtractorType); err != nil {
				return err
			}

			if err := r.registerSource(src, gvk); err != nil {
				return err
			}

			// PollingDispatchers own their extractors; notification/endpoint use extractorMap.
			if disp, ok := src.(fwkdl.PollingDispatcher); ok {
				for _, ext := range srcCfg.Extractors {
					if err := disp.AppendExtractor(ext); err != nil {
						return fmt.Errorf("dispatcher %s rejected extractor %s: %w",
							src.TypedName(), ext.TypedName(), err)
					}
					markBound(srcName, ext.TypedName().Type)
				}
			} else {
				for _, ext := range srcCfg.Extractors {
					r.extractors.Append(srcName, ext)
					markBound(srcName, ext.TypedName().Type)
				}
			}

			extractorNames := make([]string, len(srcCfg.Extractors))
			for i, ext := range srcCfg.Extractors {
				extractorNames[i] = ext.TypedName().String()
			}
			logger.V(logging.DEFAULT).Info("Source configured", "source", srcName, "extractors", extractorNames)
		}
	}

	if err := r.validateNoCrossVariantCollisions(); err != nil {
		return err
	}

	// Resolve code-registered pending registrations after processing user config.
	for _, pending := range r.pendingRegistrations {
		var gvkFilter *schema.GroupVersionKind
		if ns, ok := pending.DefaultSource.(fwkdl.NotificationSource); ok {
			sourceGVK := ns.GVK()
			gvkFilter = &sourceGVK
		}
		srcName, matchedSrc, err := r.findSourceByType(pending.SourceType, gvkFilter)
		if err != nil {
			return fmt.Errorf("resolve %s: %w", pending.Extractor.TypedName(), err)
		}

		if matchedSrc == nil {
			if pending.DefaultSource == nil {
				msg := fmt.Sprintf("extractor %s requires source type %s, not configured",
					pending.Extractor.TypedName(), pending.SourceType)
				if pending.IfMissing == fwkdl.Warn {
					logger.Info("datalayer: skipping unresolved dependency", "reason", msg)
					continue
				}
				return errors.New(msg)
			}
			if err := r.registerSource(pending.DefaultSource, gvk); err != nil {
				return fmt.Errorf("auto-register default source for %s: %w", pending.Extractor.TypedName(), err)
			}
			srcName = pending.DefaultSource.TypedName().Name
			matchedSrc = pending.DefaultSource
		}

		if err := r.validateSourceExtractors(matchedSrc, []fwkplugin.Plugin{pending.Extractor}, disallowedExtractorType); err != nil {
			return fmt.Errorf("code-registered extractor %s incompatible with source %s: %w",
				pending.Extractor.TypedName(), srcName, err)
		}

		// Yield to user-config extractor of the same Type on this source.
		extType := pending.Extractor.TypedName().Type
		if _, alreadyBound := boundTypes[srcName][extType]; alreadyBound {
			continue
		}

		if disp, ok := matchedSrc.(fwkdl.PollingDispatcher); ok {
			if err := disp.AppendExtractor(pending.Extractor); err != nil {
				return fmt.Errorf("dispatcher %s rejected pending extractor %s: %w",
					matchedSrc.TypedName(), pending.Extractor.TypedName(), err)
			}
		} else {
			r.extractors.Append(srcName, pending.Extractor)
		}
		markBound(srcName, extType)
	}

	logger.Info("Datalayer runtime configured",
		"pollers", r.dispatchers.Count(),
		"notifiers", r.notification.Count(),
		"endpointSources", r.endpoint.Count())
	return nil
}

// Register stores a pending source/extractor dependency declared by a plugin.
// Called by plugins implementing Registrant.RegisterDependencies() before Configure() runs.
func (r *Runtime) Register(reg fwkdl.PendingRegistration) error {
	if reg.Extractor == nil {
		return fmt.Errorf("plugin %s: PendingRegistration.Extractor must not be nil", reg.Owner)
	}
	r.pendingMu.Lock()
	r.pendingRegistrations = append(r.pendingRegistrations, reg)
	r.pendingMu.Unlock()
	return nil
}

// registerSource dispatches src to the matching variant manager. g enforces
// per-Configure-call GVK uniqueness for NotificationSources. src may be a
// PollingDispatcher (not a DataSource), so the parameter is plugin.Plugin.
//
// A source that implements more than one variant interface is rejected: type-
// switch order would otherwise silently bind it to the first match, hiding the
// ambiguity until a notification or endpoint event mismatched its handler.
func (r *Runtime) registerSource(src fwkplugin.Plugin, g *gvk) error {
	var variants []string
	if _, ok := src.(fwkdl.PollingDispatcher); ok {
		variants = append(variants, string(variantPolling))
	}
	if _, ok := src.(fwkdl.NotificationSource); ok {
		variants = append(variants, string(variantNotification))
	}
	if _, ok := src.(fwkdl.EndpointSource); ok {
		variants = append(variants, string(variantEndpoint))
	}
	if len(variants) > 1 {
		return fmt.Errorf("source %s implements multiple variant interfaces (%v); a source must implement exactly one of PollingDispatcher, NotificationSource, EndpointSource",
			src.TypedName().String(), variants)
	}

	switch s := src.(type) {
	case fwkdl.PollingDispatcher:
		return r.dispatchers.Register(s)
	case fwkdl.NotificationSource:
		if err := g.Check(s); err != nil {
			return err
		}
		r.notification.Set(s)
		return nil
	case fwkdl.EndpointSource:
		r.endpoint.Set(s)
		return nil
	default:
		return fmt.Errorf("skipping unknown datasource plugin type %s", src.TypedName().String())
	}
}

// validateNoCrossVariantCollisions errors if any SourceType is registered in
// more than one variant manager. findSourceByType already catches this when
// a pending extractor references the colliding type; this check fires
// exhaustively at end of Configure for the no-pending case.
func (r *Runtime) validateNoCrossVariantCollisions() error {
	type seenSource struct {
		variant sourceVariant
		name    string
	}
	seen := make(map[string]seenSource)

	check := func(name string, src fwkplugin.Plugin, v sourceVariant) error {
		t := src.TypedName().Type
		if prior, ok := seen[t]; ok && prior.variant != v {
			return fmt.Errorf("%w: %q in %s (%s) and %s (%s)",
				ErrSourceTypeCollision, t, prior.variant, prior.name, v, name)
		}
		seen[t] = seenSource{variant: v, name: name}
		return nil
	}

	for name, disp := range r.dispatchers.Dispatchers() {
		if err := check(name, disp, variantPolling); err != nil {
			return err
		}
	}

	var firstErr error
	r.notification.Range(func(name string, src fwkdl.NotificationSource) bool {
		if err := check(name, src, variantNotification); err != nil {
			firstErr = err
			return false
		}
		return true
	})
	if firstErr != nil {
		return firstErr
	}
	r.endpoint.Range(func(name string, src fwkdl.EndpointSource) bool {
		if err := check(name, src, variantEndpoint); err != nil {
			firstErr = err
			return false
		}
		return true
	})
	return firstErr
}

// findSourceByType walks every variant manager and returns the matching source.
// Returns ErrSourceTypeCollision if sourceType is registered in more than one variant.
// Return type is plugin.Plugin because PollingDispatcher is not a DataSource.
func (r *Runtime) findSourceByType(sourceType string, gvkFilter *schema.GroupVersionKind) (string, fwkplugin.Plugin, error) {
	matches := func(src fwkplugin.Plugin) bool {
		if src.TypedName().Type != sourceType {
			return false
		}
		if gvkFilter != nil {
			if ns, ok := src.(fwkdl.NotificationSource); ok {
				if ns.GVK().String() != gvkFilter.String() {
					return false
				}
			}
		}
		return true
	}

	// Polling dispatchers searched first; one-pass scan.
	var pollingHit sourceHit
	for name, disp := range r.dispatchers.Dispatchers() {
		if matches(disp) {
			pollingHit = sourceHit{variant: variantPolling, name: name, src: disp}
			break
		}
	}

	matched, err := findUnique(sourceType,
		pollingHit,
		r.notification.findFirst(matches),
		r.endpoint.findFirst(matches),
	)
	if err != nil {
		return "", nil, err
	}
	return matched.name, matched.src, nil
}

// Start is called to enable the Runtime to start processing data collection. It wires
// Kubernetes notifications into the manager.
func (r *Runtime) Start(ctx context.Context, mgr ctrl.Manager) error {
	return r.notification.ForEach(func(srcName string, src fwkdl.NotificationSource) error {
		var extractors []fwkdl.NotificationExtractor
		if rawExts, ok := r.extractors.Get(srcName); ok {
			extractors = make([]fwkdl.NotificationExtractor, len(rawExts))
			for i, e := range rawExts {
				extractors[i] = e.(fwkdl.NotificationExtractor)
			}
		}
		if err := BindNotificationSource(src, extractors, mgr); err != nil {
			return fmt.Errorf("failed to bind notification source %s: %w", src.TypedName(), err)
		}
		return nil
	})
}

// NewEndpoint sets up data polling on the provided endpoint.
func (r *Runtime) NewEndpoint(ctx context.Context, endpointMetadata *fwkdl.EndpointMetadata, _ PoolInfo) fwkdl.Endpoint {
	logger, _ := logr.FromContext(ctx)
	logger = logger.WithValues("endpoint", endpointMetadata.GetNamespacedName())

	dispatchers := make([]fwkdl.PollingDispatcher, 0, r.dispatchers.Count())
	for _, d := range r.dispatchers.Dispatchers() {
		dispatchers = append(dispatchers, d)
	}
	if len(dispatchers) == 0 {
		logger.Info("No polling sources configured, creating endpoint without collector")
		endpoint := fwkdl.NewEndpoint(endpointMetadata, nil)
		r.dispatchEndpointEvent(ctx, logger, fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: endpoint})
		return endpoint
	}

	endpoint := fwkdl.NewEndpoint(endpointMetadata, nil)
	collector := NewCollector()

	key := endpointMetadata.GetNamespacedName()
	if !r.collectors.Register(key, collector) {
		logger.V(logging.DEFAULT).Info("collector already running for endpoint", "endpoint", key)
		return nil
	}

	ticker := NewTimeTicker(r.pollingInterval)
	if err := collector.Start(ctx, ticker, endpoint, dispatchers); err != nil {
		logger.Error(err, "failed to start collector for endpoint", "endpoint", key)
		r.collectors.Remove(key)
		return nil
	}

	r.dispatchEndpointEvent(ctx, logger, fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: endpoint})
	return endpoint
}

// ReleaseEndpoint terminates polling for data on the given endpoint.
func (r *Runtime) ReleaseEndpoint(ep fwkdl.Endpoint) {
	r.dispatchEndpointEvent(context.Background(), r.logger, fwkdl.EndpointEvent{Type: fwkdl.EventDelete, Endpoint: ep})

	key := ep.GetMetadata().GetNamespacedName()
	if collector, ok := r.collectors.Remove(key); ok {
		collector.Stop()
	}
}

// dispatchEndpointEvent routes an endpoint lifecycle event to all registered
// EndpointSources and their extractors.
func (r *Runtime) dispatchEndpointEvent(ctx context.Context, logger logr.Logger, event fwkdl.EndpointEvent) {
	if r.endpoint.IsEmpty() {
		return
	}
	r.endpoint.Range(func(srcName string, src fwkdl.EndpointSource) bool {
		processed, err := src.NotifyEndpoint(ctx, event)
		if err != nil {
			logger.Error(err, "endpoint source failed to process event", "source", srcName)
			return true
		}
		if processed == nil {
			return true
		}

		exts, ok := r.extractors.Get(srcName)
		if !ok {
			return true
		}
		for _, ext := range exts {
			if epExt, ok := ext.(fwkdl.EndpointExtractor); ok {
				if err := epExt.Extract(ctx, *processed); err != nil {
					logger.Error(err, "endpoint extractor failed", "extractor", ext.TypedName())
				}
			}
		}
		return true
	})
}

// validateSourceExtractors enforces config-time invariants on the extractors
// declared for a source: duplicate Type detection, disallowed-type rejection,
// and variant-specific contracts (NotificationExtractor + GVK match for
// NotificationSources; EndpointExtractor for EndpointSources). Compile-time
// Extractor[T] dispatch replaces the prior reflect-based input/output checks.
//
// src is plugin.Plugin to accommodate PollingDispatcher (not a DataSource);
// dispatcher.AppendExtractor enforces its own typed contract.
func (r *Runtime) validateSourceExtractors(src fwkplugin.Plugin, extractors []fwkplugin.Plugin, disallowedExtractorType string) error {
	seenTypes := make(map[string]struct{}, len(extractors))
	for _, ext := range extractors {
		extType := ext.TypedName().Type
		if _, dup := seenTypes[extType]; dup {
			return fmt.Errorf("%w: source=%s, extractor type=%s",
				ErrDuplicateExtractorType, src.TypedName().String(), extType)
		}
		seenTypes[extType] = struct{}{}
	}

	for _, ext := range extractors {
		if disallowedExtractorType != "" && ext.TypedName().Type == disallowedExtractorType {
			return fmt.Errorf("disallowed Extractor %s is configured for source %s",
				ext.TypedName().String(), src.TypedName().String())
		}

		if notifySrc, ok := src.(fwkdl.NotificationSource); ok {
			notifyExt, ok := ext.(fwkdl.NotificationExtractor)
			if !ok {
				return fmt.Errorf("notification source %s requires a NotificationExtractor; extractor %s does not implement it",
					src.TypedName(), ext.TypedName())
			}
			if notifySrc.GVK().String() != notifyExt.GVK().String() {
				return fmt.Errorf("extractor %s GVK %s does not match source %s GVK %s",
					ext.TypedName(), notifyExt.GVK(), src.TypedName(), notifySrc.GVK())
			}
		}
		if _, ok := src.(fwkdl.EndpointSource); ok {
			if _, ok := ext.(fwkdl.EndpointExtractor); !ok {
				return fmt.Errorf("endpoint source %s requires an EndpointExtractor; extractor %s does not implement it",
					src.TypedName(), ext.TypedName())
			}
		}
	}
	return nil
}

// gvk enforces per-Configure-call GVK uniqueness for NotificationSources.
type gvk struct {
	seen map[string]string // gvk -> registered source name
}

func newGvk() *gvk {
	return &gvk{seen: make(map[string]string)}
}

// Check rejects src if its GVK has already been seen by this tracker.
func (g *gvk) Check(src fwkdl.NotificationSource) error {
	key := src.GVK().String()
	if existing, ok := g.seen[key]; ok {
		return fmt.Errorf("duplicate notification source GVK %s: already used by source %s, cannot add %s",
			key, existing, src.TypedName().String())
	}
	g.seen[key] = src.TypedName().Name
	return nil
}

// findUnique returns the single matching source across hits.
// Returns ErrSourceTypeCollision if more than one hit is present.
func findUnique(sourceType string, hits ...sourceHit) (sourceHit, error) {
	var matched sourceHit
	for _, h := range hits {
		if h.src == nil {
			continue
		}
		if matched.src != nil {
			return sourceHit{}, fmt.Errorf("%w: %q in %s (%s) and %s (%s)",
				ErrSourceTypeCollision, sourceType,
				matched.variant, matched.name,
				h.variant, h.name)
		}
		matched = h
	}
	return matched, nil
}

var _ EndpointFactory = (*Runtime)(nil)
var _ fwkdl.Registrar = (*Runtime)(nil)
