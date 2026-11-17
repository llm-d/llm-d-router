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

package controller

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/utils/clock"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
	poolutil "github.com/llm-d/llm-d-router/pkg/epp/util/pool"
)

// DiscoveryFreshness verifies watch-driven routing membership against uncached API reads.
// It never writes membership; the Kubernetes reconcilers own those updates.
type DiscoveryFreshness struct {
	reader                 client.Reader
	datastore              datastore.Datastore
	pool                   types.NamespacedName
	threshold              time.Duration
	clock                  clock.Clock
	validated              atomic.Pointer[validatedMembership]
	previousDisagreement   datastore.MembershipDisagreement
	persistentDisagreement bool
}

type validatedMembership struct {
	at        time.Time
	endpoints map[types.NamespacedName]*fwkdl.EndpointMetadata
}

var _ fwkplugin.ReadinessChecker = (*DiscoveryFreshness)(nil)
var _ fwkrh.Admitter = (*DiscoveryFreshness)(nil)

// NewDiscoveryFreshness requires an uncached reader and a positive threshold.
func NewDiscoveryFreshness(reader client.Reader, ds datastore.Datastore, pool types.NamespacedName, threshold time.Duration) *DiscoveryFreshness {
	return &DiscoveryFreshness{reader: reader, datastore: ds, pool: pool, threshold: threshold, clock: clock.RealClock{}}
}

func (*DiscoveryFreshness) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "kubernetes-discovery-freshness", Name: "kubernetes-discovery-freshness"}
}

func (*DiscoveryFreshness) NeedLeaderElection() bool { return false }

func (d *DiscoveryFreshness) Start(ctx context.Context) error {
	defer d.validated.Store(nil)
	ready := false
	for ctx.Err() == nil {
		// At most 0.4 thresholds between starts plus a one-third-threshold read
		// leaves time to renew the previous snapshot before it expires.
		timer := d.clock.NewTimer(wait.Jitter(d.threshold/3, 0.2))
		attemptCtx, cancel := context.WithTimeout(ctx, d.threshold/3)
		if err := d.refresh(attemptCtx); err != nil && ctx.Err() == nil {
			log.FromContext(ctx).V(logutil.DEBUG).Info("Unable to sample Kubernetes discovery membership", "error", err)
		}
		cancel()
		nextReady := d.CheckReady() == nil
		if nextReady != ready {
			log.FromContext(ctx).Info("Kubernetes discovery freshness state changed", "ready", nextReady, "pool", d.pool.String())
			ready = nextReady
		}
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil
		case <-timer.C():
		}
	}
	return nil
}

func (d *DiscoveryFreshness) refresh(ctx context.Context) error {
	// Charge API and comparison time to the freshness budget.
	started := d.clock.Now()
	pool := &v1.InferencePool{}
	if err := d.reader.Get(ctx, d.pool, pool); err != nil {
		if apierrors.IsNotFound(err) {
			d.validated.Store(nil)
		}
		return fmt.Errorf("read InferencePool membership: %w", err)
	}
	if !pool.DeletionTimestamp.IsZero() {
		d.validated.Store(nil)
		return fmt.Errorf("InferencePool %s is being deleted", d.pool)
	}
	endpointPool := poolutil.InferencePoolToEndpointPool(pool)
	if err := datastore.VerifyPool(d.datastore, endpointPool); err != nil {
		d.validated.Store(nil)
		return err
	}
	pods := &corev1.PodList{}
	if err := d.reader.List(ctx, pods, client.InNamespace(d.pool.Namespace), client.MatchingLabelsSelector{Selector: endpointPool.Selector}); err != nil {
		return fmt.Errorf("read Pod membership: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	endpoints, err := datastore.VerifyMembership(d.datastore, endpointPool, pods.Items)
	if endpoints == nil {
		d.validated.Store(nil)
		return err
	}
	var mismatches datastore.MembershipDisagreement
	if err != nil {
		if !errors.As(err, &mismatches) {
			return err
		}
	}
	d.observeDisagreement(ctx, mismatches)
	d.validated.Store(&validatedMembership{at: started, endpoints: endpoints})
	return nil
}

func (d *DiscoveryFreshness) CheckReady() error {
	return d.checkReady(d.validated.Load())
}

func (d *DiscoveryFreshness) checkReady(validated *validatedMembership) error {
	if validated == nil || d.clock.Since(validated.at) >= d.threshold {
		return fmt.Errorf("discovery membership has not been validated within %s", d.threshold)
	}
	return nil
}

// CheckEndpoints validates freshness and any selected endpoints against one snapshot.
func (d *DiscoveryFreshness) CheckEndpoints(endpoints ...*fwkdl.EndpointMetadata) error {
	validated := d.validated.Load()
	if err := d.checkReady(validated); err != nil {
		return err
	}
	for _, meta := range endpoints {
		if meta == nil || !meta.Equal(validated.endpoints[meta.ID]) {
			return errors.New("selected endpoint does not match validated Kubernetes membership")
		}
	}
	return nil
}

func (d *DiscoveryFreshness) Admit(_ context.Context, _ *fwksched.InferenceRequest, candidates []fwksched.Endpoint) error {
	endpoints := make([]*fwkdl.EndpointMetadata, len(candidates))
	for i, candidate := range candidates {
		endpoints[i] = candidate.GetMetadata()
	}
	if err := d.CheckEndpoints(endpoints...); err != nil {
		return errcommon.Error{Code: errcommon.ServiceUnavailable, Msg: err.Error()}
	}
	return nil
}

// FilterCandidates keeps only candidates represented in a fresh authoritative sample.
func (d *DiscoveryFreshness) FilterCandidates(ctx context.Context, candidates []fwksched.Endpoint) ([]fwksched.Endpoint, error) {
	validated := d.validated.Load()
	if err := d.checkReady(validated); err != nil {
		return nil, errcommon.Error{Code: errcommon.ServiceUnavailable, Msg: err.Error()}
	}
	eligible := make([]fwksched.Endpoint, 0, len(candidates))
	for _, candidate := range candidates {
		meta := candidate.GetMetadata()
		if meta != nil && meta.Equal(validated.endpoints[meta.ID]) {
			eligible = append(eligible, candidate)
		}
	}
	if excluded := len(candidates) - len(eligible); excluded > 0 {
		log.FromContext(ctx).V(logutil.DEBUG).Info("Excluded candidates outside sampled Kubernetes membership", "excluded", excluded, "eligible", len(eligible))
	}
	if len(eligible) == 0 {
		return nil, errcommon.Error{Code: errcommon.ServiceUnavailable, Msg: "no candidates match sampled Kubernetes membership"}
	}
	return eligible, nil
}

func (d *DiscoveryFreshness) observeDisagreement(ctx context.Context, mismatches datastore.MembershipDisagreement) {
	logger := log.FromContext(ctx)
	persistent := false
	for id, reason := range mismatches {
		if _, ok := d.previousDisagreement[id]; ok {
			persistent = true
		}
		logger.V(logutil.DEBUG).Info("Kubernetes membership differs from routing datastore", "endpoint", id.String(), "reason", reason)
	}
	if persistent != d.persistentDisagreement {
		if persistent {
			logger.Info("Kubernetes membership disagreement persists across successful samples", "pool", d.pool.String())
		} else {
			logger.Info("Persistent Kubernetes membership disagreement cleared", "pool", d.pool.String())
		}
	}
	d.previousDisagreement = mismatches
	d.persistentDisagreement = persistent
	metrics.RecordDiscoveryDisagreement(persistent)
}
