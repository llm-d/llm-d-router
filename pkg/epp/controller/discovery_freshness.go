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
	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	poolutil "github.com/llm-d/llm-d-router/pkg/epp/util/pool"
)

// DiscoveryFreshness verifies watch-driven routing membership against uncached API reads.
// It never writes membership; the Kubernetes reconcilers own those updates.
type DiscoveryFreshness struct {
	reader    client.Reader
	datastore datastore.Datastore
	pool      types.NamespacedName
	threshold time.Duration
	clock     clock.Clock
	validated atomic.Pointer[validatedMembership]
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
	for ctx.Err() == nil {
		// At most 0.4 thresholds between starts plus a one-third-threshold read
		// leaves time to renew the previous snapshot before it expires.
		timer := d.clock.NewTimer(wait.Jitter(d.threshold/3, 0.2))
		attemptCtx, cancel := context.WithTimeout(ctx, d.threshold/3)
		if err := d.refresh(attemptCtx); err != nil && ctx.Err() == nil {
			log.FromContext(ctx).Error(err, "Unable to validate Kubernetes discovery membership")
		}
		cancel()
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
	if err != nil {
		// A known mismatch cannot use the grace period for unavailable discovery.
		d.validated.Store(nil)
		return err
	}
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

func (d *DiscoveryFreshness) Admit(_ context.Context, _ *fwksched.InferenceRequest, candidates []fwksched.Endpoint) error {
	validated := d.validated.Load()
	if err := d.checkReady(validated); err != nil {
		return errcommon.Error{Code: errcommon.ServiceUnavailable, Msg: err.Error()}
	}
	for _, candidate := range candidates {
		meta := candidate.GetMetadata()
		if !meta.Equal(validated.endpoints[meta.ID]) {
			return errcommon.Error{Code: errcommon.ServiceUnavailable, Msg: fmt.Sprintf("request endpoint %s does not match validated Kubernetes membership", meta.ID)}
		}
	}
	return nil
}
