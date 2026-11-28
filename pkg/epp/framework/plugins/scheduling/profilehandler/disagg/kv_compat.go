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

package disagg

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
)

const (
	// KVCompatPDDeciderPluginType is the type of the decider that declines remote prefill when no
	// prefill endpoint can exchange KV with the decode endpoint, and otherwise asks another decider.
	KVCompatPDDeciderPluginType = "kv-compat-pd-decider"
	// KVCompatFilterPluginType is the type of the filter that keeps prefill candidates which can
	// exchange KV with the decode endpoint picked earlier in the cycle.
	KVCompatFilterPluginType = "kv-compat-filter"

	defaultKVCompatKeyLabel          = "llm-d.ai/kv-compat-key"
	defaultKVCompatAcceptLabelPrefix = "accept.kv-compat.llm-d.ai/"
	kvCompatPairSeparator            = "->"
)

// prefillCapableRoles are the role label values of endpoints that can run prefill.
var prefillCapableRoles = map[string]struct{}{
	bylabel.RolePrefill:             {},
	bylabel.RoleEncodePrefill:       {},
	bylabel.RolePrefillDecode:       {},
	bylabel.RoleEncodePrefillDecode: {},
}

// KVCompatPDDeciderParameters configures the kv-compat-pd-decider.
type KVCompatPDDeciderParameters struct {
	// Decider names the prefill decider asked when a compatible prefill endpoint exists.
	Decider string `json:"decider" pluginRef:""`
	// KeyLabel is the endpoint label holding the KV compatibility key. Two endpoints can exchange
	// KV when their keys are equal. Defaults to llm-d.ai/kv-compat-key.
	KeyLabel string `json:"keyLabel,omitempty"`
	// AcceptLabelPrefix marks approvals published on decode endpoints: a decode endpoint labelled
	// <prefix><prefillKey>=true accepts KV from prefill endpoints with that key. Defaults to
	// accept.kv-compat.llm-d.ai/. Set to "-" to disable label approvals.
	AcceptLabelPrefix string `json:"acceptLabelPrefix,omitempty"`
	// ApprovedPairs lists "prefillKey->decodeKey" pairs that may exchange KV although their keys
	// differ. An approval applies in the listed direction only.
	ApprovedPairs []string `json:"approvedPairs,omitempty"`
	// FailureThreshold is the number of consecutive failed disaggregated requests between a
	// prefill key and a decode endpoint that excludes the pair. 0 disables exclusion. Defaults to 3.
	FailureThreshold *int `json:"failureThreshold,omitempty"`
	// ExclusionDuration is how long an excluded pair stays excluded. Defaults to 5m.
	ExclusionDuration *metav1.Duration `json:"exclusionDuration,omitempty"`
	// FailureCodes are the response statuses counted as a failed request. Defaults to 500, 502
	// and 503.
	FailureCodes []int `json:"failureCodes,omitempty"`
	// PrefillProfile names the scheduling profile that picks the prefill endpoint. Defaults to
	// "prefill".
	PrefillProfile string `json:"prefillProfile,omitempty"`
}

// KVCompatFilterParameters configures the kv-compat-filter.
type KVCompatFilterParameters struct {
	// Decider names the kv-compat-pd-decider whose compatibility rule the filter applies.
	Decider string `json:"decider" pluginRef:""`
}

// kvCompatRule decides whether a prefill endpoint can hand KV to a decode endpoint.
//
// Endpoints carry a compatibility key in a label. When no endpoint in the pool carries a key the
// rule allows every pair, so deployments without keys behave as before. Once any endpoint carries
// a key, an endpoint without one is treated as incompatible with every other endpoint.
type kvCompatRule struct {
	keyLabel     string
	acceptPrefix string
	// approved maps a decode key to the prefill keys it accepts through ApprovedPairs.
	approved map[string][]string
}

func newKVCompatRule(p KVCompatPDDeciderParameters) (kvCompatRule, error) {
	r := kvCompatRule{
		keyLabel:     p.KeyLabel,
		acceptPrefix: p.AcceptLabelPrefix,
		approved:     map[string][]string{},
	}
	if r.keyLabel == "" {
		r.keyLabel = defaultKVCompatKeyLabel
	}
	switch r.acceptPrefix {
	case "":
		r.acceptPrefix = defaultKVCompatAcceptLabelPrefix
	case "-":
		r.acceptPrefix = ""
	}
	for _, pair := range p.ApprovedPairs {
		prefillKey, decodeKey, ok := strings.Cut(pair, kvCompatPairSeparator)
		prefillKey, decodeKey = strings.TrimSpace(prefillKey), strings.TrimSpace(decodeKey)
		if !ok || prefillKey == "" || decodeKey == "" {
			return kvCompatRule{}, fmt.Errorf("approvedPairs entry %q must have the form prefillKey%sdecodeKey", pair, kvCompatPairSeparator)
		}
		r.approved[decodeKey] = append(r.approved[decodeKey], prefillKey)
	}
	return r, nil
}

// accepts reports whether a decode endpoint with the given labels and key accepts KV from a
// prefill endpoint with prefillKey. Both keys must be non-empty.
func (r kvCompatRule) accepts(decodeLabels map[string]string, decodeKey, prefillKey string) bool {
	if prefillKey == decodeKey {
		return true
	}
	if r.acceptPrefix != "" && decodeLabels[r.acceptPrefix+prefillKey] == "true" {
		return true
	}
	for _, k := range r.approved[decodeKey] {
		if k == prefillKey {
			return true
		}
	}
	return false
}

// KVCompatPDDecider declines remote prefill when no prefill endpoint in the pool can exchange KV
// with the decode endpoint, so the request is served decode-only instead of failing at the KV
// handshake. Otherwise it returns the answer of the configured decider.
//
// It keeps an index of the compatibility keys of prefill-capable endpoints, updated from endpoint
// events, so a decision costs a few map lookups regardless of the pool size.
type KVCompatPDDecider struct {
	typedName plugin.TypedName
	decider   deciderPlugin
	rule      kvCompatRule

	mu sync.RWMutex
	// endpoints holds the key and role of every endpoint seen in an endpoint event.
	endpoints map[fwkdl.ID]kvCompatEntry
	// prefillKeys counts prefill-capable endpoints per compatibility key.
	prefillKeys map[string]int
	// keyed counts endpoints that carry a compatibility key.
	keyed int

	exclusion   kvCompatExclusionConfig
	state       kvCompatExclusionState
	spec        fwkdl.CrossReplicaSpec
	pluginState *plugin.PluginState
	handle      plugin.Handle
	now         func() time.Time
}

type kvCompatEntry struct {
	key     string
	prefill bool
	// id is the endpoint ID as a string, kept so a decision can look up shared state without
	// building the string.
	id string
}

var (
	_ deciderPlugin                 = &KVCompatPDDecider{}
	_ prefixMatchInfoConsumer       = &KVCompatPDDecider{}
	_ fwkdl.EndpointExtractor       = &KVCompatPDDecider{}
	_ fwkdl.Registrant              = &KVCompatPDDecider{}
	_ fwkdl.CrossReplicaContributor = &KVCompatPDDecider{}
	_ fwkrh.PreRequest              = &KVCompatPDDecider{}
	_ fwkrh.ResponseHeaderProcessor = &KVCompatPDDecider{}
	_ fwkrh.ResponseBodyProcessor   = &KVCompatPDDecider{}
)

// KVCompatPDDeciderConfigParser parses the parameters, so the loader can resolve the decider
// reference before the factory runs.
func KVCompatPDDeciderConfigParser(rawParameters *json.Decoder, _ plugin.Handle) (any, error) {
	params := KVCompatPDDeciderParameters{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse %s plugin config: %w", KVCompatPDDeciderPluginType, err)
		}
	}
	if params.Decider == "" {
		return nil, fmt.Errorf("%s requires the decider parameter", KVCompatPDDeciderPluginType)
	}
	return params, nil
}

// KVCompatPDDeciderPluginFactory creates a kv-compat-pd-decider.
func KVCompatPDDeciderPluginFactory(name string, rawParameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	if handle == nil {
		return nil, errors.New("plugin handle is required")
	}
	parsed, err := KVCompatPDDeciderConfigParser(rawParameters, handle)
	if err != nil {
		return nil, err
	}
	params := parsed.(KVCompatPDDeciderParameters)
	p := handle.Plugin(params.Decider)
	if p == nil {
		return nil, fmt.Errorf("%s: decider plugin not found: %s", KVCompatPDDeciderPluginType, params.Decider)
	}
	inner, ok := p.(deciderPlugin)
	if !ok {
		return nil, fmt.Errorf("%s: plugin %s is not a prefill decider", KVCompatPDDeciderPluginType, params.Decider)
	}
	if err := registerMetrics(handle.Metrics()); err != nil {
		return nil, err
	}
	d, err := NewKVCompatPDDecider(inner, params)
	if err != nil {
		return nil, err
	}
	d.typedName.Name = name
	d.spec = d.buildSpec()
	if aware, ok := inner.(kvCompatAware); ok {
		aware.useKVCompat(d)
	}
	d.handle = handle
	d.pluginState = plugin.NewPluginState(handle.Context())
	return d, nil
}

// NewKVCompatPDDecider returns a kv-compat-pd-decider that asks inner when a compatible prefill
// endpoint exists.
func NewKVCompatPDDecider(inner deciderPlugin, params KVCompatPDDeciderParameters) (*KVCompatPDDecider, error) {
	if inner == nil {
		return nil, errors.New("decider is required")
	}
	rule, err := newKVCompatRule(params)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", KVCompatPDDeciderPluginType, err)
	}
	exclusion, err := newKVCompatExclusionConfig(params)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", KVCompatPDDeciderPluginType, err)
	}
	d := &KVCompatPDDecider{
		typedName:   plugin.TypedName{Type: KVCompatPDDeciderPluginType, Name: KVCompatPDDeciderPluginType},
		decider:     inner,
		rule:        rule,
		endpoints:   map[fwkdl.ID]kvCompatEntry{},
		prefillKeys: map[string]int{},
		exclusion:   exclusion,
		state:       kvCompatExclusionState{pairs: map[string]map[string]*kvCompatPairHealth{}},
		pluginState: plugin.NewPluginState(context.Background()),
		now:         time.Now,
	}
	d.spec = d.buildSpec()
	return d, nil
}

// TypedName returns the typed name of the plugin.
func (d *KVCompatPDDecider) TypedName() plugin.TypedName {
	return d.typedName
}

// prefixMatchInfoDataKey returns the key of the wrapped decider, so the profile handler wires the
// producer that decider reads.
func (d *KVCompatPDDecider) prefixMatchInfoDataKey() plugin.DataKey {
	if consumer, ok := d.decider.(prefixMatchInfoConsumer); ok {
		return consumer.prefixMatchInfoDataKey()
	}
	return attrprefix.PrefixCacheMatchInfoDataKey
}

func (d *KVCompatPDDecider) disaggregate(ctx context.Context, request *scheduling.InferenceRequest, endpoint scheduling.Endpoint) bool {
	if endpoint == nil || endpoint.GetMetadata() == nil {
		return false
	}
	if !d.hasCompatiblePrefill(ctx, endpoint.GetMetadata()) {
		recordKVCompatDeclined(d.typedName.Name)
		log.FromContext(ctx).V(logging.DEBUG).Info("kv-compat-pd-decider: no prefill endpoint can exchange KV with the decode endpoint",
			"decode", endpoint.GetMetadata().ID)
		return false
	}
	return d.decider.disaggregate(ctx, request, endpoint)
}

// hasCompatiblePrefill reports whether a prefill-capable endpoint exists whose key the decode
// endpoint accepts and whose pair with the decode endpoint is not excluded.
func (d *KVCompatPDDecider) hasCompatiblePrefill(ctx context.Context, decode *fwkdl.EndpointMetadata) bool {
	excl := d.exclusionsFor(ctx, decode.ID)
	nowMs := d.now().UnixMilli()
	usable := func(key string) bool { return d.prefillKeys[key] > 0 && !excl.excluded(key, nowMs) }
	decodeLabels := decode.Labels
	d.mu.RLock()
	defer d.mu.RUnlock()
	if d.keyed == 0 {
		if len(excl.Until) == 0 {
			return true
		}
		for id, e := range d.endpoints {
			if e.prefill && !excl.excluded(kvCompatEndpointPairPrefix+id.String(), nowMs) {
				return true
			}
		}
		return false
	}
	decodeKey := decodeLabels[d.rule.keyLabel]
	if decodeKey == "" {
		return false
	}
	if usable(decodeKey) {
		return true
	}
	if d.rule.acceptPrefix != "" {
		for label, value := range decodeLabels {
			if value == "true" && strings.HasPrefix(label, d.rule.acceptPrefix) && usable(label[len(d.rule.acceptPrefix):]) {
				return true
			}
		}
	}
	for _, k := range d.rule.approved[decodeKey] {
		if usable(k) {
			return true
		}
	}
	return false
}

// compatible reports whether a prefill endpoint with prefillLabels can hand KV to a decode
// endpoint with decodeLabels.
func (d *KVCompatPDDecider) compatible(prefillLabels, decodeLabels map[string]string) bool {
	prefillKey, decodeKey := prefillLabels[d.rule.keyLabel], decodeLabels[d.rule.keyLabel]
	if prefillKey == "" || decodeKey == "" {
		return !d.keysInUse()
	}
	return d.rule.accepts(decodeLabels, decodeKey, prefillKey)
}

func (d *KVCompatPDDecider) keysInUse() bool {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.keyed > 0
}

// Extract updates the key index from an endpoint event.
func (d *KVCompatPDDecider) Extract(_ context.Context, event fwkdl.EndpointEvent) error {
	if event.Endpoint == nil || event.Endpoint.GetMetadata() == nil {
		return nil
	}
	md := event.Endpoint.GetMetadata()
	d.mu.Lock()
	defer d.mu.Unlock()
	if old, ok := d.endpoints[md.ID]; ok {
		d.removeLocked(old)
		delete(d.endpoints, md.ID)
	}
	if event.Type == fwkdl.EventDelete {
		d.forgetEndpoint(md.ID)
		return nil
	}
	_, prefill := prefillCapableRoles[md.Labels[bylabel.RoleLabel]]
	entry := kvCompatEntry{key: md.Labels[d.rule.keyLabel], prefill: prefill, id: md.ID.String()}
	d.endpoints[md.ID] = entry
	if entry.key != "" {
		d.keyed++
		if entry.prefill {
			d.prefillKeys[entry.key]++
		}
	}
	return nil
}

func (d *KVCompatPDDecider) removeLocked(e kvCompatEntry) {
	if e.key == "" {
		return
	}
	d.keyed--
	if e.prefill {
		if d.prefillKeys[e.key]--; d.prefillKeys[e.key] <= 0 {
			delete(d.prefillKeys, e.key)
		}
	}
}

// RegisterDependencies subscribes the decider to endpoint lifecycle events.
func (d *KVCompatPDDecider) RegisterDependencies(r fwkdl.Registrar) error {
	return r.Register(fwkdl.PendingRegistration{
		Owner:      d.TypedName(),
		SourceType: sourcenotifications.EndpointNotificationSourceType,
		Extractor:  d,
		DefaultSource: sourcenotifications.NewEndpointDataSource(sourcenotifications.EndpointNotificationSourceType,
			sourcenotifications.EndpointNotificationSourceType),
	})
}

// KVCompatFilter keeps prefill candidates that can exchange KV with the decode endpoint the
// disagg profile handler picked earlier in the cycle, using the rule of a kv-compat-pd-decider.
// Without a decode pick, as in a non-disaggregated cycle, it keeps every candidate. It never adds
// an incompatible candidate back: an empty result means no candidate can take the KV.
type KVCompatFilter struct {
	typedName plugin.TypedName
	decider   *KVCompatPDDecider
}

var (
	_ scheduling.Filter     = &KVCompatFilter{}
	_ plugin.ConsumerPlugin = &KVCompatFilter{}
)

// KVCompatFilterConfigParser parses the parameters, so the loader can resolve the decider
// reference before the factory runs.
func KVCompatFilterConfigParser(rawParameters *json.Decoder, _ plugin.Handle) (any, error) {
	params := KVCompatFilterParameters{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse %s plugin config: %w", KVCompatFilterPluginType, err)
		}
	}
	if params.Decider == "" {
		return nil, fmt.Errorf("%s requires the decider parameter", KVCompatFilterPluginType)
	}
	return params, nil
}

// KVCompatFilterPluginFactory creates a kv-compat-filter.
func KVCompatFilterPluginFactory(name string, rawParameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	if handle == nil {
		return nil, errors.New("plugin handle is required")
	}
	parsed, err := KVCompatFilterConfigParser(rawParameters, handle)
	if err != nil {
		return nil, err
	}
	params := parsed.(KVCompatFilterParameters)
	decider, ok := handle.Plugin(params.Decider).(*KVCompatPDDecider)
	if !ok {
		return nil, fmt.Errorf("%s: plugin %s is not a %s", KVCompatFilterPluginType, params.Decider, KVCompatPDDeciderPluginType)
	}
	if name == "" {
		name = KVCompatFilterPluginType
	}
	return &KVCompatFilter{typedName: plugin.TypedName{Type: KVCompatFilterPluginType, Name: name}, decider: decider}, nil
}

// TypedName returns the typed name of the plugin.
func (f *KVCompatFilter) TypedName() plugin.TypedName {
	return f.typedName
}

// Consumes declares the decode endpoint published by the disagg profile handler as optional: it
// is absent in cycles that do not disaggregate.
func (f *KVCompatFilter) Consumes() plugin.DataDependencies {
	return plugin.DataDependencies{
		Optional: map[plugin.DataKey]any{PeerEndpointAttributeKey: scheduling.Endpoint(nil)},
	}
}

// Filter keeps the candidates that can hand KV to the decode endpoint and whose pair with it is
// not excluded.
func (f *KVCompatFilter) Filter(ctx context.Context, request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) []scheduling.Endpoint {
	if request == nil {
		return endpoints
	}
	peer, ok := scheduling.ReadRequestAttribute[scheduling.Endpoint](request, PeerEndpointAttributeKey)
	if !ok || peer == nil || peer.GetMetadata() == nil {
		return endpoints
	}
	excl := f.decider.exclusionsFor(ctx, peer.GetMetadata().ID)
	if !f.decider.keysInUse() && len(excl.Until) == 0 {
		return endpoints
	}
	nowMs := f.decider.now().UnixMilli()
	decodeLabels := peer.GetMetadata().Labels
	kept := make([]scheduling.Endpoint, 0, len(endpoints))
	for _, ep := range endpoints {
		md := ep.GetMetadata()
		if md == nil || !f.decider.compatible(md.Labels, decodeLabels) ||
			excl.excluded(kvCompatPairOf(f.decider.rule, md), nowMs) {
			continue
		}
		kept = append(kept, ep)
	}
	return kept
}
