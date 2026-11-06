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

// Package capacity defines the capacity ledger's per-endpoint view. The types are Alpha: their
// fields and Eligibility values may change.
package capacity

import (
	"time"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	capacityledgerconstants "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/capacityledger/constants"
)

// EndpointCapacityDataKey carries the capacity ledger's per-endpoint view. The ledger installs it
// as a dynamic attribute, so each read reflects the ledger's state at that moment.
var EndpointCapacityDataKey = plugin.NewDataKey("EndpointCapacityDataKey", capacityledgerconstants.CapacityLedgerType)

// Eligibility states whether the ledger's accounting holds for an endpoint, and if not, why.
type Eligibility string

const (
	// Eligible means every premise of the accounting holds.
	Eligible Eligibility = "eligible"
	// Stale means the endpoint's metrics are older than the staleness threshold, or were never
	// scraped.
	Stale Eligibility = "stale"
	// NoBlockCount means the engine reports no KV cache block count or block size.
	NoBlockCount Eligibility = "no-block-count"
	// Disaggregated means the endpoint serves one stage of prefill/decode disaggregation.
	Disaggregated Eligibility = "disaggregated"
)

// Axis is the ledger's account of one resource of an endpoint.
type Axis struct {
	// Capacity is the resource's limit.
	Capacity int64 `json:"capacity"`
	// Booked is the ledger's own account: the sum of its placed requests' charges.
	Booked int64 `json:"booked"`
	// Scraped is the engine's latest report; 0 on an axis the engine does not report.
	Scraped int64 `json:"scraped"`
	// Used is the endpoint's state: Scraped plus the bookings made since that scrape, and never
	// less than Booked.
	Used int64 `json:"used"`
}

// EndpointCapacity is the capacity ledger's view of one endpoint.
type EndpointCapacity struct {
	// Memory is KV cache blocks.
	Memory Axis `json:"memory"`
	// Step is tokens committed to the engine's next step: prompts not yet prefilled plus one
	// decode step for each running sequence. Capacity is the engine's per-step token budget.
	// Scraped is one decode step per running request, since the engine reports no pending prefill.
	Step Axis `json:"step"`
	// Slots is concurrent sequences.
	Slots Axis `json:"slots"`
	// BlockSize is the engine's KV cache block size in tokens.
	BlockSize int64 `json:"blockSize"`
	// Eligibility states whether the accounting holds for the endpoint.
	Eligibility Eligibility `json:"eligibility"`
	// ScrapeTime is when the endpoint's metrics were last scraped; zero if never.
	ScrapeTime time.Time `json:"scrapeTime"`
}

// Clone returns an independent copy of the EndpointCapacity.
func (c *EndpointCapacity) Clone() fwkdl.Cloneable {
	if c == nil {
		return nil
	}
	cp := *c
	return &cp
}
