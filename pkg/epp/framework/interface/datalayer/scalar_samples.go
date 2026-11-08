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

import "time"

// ScalarSample is the last valid observation of one scalar metric.
type ScalarSample struct {
	// Value is the last valid value.
	Value float64
	// ScrapeTime is the time of the scrape that observed Value. It equals the
	// Metrics.UpdateTime written by that scrape, and is zero if the metric was
	// never observed.
	ScrapeTime time.Time
}

// FreshWithin reports whether the sample was observed within maxAge of now.
// A metric missing from later scrapes and an endpoint whose polls fail both
// stop being fresh once maxAge passes.
func (s ScalarSample) FreshWithin(maxAge time.Duration) bool {
	return !s.ScrapeTime.IsZero() && time.Since(s.ScrapeTime) <= maxAge
}

// ScalarField identifies a standard scalar metric of Metrics that carries a
// ScalarSample.
type ScalarField int

const (
	ScalarFieldWaitingQueueSize ScalarField = iota
	ScalarFieldRunningRequestsSize
	ScalarFieldKVCacheUsagePercent

	scalarFieldCount
)

// String returns the name of the Metrics field the sample describes.
func (f ScalarField) String() string {
	switch f {
	case ScalarFieldWaitingQueueSize:
		return "WaitingQueueSize"
	case ScalarFieldRunningRequestsSize:
		return "RunningRequestsSize"
	case ScalarFieldKVCacheUsagePercent:
		return "KVCacheUsagePercent"
	default:
		return "Unknown"
	}
}

// ScalarSamples holds the samples of one endpoint's scalar metrics. A value is
// immutable once attached to Metrics, so clones of Metrics share it by
// pointer. Build one with ScalarSampleUpdate.
type ScalarSamples struct {
	// origin identifies the engine mapping that produced the samples. Samples
	// recorded under another mapping describe different metrics.
	origin   any
	standard [scalarFieldCount]ScalarSample
	// custom is a slice rather than a map: a mapping configures few custom
	// metrics, and a slice copy costs one allocation per scrape where a map
	// costs two.
	custom []customScalarSample
}

type customScalarSample struct {
	attributeKey string
	sample       ScalarSample
}

// Equal lets go-cmp compare Metrics values without access to unexported fields.
func (s *ScalarSamples) Equal(other *ScalarSamples) bool {
	if s == nil || other == nil {
		return s == other
	}
	if s.origin != other.origin || s.standard != other.standard || len(s.custom) != len(other.custom) {
		return false
	}
	for _, entry := range s.custom {
		if sample, ok := other.customSample(entry.attributeKey); !ok || sample != entry.sample {
			return false
		}
	}
	return true
}

func (s *ScalarSamples) customSample(attributeKey string) (ScalarSample, bool) {
	for _, entry := range s.custom {
		if entry.attributeKey == attributeKey {
			return entry.sample, true
		}
	}
	return ScalarSample{}, false
}

// ScalarSample returns the sample of a standard scalar metric. The boolean
// reports whether the metric has been observed.
func (m *Metrics) ScalarSample(field ScalarField) (ScalarSample, bool) {
	if m == nil || m.ScalarSamples == nil || field < 0 || field >= scalarFieldCount {
		return ScalarSample{}, false
	}
	sample := m.ScalarSamples.standard[field]
	return sample, !sample.ScrapeTime.IsZero()
}

// CustomScalarSample returns the sample of the custom scalar metric configured
// as attributeKey. The boolean reports whether the metric has been observed.
func (m *Metrics) CustomScalarSample(attributeKey string) (ScalarSample, bool) {
	if m == nil || m.ScalarSamples == nil {
		return ScalarSample{}, false
	}
	return m.ScalarSamples.customSample(attributeKey)
}

// ScalarSampleUpdate derives the samples of one scrape from the previous
// snapshot. It copies the previous snapshot on the first observation only, so
// a scrape costs at most one allocation for the snapshot and one for its
// custom samples.
type ScalarSampleUpdate struct {
	prev        *ScalarSamples
	next        *ScalarSamples
	origin      any
	scrapeTime  time.Time
	customCount int
	ownsCustom  bool
}

// NewScalarSampleUpdate starts an update of prev for a scrape at scrapeTime
// under the mapping identified by origin. origin must be comparable. When it
// differs from the origin of prev, the previous samples are discarded.
// customCount is the number of custom metrics the mapping configures.
func NewScalarSampleUpdate(prev *ScalarSamples, origin any, scrapeTime time.Time, customCount int) ScalarSampleUpdate {
	if prev != nil && prev.origin != origin {
		prev = nil
	}
	return ScalarSampleUpdate{prev: prev, origin: origin, scrapeTime: scrapeTime, customCount: customCount}
}

// Observe records a valid value of a standard scalar metric.
func (u *ScalarSampleUpdate) Observe(field ScalarField, value float64) {
	if field < 0 || field >= scalarFieldCount {
		return
	}
	u.ensureNext()
	u.next.standard[field] = ScalarSample{Value: value, ScrapeTime: u.scrapeTime}
}

// ObserveCustom records a valid value of the custom scalar metric configured
// as attributeKey.
func (u *ScalarSampleUpdate) ObserveCustom(attributeKey string, value float64) {
	u.ensureNext()
	if !u.ownsCustom {
		custom := make([]customScalarSample, len(u.next.custom), max(u.customCount, len(u.next.custom)+1))
		copy(custom, u.next.custom)
		u.next.custom = custom
		u.ownsCustom = true
	}
	sample := ScalarSample{Value: value, ScrapeTime: u.scrapeTime}
	for i := range u.next.custom {
		if u.next.custom[i].attributeKey == attributeKey {
			u.next.custom[i].sample = sample
			return
		}
	}
	u.next.custom = append(u.next.custom, customScalarSample{attributeKey: attributeKey, sample: sample})
}

// Result returns the snapshot to attach to Metrics: a new one when the scrape
// observed a metric, otherwise the previous one, or nil when the mapping
// changed.
func (u *ScalarSampleUpdate) Result() *ScalarSamples {
	if u.next != nil {
		return u.next
	}
	return u.prev
}

func (u *ScalarSampleUpdate) ensureNext() {
	if u.next != nil {
		return
	}
	u.next = &ScalarSamples{origin: u.origin}
	if u.prev != nil {
		u.next.standard = u.prev.standard
		// Shared with prev until ObserveCustom copies it.
		u.next.custom = u.prev.custom
	}
}
