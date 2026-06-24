package preciseprefixcache

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestHealthMonitor_RecordConfirmedEntry(t *testing.T) {
	m := NewKVEventsHealthMonitor(true)

	before := time.Now()
	m.RecordConfirmedEntry("10.0.0.1:8000")
	after := time.Now()

	lastConfirmed, _, known := m.GetHealthStatus("10.0.0.1:8000")
	assert.True(t, known)
	assert.True(t, !lastConfirmed.Before(before) && !lastConfirmed.After(after))
}

func TestHealthMonitor_RecordRouting(t *testing.T) {
	m := NewKVEventsHealthMonitor(true)

	before := time.Now()
	m.RecordRouting("10.0.0.1:8000")
	after := time.Now()

	_, lastRouted, known := m.GetHealthStatus("10.0.0.1:8000")
	assert.True(t, known)
	assert.True(t, !lastRouted.Before(before) && !lastRouted.After(after))
}

func TestHealthMonitor_UnknownEndpoint(t *testing.T) {
	m := NewKVEventsHealthMonitor(true)

	_, _, known := m.GetHealthStatus("unknown")
	assert.False(t, known)
}

func TestHealthMonitor_RemoveEndpoint(t *testing.T) {
	m := NewKVEventsHealthMonitor(true)

	m.RecordRouting("10.0.0.1:8000")
	_, _, known := m.GetHealthStatus("10.0.0.1:8000")
	assert.True(t, known)

	m.RemoveEndpoint("10.0.0.1:8000")
	_, _, known = m.GetHealthStatus("10.0.0.1:8000")
	assert.False(t, known)
}

func TestHealthMonitor_HasKVEventsConfig(t *testing.T) {
	withCfg := NewKVEventsHealthMonitor(true)
	assert.True(t, withCfg.HasKVEventsConfig())

	withoutCfg := NewKVEventsHealthMonitor(false)
	assert.False(t, withoutCfg.HasKVEventsConfig())
}

func TestHealthMonitor_MultipleEndpoints(t *testing.T) {
	m := NewKVEventsHealthMonitor(true)

	m.RecordRouting("10.0.0.1:8000")
	m.RecordConfirmedEntry("10.0.0.2:8000")

	_, lastRouted1, known1 := m.GetHealthStatus("10.0.0.1:8000")
	assert.True(t, known1)
	assert.False(t, lastRouted1.IsZero())

	lastConfirmed2, _, known2 := m.GetHealthStatus("10.0.0.2:8000")
	assert.True(t, known2)
	assert.False(t, lastConfirmed2.IsZero())
}

func TestHealthMonitor_DistinguishBrokenFromIdle(t *testing.T) {
	m := NewKVEventsHealthMonitor(true)

	// Endpoint A: routing but no confirmed events → pipeline broken
	m.RecordRouting("pod-a")
	lastConfirmedA, lastRoutedA, _ := m.GetHealthStatus("pod-a")
	assert.True(t, lastConfirmedA.IsZero(), "no confirmed events yet")
	assert.False(t, lastRoutedA.IsZero(), "was routed")

	// Endpoint B: no routing, no events → idle
	_, _, knownB := m.GetHealthStatus("pod-b")
	assert.False(t, knownB, "never seen")

	// Endpoint C: routing and confirmed → healthy
	m.RecordRouting("pod-c")
	m.RecordConfirmedEntry("pod-c")
	lastConfirmedC, lastRoutedC, _ := m.GetHealthStatus("pod-c")
	assert.False(t, lastConfirmedC.IsZero(), "has confirmed events")
	assert.False(t, lastRoutedC.IsZero(), "was routed")
}
