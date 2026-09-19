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

package interturnlatency

import (
	"math"
	"sync"
)

// minLogStd is the floor applied to the log-scale standard deviation so the
// fitted distribution never collapses to a point mass.
const minLogStd = 0.1

// logNormalEstimator fits a log-normal distribution to observed inter-turn
// intervals online. A sliding window over ln(interval) provides the
// maximum-likelihood sample estimate (mean and standard deviation of the log
// values), which is blended into the running parameters with an exponential
// moving average once the window holds minSamples observations.
type logNormalEstimator struct {
	mu sync.Mutex

	// logMean and logStd are the fitted parameters, in log-seconds.
	logMean float64
	logStd  float64

	window   []float64 // ring buffer of ln(interval) samples
	next     int       // ring buffer write position
	filled   bool      // the ring buffer has wrapped at least once
	observed int64     // total accepted observations

	ema        float64
	minSamples int
}

func newLogNormalEstimator(initialLogMean, initialLogStd, ema float64, minSamples, windowSize int) *logNormalEstimator {
	return &logNormalEstimator{
		logMean:    initialLogMean,
		logStd:     math.Max(initialLogStd, minLogStd),
		window:     make([]float64, windowSize),
		ema:        ema,
		minSamples: minSamples,
	}
}

// observe records one inter-turn interval, in seconds, and refreshes the
// fitted parameters when enough samples are buffered. Non-positive intervals
// are ignored.
func (e *logNormalEstimator) observe(seconds float64) {
	if seconds <= 0 {
		return
	}
	logInterval := math.Log(seconds)

	e.mu.Lock()
	defer e.mu.Unlock()

	e.window[e.next] = logInterval
	e.next++
	if e.next == len(e.window) {
		e.next = 0
		e.filled = true
	}
	e.observed++

	n := e.sampleCount()
	if n < e.minSamples {
		return
	}
	sampleMean, sampleStd := logMoments(e.window[:n])
	e.logMean += e.ema * (sampleMean - e.logMean)
	e.logStd += e.ema * (sampleStd - e.logStd)
	if e.logStd < minLogStd {
		e.logStd = minLogStd
	}
}

// sampleCount returns the number of valid samples in the window. Callers must
// hold e.mu.
func (e *logNormalEstimator) sampleCount() int {
	if e.filled {
		return len(e.window)
	}
	return e.next
}

// snapshot returns the fitted parameters and the total observation count.
func (e *logNormalEstimator) snapshot() (logMean, logStd float64, observed int64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.logMean, e.logStd, e.observed
}

// logMoments returns the mean and population standard deviation of values.
func logMoments(values []float64) (mean, std float64) {
	n := float64(len(values))
	for _, v := range values {
		mean += v
	}
	mean /= n

	var sumSq float64
	for _, v := range values {
		d := v - mean
		sumSq += d * d
	}
	return mean, math.Sqrt(sumSq / n)
}
