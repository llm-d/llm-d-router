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
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestLogNormalEstimator_ConvergesToSampledDistribution(t *testing.T) {
	t.Parallel()

	const (
		trueLogMean = 1.8
		trueLogStd  = 1.1
	)
	// Seeded far from the target so convergence is attributable to learning.
	estimator := newLogNormalEstimator(5.0, 0.5, 0.1, 20, 200)

	rng := rand.New(rand.NewSource(42))
	for range 5000 {
		interval := math.Exp(trueLogMean + trueLogStd*rng.NormFloat64())
		estimator.observe(interval)
	}

	logMean, logStd, observed := estimator.snapshot()
	assert.EqualValues(t, 5000, observed)
	assert.InDelta(t, trueLogMean, logMean, 0.15)
	assert.InDelta(t, trueLogStd, logStd, 0.15)
}

func TestLogNormalEstimator_NoUpdateBelowMinSamples(t *testing.T) {
	t.Parallel()

	estimator := newLogNormalEstimator(2.0, 1.0, 0.5, 10, 100)
	for range 9 {
		estimator.observe(1000)
	}

	logMean, logStd, observed := estimator.snapshot()
	assert.EqualValues(t, 9, observed)
	assert.Equal(t, 2.0, logMean)
	assert.Equal(t, 1.0, logStd)
}

func TestLogNormalEstimator_IgnoresNonPositiveIntervals(t *testing.T) {
	t.Parallel()

	estimator := newLogNormalEstimator(2.0, 1.0, 0.5, 2, 10)
	estimator.observe(0)
	estimator.observe(-3)

	_, _, observed := estimator.snapshot()
	assert.EqualValues(t, 0, observed)
}

func TestLogNormalEstimator_StdFloor(t *testing.T) {
	t.Parallel()

	estimator := newLogNormalEstimator(2.0, 1.0, 1.0, 2, 10)
	// Identical intervals: the sample standard deviation is zero, so the
	// floor must hold.
	for range 10 {
		estimator.observe(10)
	}

	_, logStd, _ := estimator.snapshot()
	assert.Equal(t, minLogStd, logStd)
}
