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

package contracts

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	testclock "k8s.io/utils/clock/testing"
)

func TestStandingQueueKeepsShortBudgetThroughBriefRelief(t *testing.T) {
	const long, short = 10 * time.Second, 2 * time.Second
	clk := testclock.NewFakeClock(time.Unix(100, 0))
	q := NewStandingQueue(long, short, clk)
	q.Start()
	clk.Step(long + short)
	backlogged := clk.Now()
	q.RecordDispatch(backlogged.Add(-long), backlogged)

	clk.Step(short / 2)
	q.RecordDispatch(clk.Now(), clk.Now())
	require.False(t, q.ExpiryCutoff(clk.Now(), time.Time{}).IsZero(),
		"a short wait within one short budget of a long one keeps the window")

	clk.Step(short / 2)
	q.RecordDispatch(clk.Now(), clk.Now())
	require.True(t, q.ExpiryCutoff(clk.Now(), time.Time{}).IsZero(),
		"a short wait one short budget after the last long one restarts the window")
}
