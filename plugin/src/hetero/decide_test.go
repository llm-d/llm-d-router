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

package hetero

import (
	"math"
	"testing"
)

const (
	classA30    = "nvidia-a30"
	classGaudi3 = "intel-gaudi3"
	promptLong  = 2000
)

// labConfig models one A30 prefill pod feeding Gaudi 3 decode pods over
// 25 GbE with Llama-3.1-8B. Throughputs are placeholders, not measurements.
func labConfig() Config {
	c := DefaultConfig()
	c.Strategy = StrategyModel
	// The cost-model tests leave KV compatibility out; it has its own tests.
	c.Compatibility = CompatibilityOff
	c.TTFTSLOMs = 1000
	c.BandwidthGbps = 25
	c.Profiles = map[string]Profile{
		classA30:    {PrefillTokensPerSec: 5000},
		classGaudi3: {PrefillTokensPerSec: 20000},
	}
	return c
}

func a30(waiting int) Pod {
	return Pod{Name: "a30-0", Class: classA30, Waiting: waiting}
}

func gaudi(running int) Pod {
	return Pod{Name: "gaudi-0", Class: classGaudi3, Running: running}
}

func TestDecideLabScenarios(t *testing.T) {
	tests := []struct {
		name       string
		tokens     int
		decode     Pod
		prefill    []Pod
		wantRemote bool
		wantReason string
	}{
		{"long prompt, busy decode pod offloads", promptLong, gaudi(40), []Pod{a30(0)}, true, ReasonRemoteBetter},
		{"long prompt, idle decode pod prefills locally", promptLong, gaudi(0), []Pod{a30(0)}, false, ReasonLocalBetter},
		{"short prompt stays local", 200, gaudi(40), []Pod{a30(0)}, false, ReasonShortPrompt},
		{"prefill queue breaks the SLO", promptLong, gaudi(40), []Pod{a30(3)}, false, ReasonSLOMiss},
		{"decode KV nearly full", promptLong, Pod{Name: "gaudi-0", Class: classGaudi3, Running: 40, KVUsage: 0.95}, []Pod{a30(0)}, false, ReasonDecodeKVFull},
		{"decode pod opted out", promptLong, Pod{Name: "gaudi-0", Class: classGaudi3, Running: 40, Disabled: true}, []Pod{a30(0)}, false, ReasonDecodeDisabled},
		{"no prefill pods", promptLong, gaudi(40), nil, false, ReasonNoPrefillCandidate},
		{"decode pod is not its own prefill candidate", promptLong, gaudi(40), []Pod{gaudi(0)}, false, ReasonNoPrefillCandidate},
		{"unknown decode class falls back", promptLong, Pod{Name: "x", Class: "unknown", Running: 40}, []Pod{a30(0)}, false, ReasonMissingProfile},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := Decide(labConfig(), tt.tokens, tt.decode, tt.prefill)
			if d.Remote != tt.wantRemote || d.Reason != tt.wantReason {
				t.Fatalf("got remote=%v reason=%q, want remote=%v reason=%q (%+v)",
					d.Remote, d.Reason, tt.wantRemote, tt.wantReason, d)
			}
		})
	}
}

func TestDecideEstimates(t *testing.T) {
	d := Decide(labConfig(), promptLong, gaudi(40), []Pod{a30(0)})
	// 2000 tokens / 5000 tok/s + 2000 * 131072 B / 3.125e9 B/s
	wantTransfer := 2000.0 * 131072 / 3.125e9
	assertNear(t, "transfer", d.TransferSeconds, wantTransfer)
	assertNear(t, "remote", d.RemoteSeconds, 0.4+wantTransfer)
	assertNear(t, "local", d.LocalSeconds, 0.1)
	assertNear(t, "stall", d.StallSeconds, 4.0)
	if d.Prefill != "a30-0" {
		t.Fatalf("prefill pod %q, want a30-0", d.Prefill)
	}
}

func TestDecideMissingProfileFallsBack(t *testing.T) {
	d := Decide(labConfig(), promptLong, Pod{Name: "x", Class: "unknown"}, []Pod{a30(0)})
	if !d.Fallback {
		t.Fatalf("want fallback, got %+v", d)
	}
}

func TestDecideBurstSplitsUntilSLO(t *testing.T) {
	// Each request queued on the A30 adds AvgQueuedPromptTokens / 5000 s.
	remote := 0
	for waiting := 0; waiting < 10; waiting++ {
		if Decide(labConfig(), promptLong, gaudi(40), []Pod{a30(waiting)}).Remote {
			remote++
		}
	}
	if remote != 3 {
		t.Fatalf("remote decisions in a burst of 10 = %d, want 3", remote)
	}
}

func TestDecideStallCrossover(t *testing.T) {
	// Busy decode pods offload; lightly loaded ones keep prefill local.
	var firstRemote = -1
	for running := 0; running <= 64; running++ {
		if Decide(labConfig(), promptLong, gaudi(running), []Pod{a30(0)}).Remote {
			firstRemote = running
			break
		}
	}
	if firstRemote < 10 || firstRemote > 30 {
		t.Fatalf("first offload at %d running streams, want between 10 and 30", firstRemote)
	}
	for running := firstRemote; running <= 64; running++ {
		if !Decide(labConfig(), promptLong, gaudi(running), []Pod{a30(0)}).Remote {
			t.Fatalf("decision flipped back to local at %d running streams", running)
		}
	}
}

func TestDecidePicksFastestPrefillCandidate(t *testing.T) {
	c := labConfig()
	c.Profiles["nvidia-b200"] = Profile{PrefillTokensPerSec: 50000}
	b200Busy := Pod{Name: "b200-0", Class: "nvidia-b200", Waiting: 40}
	d := Decide(c, promptLong, gaudi(40), []Pod{a30(0), b200Busy})
	if d.Prefill != "a30-0" {
		t.Fatalf("with a queued B200 want a30-0, got %q", d.Prefill)
	}
	b200Idle := Pod{Name: "b200-0", Class: "nvidia-b200"}
	d = Decide(c, promptLong, gaudi(40), []Pod{a30(0), b200Idle})
	if d.Prefill != "b200-0" {
		t.Fatalf("with an idle B200 want b200-0, got %q", d.Prefill)
	}
}

func TestDecideSameClassBandwidth(t *testing.T) {
	c := labConfig()
	c.SameClassBandwidthGbps = 200
	peer := Pod{Name: "gaudi-1", Class: classGaudi3}
	d := Decide(c, promptLong, gaudi(40), []Pod{a30(0), peer})
	if d.Prefill != "gaudi-1" || !d.Remote {
		t.Fatalf("want remote prefill on the idle Gaudi peer, got %+v", d)
	}
	assertNear(t, "transfer", d.TransferSeconds, 2000.0*131072/25e9)
}

func TestDecideCompatKeyMismatch(t *testing.T) {
	c := labConfig()
	c.Compatibility = CompatibilityAuto
	decode := gaudi(40)
	decode.CompatKey = "qwen7b|hpu|hetero-layout"
	other := a30(0)
	other.CompatKey = "qwen7b|flash-attn|hnd"
	d := Decide(c, promptLong, decode, []Pod{other})
	if d.Remote || d.Reason != ReasonNoCompatiblePrefill {
		t.Fatalf("different keys must not split, got %+v", d)
	}
	c.ApprovedPairs = []string{other.CompatKey + "->" + decode.CompatKey}
	if !Decide(c, promptLong, decode, []Pod{other}).Remote {
		t.Fatal("an approved pair should split")
	}
	// Approval is directional: the reverse direction stays blocked.
	rev := Decide(c, promptLong, Pod{Name: "a30-d", Class: classA30, CompatKey: other.CompatKey, Running: 40},
		[]Pod{{Name: "gaudi-p", Class: classGaudi3, CompatKey: decode.CompatKey}})
	if rev.Remote || rev.Reason != ReasonNoCompatiblePrefill {
		t.Fatalf("approval must not apply in reverse, got %+v", rev)
	}
	// A live approval on the decode pod works like an approved pair, in one direction.
	c.ApprovedPairs = nil
	if Decide(c, promptLong, decode, []Pod{other}).Remote {
		t.Fatal("without approval different keys must not split")
	}
	approved := decode
	approved.Accepts = []string{"something-else", other.CompatKey}
	if !Decide(c, promptLong, approved, []Pod{other}).Remote {
		t.Fatal("a decode pod that accepts the prefill key should split")
	}
	other.CompatKey = decode.CompatKey
	if !Decide(c, promptLong, decode, []Pod{other}).Remote {
		t.Fatal("equal keys should split")
	}
}

func TestMixed(t *testing.T) {
	a, b := Pod{Name: "a", Class: classA30}, Pod{Name: "b", Class: classA30}
	if Mixed(a, []Pod{b}) {
		t.Fatal("one class is not mixed")
	}
	if !Mixed(a, []Pod{b, {Name: "g", Class: classGaudi3}}) {
		t.Fatal("two classes are mixed")
	}
	if !Mixed(a, []Pod{{Name: "u"}}) {
		t.Fatal("a labelled pod next to an unlabelled one is mixed")
	}
	if Mixed(Pod{Name: "x"}, []Pod{{Name: "y"}}) {
		t.Fatal("a pool without class labels is one class")
	}
	if Mixed(a, nil) {
		t.Fatal("no candidates is not mixed")
	}
}

func TestCompatibleModes(t *testing.T) {
	keyed := func(name, class, key string) Pod { return Pod{Name: name, Class: class, CompatKey: key} }
	tests := []struct {
		name  string
		mode  string
		p, d  Pod
		mixed bool
		want  bool
	}{
		{"auto, same key", CompatibilityAuto, keyed("p", classA30, "k1"), keyed("d", classGaudi3, "k1"), true, true},
		{"auto, different keys", CompatibilityAuto, keyed("p", classA30, "k1"), keyed("d", classA30, "k2"), false, false},
		{"auto, missing key, one class", CompatibilityAuto, keyed("p", classA30, ""), keyed("d", classA30, ""), false, true},
		{"auto, missing key, mixed pool", CompatibilityAuto, keyed("p", classA30, ""), keyed("d", classGaudi3, "k1"), true, false},
		{"empty mode is auto", "", keyed("p", classA30, ""), keyed("d", classGaudi3, ""), true, false},
		{"strict, missing key, one class", CompatibilityStrict, keyed("p", classA30, ""), keyed("d", classA30, "k1"), false, false},
		{"strict, same key", CompatibilityStrict, keyed("p", classA30, "k1"), keyed("d", classA30, "k1"), false, true},
		{"off, different keys", CompatibilityOff, keyed("p", classA30, "k1"), keyed("d", classGaudi3, "k2"), true, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := DefaultConfig()
			c.Compatibility = tt.mode
			if got := c.Compatible(tt.p, tt.d, tt.mixed); got != tt.want {
				t.Fatalf("Compatible = %v, want %v", got, tt.want)
			}
		})
	}
}

// A mixed pool without compatibility keys never splits, whatever the strategy, so a
// pair that cannot exchange KV never reaches a handshake.
func TestDecideMixedPoolWithoutKeysKeepsLocal(t *testing.T) {
	for _, strategy := range []string{StrategyAuto, StrategyModel, StrategyGuardedStock} {
		t.Run(strategy, func(t *testing.T) {
			c := labConfig()
			c.Compatibility = CompatibilityAuto
			c.Strategy = strategy
			c.StockNonCachedTokens = 1024
			d := Decide(c, promptLong, gaudi(40), []Pod{a30(0)})
			if d.Remote || d.Reason != ReasonNoCompatiblePrefill {
				t.Fatalf("got %+v", d)
			}
		})
	}
}

func TestDecideExcludedPairIsNotACandidate(t *testing.T) {
	c := labConfig()
	p := a30(0)
	p.Excluded = true
	d := Decide(c, promptLong, gaudi(40), []Pod{p})
	if d.Remote || d.Reason != ReasonNoCompatiblePrefill {
		t.Fatalf("an excluded pair must not be used, got %+v", d)
	}
	other := Pod{Name: "a30-1", Class: classA30}
	if d := Decide(c, promptLong, gaudi(40), []Pod{p, other}); !d.Remote || d.Prefill != "a30-1" {
		t.Fatalf("want the remaining pod, got %+v", d)
	}
}

func TestDecideModelFallbackNamesCompatiblePod(t *testing.T) {
	c := labConfig()
	delete(c.Profiles, classGaudi3)
	d := Decide(c, promptLong, gaudi(40), []Pod{a30(0)})
	if !d.Fallback || d.Prefill != "a30-0" {
		t.Fatalf("fallback should name the compatible pod, got %+v", d)
	}
	c.Compatibility = CompatibilityAuto
	if d := Decide(c, promptLong, gaudi(40), []Pod{a30(0)}); d.Fallback || d.Reason != ReasonNoCompatiblePrefill {
		t.Fatalf("without a compatible pod there is nothing to fall back to, got %+v", d)
	}
}

func TestDecideNoCandidateVersusNoCompatible(t *testing.T) {
	c := labConfig()
	c.Compatibility = CompatibilityAuto
	if d := Decide(c, promptLong, gaudi(40), nil); d.Reason != ReasonNoPrefillCandidate {
		t.Fatalf("no pods at all: got %q", d.Reason)
	}
	off := a30(0)
	off.Disabled = true
	if d := Decide(c, promptLong, gaudi(40), []Pod{off}); d.Reason != ReasonNoPrefillCandidate {
		t.Fatalf("only a disabled pod: got %q", d.Reason)
	}
}

func TestDecideZeroSLODisablesCheck(t *testing.T) {
	c := labConfig()
	c.TTFTSLOMs = 0
	if !Decide(c, promptLong, gaudi(200), []Pod{a30(20)}).Remote {
		t.Fatal("without an SLO a heavily loaded decode pod should still offload")
	}
}

func TestConfigValidate(t *testing.T) {
	if err := labConfig().Validate(); err != nil {
		t.Fatalf("lab config invalid: %v", err)
	}
	bad := []func(*Config){
		func(c *Config) { c.KVBytesPerToken = 0 },
		func(c *Config) { c.TTFTSLOMs = -1 },
		func(c *Config) { c.StallWeight = -1 },
		func(c *Config) { c.SafetyMargin = 1 },
		func(c *Config) { c.MaxDecodeKVUsage = 0 },
		func(c *Config) { c.MaxDecodeKVUsage = 1.5 },
		func(c *Config) { c.MinUncachedTokens = -1 },
		func(c *Config) { c.BandwidthGbps = 0 },
		func(c *Config) { c.SameClassBandwidthGbps = -1 },
		func(c *Config) { c.AvgQueuedPromptTokens = -1 },
		func(c *Config) { c.HandoffOverheadMs = -1 },
		func(c *Config) { c.StallFraction = -0.1 },
		func(c *Config) { c.StallFraction = 1.1 },
		func(c *Config) { c.Profiles["bad"] = Profile{} },
		func(c *Config) { c.Strategy = "stock" },
		func(c *Config) { c.StockNonCachedTokens = 0 },
		func(c *Config) { c.IdleRunningRequests = -1 },
		func(c *Config) { c.OverrideMargin = 1 },
		func(c *Config) { c.OverrideMargin = -0.1 },
		func(c *Config) { c.Compatibility = "loose" },
		func(c *Config) { c.ApprovedPairs = []string{"k1"} },
		func(c *Config) { c.ApprovedPairs = []string{"->k2"} },
		func(c *Config) { c.FailureThreshold = -1 },
		func(c *Config) { c.ExclusionSeconds = -1 },
		func(c *Config) { c.FailureThreshold, c.ExclusionSeconds = 3, 0 },
	}
	for i, mutate := range bad {
		c := labConfig()
		mutate(&c)
		if c.Validate() == nil {
			t.Errorf("case %d: invalid config accepted", i)
		}
	}
}

func assertNear(t *testing.T, name string, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-9 {
		t.Fatalf("%s = %v, want %v", name, got, want)
	}
}

// sameClassLab mirrors the A30-to-A30 lab: both pods are the same class and speed.
func sameClassLab() (Config, Pod, Pod) {
	c := labConfig()
	c.Profiles[classA30] = Profile{PrefillTokensPerSec: 5000}
	decode := Pod{Name: "a30-decode", Class: classA30, Running: 32}
	prefill := Pod{Name: "a30-prefill", Class: classA30}
	return c, decode, prefill
}

func TestDecideDefaultsKeepPreviousBehaviour(t *testing.T) {
	c := DefaultConfig()
	if c.HandoffOverheadMs != 0 || c.StallFraction != 1 {
		t.Fatalf("defaults changed: handoff %v, stallFraction %v", c.HandoffOverheadMs, c.StallFraction)
	}
}

func TestDecideHandoffOverhead(t *testing.T) {
	c, decode, prefill := sameClassLab()
	c.StallFraction = 0.1
	// A 300-token chat message on a busy pod moves when the handoff is free...
	if d := Decide(c, 300, decode, []Pod{prefill}); !d.Remote {
		t.Fatalf("without overhead: want remote, got %+v", d)
	}
	// ...and stays local once the measured 300 ms handoff cost is counted.
	c.HandoffOverheadMs = 300
	d := Decide(c, 300, decode, []Pod{prefill})
	if d.Remote || d.Reason != ReasonLocalBetter {
		t.Fatalf("with 300 ms overhead: want local-better, got %+v", d)
	}
	assertNear(t, "HandoffSeconds", d.HandoffSeconds, 0.3)
	assertNear(t, "RemoteSeconds", d.RemoteSeconds, 300.0/5000+d.TransferSeconds+0.3)
}

func TestDecideStallFraction(t *testing.T) {
	c, decode, prefill := sameClassLab()
	c.HandoffOverheadMs = 50
	full := Decide(c, 300, decode, []Pod{prefill})
	c.StallFraction = 0.1
	chunked := Decide(c, 300, decode, []Pod{prefill})
	assertNear(t, "stall scales with fraction", chunked.StallSeconds, 0.1*full.StallSeconds)
	if !full.Remote || chunked.Remote {
		t.Fatalf("want remote with full-freeze stall and local with chunked stall; got %v and %v", full.Remote, chunked.Remote)
	}
}

func TestDecideCalibratedLabScenarios(t *testing.T) {
	// The A30-to-A30 suite with calibrated settings: chat and documents stay local on
	// identical cards, because moving costs a handoff and saves little.
	c, decode, prefill := sameClassLab()
	c.HandoffOverheadMs = 300
	c.StallFraction = 0.1
	for _, tc := range []struct {
		name    string
		tokens  int
		running int
	}{{"busy chat", 300, 32}, {"busy medium document", 755, 32}, {"busy long document", 2126, 32}, {"idle long document", 2126, 0}, {"light chat", 300, 8}} {
		decode.Running = tc.running
		if d := Decide(c, tc.tokens, decode, []Pod{prefill}); d.Remote {
			t.Errorf("%s: want local, got %+v", tc.name, d)
		}
	}
	// A slower prefill card feeding a much faster decode card under heavy load still offloads
	// a long prompt when the stall is large enough to pay for the handoff.
	g := labConfig()
	g.HandoffOverheadMs = 300
	g.StallFraction = 0.5
	if d := Decide(g, promptLong, gaudi(64), []Pod{a30(0)}); !d.Remote {
		t.Errorf("busy Gaudi, long prompt: want remote, got %+v", d)
	}
}

// calibratedA30Pair is the A30-to-A30 lab after calibration.
func calibratedA30Pair(strategy string) (Config, Pod, Pod) {
	c := DefaultConfig()
	c.Strategy = strategy
	c.Compatibility = CompatibilityOff // cost-model tests; compatibility has its own tests
	c.KVBytesPerToken = 57344
	c.TTFTSLOMs = 1000
	c.BandwidthGbps = 10
	c.HandoffOverheadMs = 82
	c.StallFraction = 0.1
	c.Profiles = map[string]Profile{classA30: {PrefillTokensPerSec: 7035}, classGaudi3: {PrefillTokensPerSec: 20000}}
	return c, Pod{Name: "a30-decode", Class: classA30}, Pod{Name: "a30-prefill", Class: classA30}
}

func TestDecideDefaultStrategyIsAuto(t *testing.T) {
	if DefaultConfig().Strategy != StrategyAuto {
		t.Fatal("default strategy should be auto")
	}
	c, decode, prefill := calibratedA30Pair("")
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
	if d := Decide(c, promptLong, decode, []Pod{prefill}); d.Strategy != StrategyAuto {
		t.Fatalf("empty strategy: got %+v, want auto", d)
	}
}

func TestDecideGuardedStock(t *testing.T) {
	c, idle, prefill := calibratedA30Pair(StrategyGuardedStock)
	busy := idle
	busy.Running = 32
	queued := idle
	queued.Waiting = 3
	full := busy
	full.KVUsage = 0.95
	off := busy
	off.Disabled = true
	cases := []struct {
		name     string
		tokens   int
		decode   Pod
		prefill  []Pod
		remote   bool
		reason   string
		estimate bool
	}{
		{"chat below stock threshold", 250, busy, []Pod{prefill}, false, ReasonStockKeep, false},
		{"medium doc below threshold", 700, busy, []Pod{prefill}, false, ReasonStockKeep, false},
		{"busy long doc follows stock", 2000, busy, []Pod{prefill}, true, ReasonStockSplit, true},
		{"busy huge doc follows stock past the SLO", 6000, busy, []Pod{prefill}, true, ReasonStockSplit, true},
		{"idle long doc is vetoed", 2000, idle, []Pod{prefill}, false, ReasonIdleVeto, true},
		{"queued decode pod is not idle", 2000, queued, []Pod{prefill}, true, ReasonStockSplit, true},
		{"kv full", 2000, full, []Pod{prefill}, false, ReasonDecodeKVFull, false},
		{"decode disabled", 2000, off, []Pod{prefill}, false, ReasonDecodeDisabled, false},
		{"no prefill pod", 2000, busy, nil, false, ReasonNoPrefillCandidate, false},
		{"only itself", 2000, busy, []Pod{busy}, false, ReasonNoPrefillCandidate, false},
	}
	for _, tc := range cases {
		d := Decide(c, tc.tokens, tc.decode, tc.prefill)
		if d.Remote != tc.remote || d.Reason != tc.reason || d.Strategy != StrategyGuardedStock {
			t.Errorf("%s: got remote=%v reason=%s strategy=%s, want %v %s", tc.name, d.Remote, d.Reason, d.Strategy, tc.remote, tc.reason)
		}
		if tc.estimate && (d.RemoteSeconds <= 0 || d.LocalSeconds <= 0) {
			t.Errorf("%s: estimates missing: %+v", tc.name, d)
		}
		if d.Remote && d.Prefill != prefill.Name {
			t.Errorf("%s: prefill %q", tc.name, d.Prefill)
		}
	}
}

func TestDecideGuardedStockIdleButFasterRemoteSplits(t *testing.T) {
	// Idle A30 decode, Gaudi 3 prefill: remote is faster, so the guard lets the split through.
	c, idle, _ := calibratedA30Pair(StrategyGuardedStock)
	c.BandwidthGbps = 100
	d := Decide(c, 6000, idle, []Pod{{Name: "gaudi-prefill", Class: classGaudi3}})
	if !d.Remote || d.Reason != ReasonStockSplit || d.Prefill != "gaudi-prefill" {
		t.Fatalf("got %+v", d)
	}
}

func TestDecideGuardedStockWithoutProfilesIsStock(t *testing.T) {
	c, idle, prefill := calibratedA30Pair(StrategyGuardedStock)
	c.Profiles = map[string]Profile{}
	d := Decide(c, 2000, idle, []Pod{prefill})
	if !d.Remote || d.Reason != ReasonStockSplit || d.RemoteSeconds != 0 {
		t.Fatalf("got %+v, want a plain stock split", d)
	}
}

func TestDecideAuto(t *testing.T) {
	c, idle, a30Prefill := calibratedA30Pair(StrategyAuto)
	busy := idle
	busy.Running = 32
	gaudiPrefill := Pod{Name: "gaudi-prefill", Class: classGaudi3}
	gaudiDecode := Pod{Name: "gaudi-decode", Class: classGaudi3, Running: 32}
	full := busy
	full.KVUsage = 0.95
	queuedBusy := busy
	queuedBusy.Waiting = 10
	cases := []struct {
		name    string
		tokens  int
		decode  Pod
		prefill []Pod
		remote  bool
		reason  string
	}{
		{"A30 pair busy chat: agrees with stock keep", 300, busy, []Pod{a30Prefill}, false, ReasonStockKeep},
		{"A30 pair busy medium doc: agrees with stock keep", 700, busy, []Pod{a30Prefill}, false, ReasonStockKeep},
		{"A30 pair busy long doc: keep not clearly cheaper, stock move stands", 2000, busy, []Pod{a30Prefill}, true, ReasonStockSplit},
		{"A30 pair busy huge report: stock move stands", 6000, busy, []Pod{a30Prefill}, true, ReasonStockSplit},
		{"A30 pair idle long doc: overrules to keep", 2000, idle, []Pod{a30Prefill}, false, ReasonOverrideKeep},
		{"A30 decode, Gaudi prefill, long doc: stock move stands", 2000, busy, []Pod{gaudiPrefill}, true, ReasonStockSplit},
		{"Gaudi decode, A30 prefill, long doc: overrules to keep", 2000, gaudiDecode, []Pod{a30Prefill}, false, ReasonOverrideKeep},
		{"A30 decode, Gaudi prefill, medium doc: 17% is not sure enough", 700, busy, []Pod{gaudiPrefill}, false, ReasonStockKeep},
		{"short prompt never moves", 100, busy, []Pod{gaudiPrefill}, false, ReasonShortPrompt},
		{"kv full", 2000, full, []Pod{a30Prefill}, false, ReasonDecodeKVFull},
		{"no reader", 2000, busy, nil, false, ReasonNoPrefillCandidate},
		{"unprofiled decode: stock answer stands", 2000, Pod{Name: "h100", Class: "nvidia-h100", Running: 32}, []Pod{a30Prefill}, true, ReasonStockSplit},
		{"unprofiled reader: stock answer stands", 700, busy, []Pod{{Name: "mi300", Class: "amd-mi300x"}}, false, ReasonStockKeep},
		{"unprofiled reader, long doc: stock move stands", 2000, busy, []Pod{{Name: "mi300", Class: "amd-mi300x"}}, true, ReasonStockSplit},
		{"nothing uncached", 0, busy, []Pod{a30Prefill}, false, ReasonShortPrompt},
		{"decode disabled", 2000, Pod{Name: "off", Class: classA30, Disabled: true}, []Pod{a30Prefill}, false, ReasonDecodeDisabled}, {"queued decode pod: queue counts for both paths, chat stays", 300, queuedBusy, []Pod{a30Prefill}, false, ReasonStockKeep},
		{"queued decode pod: medium doc stays", 700, queuedBusy, []Pod{a30Prefill}, false, ReasonStockKeep},
		{"queued decode pod, idle Gaudi: 900 tokens still moves on speed", 900, queuedBusy, []Pod{gaudiPrefill}, true, ReasonOverrideMove},
	}
	for _, tc := range cases {
		d := Decide(c, tc.tokens, tc.decode, tc.prefill)
		if d.Remote != tc.remote || d.Reason != tc.reason || d.Strategy != StrategyAuto {
			t.Errorf("%s: got remote=%v reason=%s (keep %.3f+stall %.3f, move %.3f), want %v %s",
				tc.name, d.Remote, d.Reason, d.LocalSeconds, d.StallSeconds, d.RemoteSeconds, tc.remote, tc.reason)
		}
	}
	// A clearly cheaper move overrules a stock keep: busy A30 decode, idle Gaudi reader, 900 tokens, fast link.
	c.BandwidthGbps = 100
	c.HandoffOverheadMs = 20
	c.StallFraction = 0.3
	d := Decide(c, 900, busy, []Pod{gaudiPrefill})
	if !d.Remote || d.Reason != ReasonOverrideMove || d.Prefill != "gaudi-prefill" {
		t.Errorf("override move: got %+v", d)
	}
	// ...but not when the move would miss the SLO.
	c.TTFTSLOMs = 50
	if d := Decide(c, 900, busy, []Pod{gaudiPrefill}); d.Remote || d.Reason != ReasonStockKeep {
		t.Errorf("override move past SLO: got %+v", d)
	}
	// Margin 0 overrules on any saving.
	c.TTFTSLOMs = 0
	c.OverrideMargin = 0
	c.BandwidthGbps, c.HandoffOverheadMs, c.StallFraction = 10, 82, 0.1
	if d := Decide(c, 700, busy, []Pod{gaudiPrefill}); !d.Remote || d.Reason != ReasonOverrideMove {
		t.Errorf("margin 0: got %+v", d)
	}
}

func TestDecideGuardedStockUnprofiledReaderIsStock(t *testing.T) {
	c, idle, _ := calibratedA30Pair(StrategyGuardedStock)
	d := Decide(c, 2000, idle, []Pod{{Name: "mi300", Class: "amd-mi300x"}})
	if !d.Remote || d.Reason != ReasonStockSplit || d.Prefill != "mi300" {
		t.Fatalf("got %+v", d)
	}
}

func TestDecideModelUnprofiledCandidateKeepsLocal(t *testing.T) {
	c := labConfig()
	d := Decide(c, promptLong, gaudi(40), []Pod{{Name: "mi300", Class: "amd-mi300x"}})
	if d.Remote || d.Reason != ReasonNoPrefillCandidate {
		t.Fatalf("an unprofiled candidate cannot be priced, got %+v", d)
	}
}

func TestDecideMeasuredSpeedOverridesProfile(t *testing.T) {
	c := labConfig()
	slow := Pod{Name: "a30-slow", Class: classA30, PrefillTokensPerSec: 1000}
	fast := Pod{Name: "a30-fast", Class: classA30, PrefillTokensPerSec: 9000}
	d := Decide(c, promptLong, gaudi(40), []Pod{slow, fast})
	if !d.Remote || d.Prefill != "a30-fast" {
		t.Fatalf("want split to the faster measured pod, got %+v", d)
	}
	// A measured decode speed replaces a missing class profile.
	dec := Pod{Name: "x", Class: "unknown", Running: 40, PrefillTokensPerSec: 20000}
	if d := Decide(c, promptLong, dec, []Pod{a30(0)}); d.Reason == ReasonMissingProfile {
		t.Fatalf("measured decode speed should replace the missing profile: %+v", d)
	}
	// Zero means not measured: the class profile is used.
	if r, ok := c.podRate(Pod{Class: classA30}); !ok || r != 5000 {
		t.Fatalf("podRate fallback = %v, %v", r, ok)
	}
}
