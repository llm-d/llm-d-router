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

// Package hetero holds the P/D split decision for pools of different
// accelerator classes. It depends only on the standard library so the
// decision can be tested and simulated outside the EPP.
package hetero

import (
	"errors"
	"fmt"
	"math"
)

// Reason values explain a Decision. They are used as a metric label.
const (
	ReasonRemoteBetter       = "remote-better"
	ReasonLocalBetter        = "local-better"
	ReasonShortPrompt        = "short-prompt"
	ReasonDecodeKVFull       = "decode-kv-full"
	ReasonDecodeDisabled     = "decode-disabled"
	ReasonNoPrefillCandidate = "no-prefill-candidate"
	ReasonSLOMiss            = "remote-misses-slo"
	ReasonMissingProfile     = "missing-profile"
	// ReasonStockSplit and ReasonStockKeep are the stock threshold's own outcomes under
	// the guarded-stock strategy; ReasonIdleVeto is the guard overruling a stock split.
	ReasonStockSplit = "stock-split"
	ReasonStockKeep  = "stock-keep"
	ReasonIdleVeto   = "idle-veto"
	// ReasonOverrideKeep and ReasonOverrideMove are the auto strategy overruling the
	// stock rule because the other path is cheaper by at least OverrideMargin.
	ReasonOverrideKeep = "override-keep"
	ReasonOverrideMove = "override-move"
)

// Strategy values select how Decide reaches a decision.
const (
	// StrategyModel compares the remote and local estimates for every request.
	StrategyModel = "model"
	// StrategyGuardedStock applies the stock rule (split when the uncached tokens reach
	// StockNonCachedTokens) and only overrules a split the estimates show cannot help:
	// the decode pod is idle and local prefill is no slower, its KV cache is full, it is
	// disabled, or no prefill pod can take the request. It never adds a split stock would not make.
	StrategyGuardedStock = "guarded-stock"
	// StrategyAuto (default) starts from the stock rule's answer and overrules it, in either
	// direction, only when the estimates show the other path is cheaper by at least
	// OverrideMargin and, for a move, meets the TTFT SLO. Without profiles for the decode
	// pod and a prefill candidate, the stock answer stands.
	StrategyAuto = "auto"
)

// Profile is the measured prefill throughput of one accelerator class
// for the served model.
type Profile struct {
	PrefillTokensPerSec float64 `json:"prefillTokensPerSec"`
}

// Config holds the decision parameters.
type Config struct {
	// KVBytesPerToken is the KV-cache size of one token across all layers.
	KVBytesPerToken int64 `json:"kvBytesPerToken"`
	// TTFTSLOMs rejects the remote path when its estimate exceeds it. Zero disables the check.
	TTFTSLOMs float64 `json:"ttftSLOMs"`
	// StallWeight is the cost of one second of decode stall, summed over the
	// streams running on the decode pod, relative to one second of the new
	// request's time to first token.
	StallWeight float64 `json:"stallWeight"`
	// SafetyMargin is the fraction by which remote must beat local, in [0, 1).
	SafetyMargin float64 `json:"safetyMargin"`
	// MaxDecodeKVUsage keeps requests local when decode KV usage (0 to 1) is at or above it.
	MaxDecodeKVUsage float64 `json:"maxDecodeKVUsage"`
	// MinUncachedTokens keeps requests with fewer uncached tokens local without estimation.
	MinUncachedTokens int `json:"minUncachedTokens"`
	// BandwidthGbps is the prefill-to-decode KV transfer bandwidth across classes.
	BandwidthGbps float64 `json:"bandwidthGbps"`
	// SameClassBandwidthGbps applies when both pods share a class. Zero means BandwidthGbps.
	SameClassBandwidthGbps float64 `json:"sameClassBandwidthGbps"`
	// AvgQueuedPromptTokens estimates the prefill work behind each waiting request.
	AvgQueuedPromptTokens int `json:"avgQueuedPromptTokens"`
	// HandoffOverheadMs is the fixed cost of the remote path beyond the KV transfer itself:
	// the decode sidecar's request to the prefill pod, its scheduling and response, and the
	// decode pod fetching the KV. Lab measurements put it near 300 ms. Zero adds nothing.
	HandoffOverheadMs float64 `json:"handoffOverheadMs"`
	// StallFraction is the share of a local prefill's time that the decode pod's running
	// streams actually lose, in [0, 1]. 1 models a full freeze. With chunked prefill the
	// streams keep decoding between chunks and lose only a small part of it.
	StallFraction float64 `json:"stallFraction"`
	// Strategy is StrategyAuto (default; empty means auto), StrategyModel or StrategyGuardedStock.
	Strategy string `json:"strategy"`
	// StockNonCachedTokens is the stock threshold the guarded-stock strategy applies.
	StockNonCachedTokens int `json:"stockNonCachedTokens"`
	// IdleRunningRequests is the most running streams at which the guarded-stock strategy
	// treats the decode pod as idle: with nothing to protect, a split that is not faster
	// than local prefill is overruled.
	IdleRunningRequests int `json:"idleRunningRequests"`
	// OverrideMargin is how much cheaper, as a fraction in [0, 1), the other path must be
	// before the auto strategy overrules the stock rule.
	OverrideMargin float64 `json:"overrideMargin"`
	// Profiles maps accelerator class to its prefill throughput.
	Profiles map[string]Profile `json:"profiles"`
}

// DefaultConfig returns the parameter defaults. Profiles are left empty and must be measured.
func DefaultConfig() Config {
	return Config{
		KVBytesPerToken:       131072,
		TTFTSLOMs:             0,
		StallWeight:           0.25,
		SafetyMargin:          0.1,
		MaxDecodeKVUsage:      0.9,
		MinUncachedTokens:     256,
		BandwidthGbps:         10,
		AvgQueuedPromptTokens: 1024,
		HandoffOverheadMs:     0,
		StallFraction:         1,
		Strategy:              StrategyAuto,
		OverrideMargin:        0.25,
		StockNonCachedTokens:  1024,
		IdleRunningRequests:   2,
		Profiles:              map[string]Profile{},
	}
}

// Validate reports the first invalid parameter.
func (c Config) Validate() error {
	switch {
	case c.KVBytesPerToken <= 0:
		return errors.New("kvBytesPerToken must be positive")
	case c.TTFTSLOMs < 0:
		return errors.New("ttftSLOMs cannot be negative")
	case c.StallWeight < 0:
		return errors.New("stallWeight cannot be negative")
	case c.SafetyMargin < 0 || c.SafetyMargin >= 1:
		return errors.New("safetyMargin must be in [0, 1)")
	case c.MaxDecodeKVUsage <= 0 || c.MaxDecodeKVUsage > 1:
		return errors.New("maxDecodeKVUsage must be in (0, 1]")
	case c.MinUncachedTokens < 0:
		return errors.New("minUncachedTokens cannot be negative")
	case c.BandwidthGbps <= 0:
		return errors.New("bandwidthGbps must be positive")
	case c.SameClassBandwidthGbps < 0:
		return errors.New("sameClassBandwidthGbps cannot be negative")
	case c.AvgQueuedPromptTokens < 0:
		return errors.New("avgQueuedPromptTokens cannot be negative")
	case c.HandoffOverheadMs < 0:
		return errors.New("handoffOverheadMs cannot be negative")
	case c.StallFraction < 0 || c.StallFraction > 1:
		return errors.New("stallFraction must be in [0, 1]")
	case c.Strategy != "" && c.Strategy != StrategyModel && c.Strategy != StrategyGuardedStock && c.Strategy != StrategyAuto:
		return fmt.Errorf("strategy must be %q, %q or %q", StrategyModel, StrategyGuardedStock, StrategyAuto)
	case c.StockNonCachedTokens <= 0:
		return errors.New("stockNonCachedTokens must be positive")
	case c.IdleRunningRequests < 0:
		return errors.New("idleRunningRequests cannot be negative")
	case c.OverrideMargin < 0 || c.OverrideMargin >= 1:
		return errors.New("overrideMargin must be in [0, 1)")
	}
	for class, p := range c.Profiles {
		if p.PrefillTokensPerSec <= 0 {
			return fmt.Errorf("profile %q: prefillTokensPerSec must be positive", class)
		}
	}
	return nil
}

// Pod is the state of one candidate pod at decision time.
type Pod struct {
	Name  string
	Class string
	// PrefillTokensPerSec is this pod's own measured prefill speed. Zero falls back to the
	// class profile.
	PrefillTokensPerSec float64
	Running             int
	Waiting             int
	KVUsage             float64
	Disabled            bool
}

// Decision is the outcome for one request. Seconds fields are zero when not estimated.
type Decision struct {
	Remote bool
	// Strategy is the strategy that produced the decision.
	Strategy string
	// Fallback is set when the inputs cannot be estimated; the caller applies its default rule.
	Fallback bool
	Reason   string
	// Prefill names the prefill pod the estimate used when Remote is set.
	Prefill       string
	RemoteSeconds float64
	// HandoffSeconds is the fixed handoff overhead included in RemoteSeconds.
	HandoffSeconds  float64
	LocalSeconds    float64
	StallSeconds    float64
	TransferSeconds float64
}

func (c Config) rate(class string) (float64, bool) {
	p, ok := c.Profiles[class]
	return p.PrefillTokensPerSec, ok && p.PrefillTokensPerSec > 0
}

// podRate is the pod's measured prefill speed when it has one, else its class profile.
func (c Config) podRate(p Pod) (float64, bool) {
	if p.PrefillTokensPerSec > 0 {
		return p.PrefillTokensPerSec, true
	}
	return c.rate(p.Class)
}

// PodRate is the prefill throughput of p: its measured speed, else its class profile.
func (c Config) PodRate(p Pod) (float64, bool) {
	return c.podRate(p)
}

// QueueSeconds estimates how long the prefill work waiting on p delays a new request.
func (c Config) QueueSeconds(p Pod, rate float64) float64 {
	return c.queueSeconds(p, rate)
}

func (c Config) bandwidthBytesPerSec(from, to string) float64 {
	gbps := c.BandwidthGbps
	if from == to && from != "" && c.SameClassBandwidthGbps > 0 {
		gbps = c.SameClassBandwidthGbps
	}
	return gbps * 1e9 / 8
}

func (c Config) queueSeconds(p Pod, rate float64) float64 {
	return float64(p.Waiting*c.AvgQueuedPromptTokens) / rate
}

// eligible reports whether p can prefill for decode: another pod that is not disabled.
func eligible(p, decode Pod) bool {
	return p.Name != decode.Name && !p.Disabled
}

// firstEligible returns the first eligible prefill pod, or an empty name when there is none.
func firstEligible(decode Pod, prefill []Pod) string {
	for _, p := range prefill {
		if eligible(p, decode) {
			return p.Name
		}
	}
	return ""
}

// Decide chooses between prefilling uncached tokens on the decode pod and
// prefilling them on the best prefill candidate followed by a KV transfer,
// using the configured strategy.
func Decide(c Config, uncachedTokens int, decode Pod, prefill []Pod) Decision {
	var d Decision
	switch c.Strategy {
	case StrategyModel:
		d = decideModel(c, uncachedTokens, decode, prefill)
	case StrategyGuardedStock:
		d = decideGuardedStock(c, uncachedTokens, decode, prefill)
	default:
		d = decideAuto(c, uncachedTokens, decode, prefill)
		d.Strategy = StrategyAuto
		return d
	}
	d.Strategy = c.Strategy
	return d
}

// decideAuto starts from the stock rule and overrules it only when the other path is
// clearly cheaper:
//
//	keep = U/R_D + w * f * N_D * U/R_D
//	move = U/R_P + Q_P + U*b/BW + H
//	stock move -> keep when keep <= (1 - overrideMargin) * move
//	stock keep -> move when move <= (1 - overrideMargin) * keep and move <= SLO
func decideAuto(c Config, uncachedTokens int, decode Pod, prefill []Pod) Decision {
	if uncachedTokens <= 0 {
		return Decision{Reason: ReasonShortPrompt}
	}
	if decode.Disabled {
		return Decision{Reason: ReasonDecodeDisabled}
	}
	if decode.KVUsage >= c.MaxDecodeKVUsage {
		return Decision{Reason: ReasonDecodeKVFull}
	}
	first := firstEligible(decode, prefill)
	if first == "" {
		return Decision{Reason: ReasonNoPrefillCandidate}
	}
	stockMove := uncachedTokens >= c.StockNonCachedTokens
	d := Decision{Reason: ReasonStockKeep}
	if stockMove {
		d = Decision{Remote: true, Reason: ReasonStockSplit, Prefill: first}
	}
	if !stockMove && uncachedTokens < c.MinUncachedTokens {
		d.Reason = ReasonShortPrompt
		return d
	}
	rateD, ok := c.podRate(decode)
	if !ok {
		return d
	}
	u := float64(uncachedTokens)
	best, saw := bestRemote(c, u, decode, prefill)
	if !saw {
		return d
	}
	d.RemoteSeconds, d.TransferSeconds, d.HandoffSeconds = best.RemoteSeconds, best.TransferSeconds, best.HandoffSeconds
	// The decode pod's waiting queue delays both paths alike: a remote-prefilled request still
	// waits for a decode slot. It is left out so it cannot tip the comparison.
	d.LocalSeconds = u / rateD
	d.StallSeconds = c.StallFraction * float64(decode.Running) * u / rateD
	keep := d.LocalSeconds + c.StallWeight*d.StallSeconds
	move := d.RemoteSeconds
	sure := 1 - c.OverrideMargin
	if stockMove {
		d.Prefill = best.Prefill
		if keep <= sure*move {
			d.Remote, d.Prefill, d.Reason = false, "", ReasonOverrideKeep
		}
		return d
	}
	if move <= sure*keep && (c.TTFTSLOMs <= 0 || move*1000 <= c.TTFTSLOMs) {
		d.Remote, d.Prefill, d.Reason = true, best.Prefill, ReasonOverrideMove
	}
	return d
}

// decideGuardedStock applies the stock threshold and vetoes splits that cannot help.
func decideGuardedStock(c Config, uncachedTokens int, decode Pod, prefill []Pod) Decision {
	if uncachedTokens < c.StockNonCachedTokens || uncachedTokens <= 0 {
		return Decision{Reason: ReasonStockKeep}
	}
	if decode.Disabled {
		return Decision{Reason: ReasonDecodeDisabled}
	}
	if decode.KVUsage >= c.MaxDecodeKVUsage {
		return Decision{Reason: ReasonDecodeKVFull}
	}
	first := firstEligible(decode, prefill)
	if first == "" {
		return Decision{Reason: ReasonNoPrefillCandidate}
	}
	d := Decision{Remote: true, Reason: ReasonStockSplit, Prefill: first}
	rateD, ok := c.podRate(decode)
	if !ok {
		return d
	}
	u := float64(uncachedTokens)
	best, saw := bestRemote(c, u, decode, prefill)
	if !saw {
		return d
	}
	d.Prefill = best.Prefill
	d.RemoteSeconds, d.TransferSeconds, d.HandoffSeconds = best.RemoteSeconds, best.TransferSeconds, best.HandoffSeconds
	d.LocalSeconds = u/rateD + c.queueSeconds(decode, rateD)
	d.StallSeconds = c.StallFraction * float64(decode.Running) * u / rateD
	if decode.Running <= c.IdleRunningRequests && decode.Waiting == 0 && d.RemoteSeconds >= d.LocalSeconds {
		d.Remote = false
		d.Reason = ReasonIdleVeto
	}
	return d
}

// bestRemote returns the fastest profiled prefill candidate's remote estimate.
func bestRemote(c Config, u float64, decode Pod, prefill []Pod) (Decision, bool) {
	handoff := c.HandoffOverheadMs / 1000
	best := Decision{RemoteSeconds: math.Inf(1)}
	saw := false
	for _, p := range prefill {
		if !eligible(p, decode) {
			continue
		}
		rateP, ok := c.podRate(p)
		if !ok {
			continue
		}
		saw = true
		transfer := u * float64(c.KVBytesPerToken) / c.bandwidthBytesPerSec(p.Class, decode.Class)
		remote := u/rateP + c.queueSeconds(p, rateP) + transfer + handoff
		if remote < best.RemoteSeconds {
			best = Decision{Prefill: p.Name, RemoteSeconds: remote, TransferSeconds: transfer, HandoffSeconds: handoff}
		}
	}
	return best, saw
}

// decideModel compares the estimates:
//
//	remote = U/R_P + Q_P + U*b/BW + H
//	local  = U/R_D + Q_D
//	stall  = f * N_D * U/R_D      (f = StallFraction, H = HandoffOverheadMs)
//	split when remote <= SLO and remote < (1 - margin) * (local + w * stall)
func decideModel(c Config, uncachedTokens int, decode Pod, prefill []Pod) Decision {
	if uncachedTokens < c.MinUncachedTokens || uncachedTokens <= 0 {
		return Decision{Reason: ReasonShortPrompt}
	}
	if decode.Disabled {
		return Decision{Reason: ReasonDecodeDisabled}
	}
	if decode.KVUsage >= c.MaxDecodeKVUsage {
		return Decision{Reason: ReasonDecodeKVFull}
	}
	first := firstEligible(decode, prefill)
	if first == "" {
		return Decision{Reason: ReasonNoPrefillCandidate}
	}
	rateD, ok := c.podRate(decode)
	if !ok {
		// Prefill names a compatible pod so the caller's fallback rule never splits
		// toward a pod that cannot take the KV.
		return Decision{Fallback: true, Reason: ReasonMissingProfile, Prefill: first}
	}

	u := float64(uncachedTokens)
	local := u/rateD + c.queueSeconds(decode, rateD)
	stall := c.StallFraction * float64(decode.Running) * u / rateD

	best, sawCandidate := bestRemote(c, u, decode, prefill)
	best.LocalSeconds = local
	best.StallSeconds = stall
	if !sawCandidate {
		best.RemoteSeconds = 0
		best.Reason = ReasonNoPrefillCandidate
		best.Prefill = ""
		return best
	}
	if c.TTFTSLOMs > 0 && best.RemoteSeconds*1000 > c.TTFTSLOMs {
		best.Reason = ReasonSLOMiss
		return best
	}
	if best.RemoteSeconds < (1-c.SafetyMargin)*(local+c.StallWeight*stall) {
		best.Remote = true
		best.Reason = ReasonRemoteBetter
		return best
	}
	best.Reason = ReasonLocalBetter
	return best
}
