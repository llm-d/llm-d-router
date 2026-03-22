package common

import (
	"encoding/json"
	"strconv"
	"strings"
)

const (
	// NoDataParallelRank is the sentinel value indicating a non-DP deployment.
	NoDataParallelRank = -1

	// DPRankSuffix is the separator used in scoring keys to encode DP rank info.
	// Scoring keys from the KV cache indexer use the format "<podIdentifier>@dp<rank>".
	DPRankSuffix = "@dp"

	// DataParallelRankHeader is the header name used to indicate the DP rank for a request.
	DataParallelRankHeader = "x-data-parallel-rank"

	// DPWinningRanksHeader is an internal header used to transport winning DP rank
	// information from the scorer to the PreRequest plugin. It carries a JSON-encoded
	// map of pod address → winning rank (e.g., {"10.0.0.1":0,"10.0.0.2":1}).
	// This header is removed by the dp-rank-header-handler PreRequest plugin before
	// the request is forwarded to the backend.
	DPWinningRanksHeader = "x-llm-d-dp-winning-ranks"
)

// DPMode represents the vLLM data parallel deployment mode.
// See https://docs.vllm.ai/en/stable/serving/data_parallel_deployment
type DPMode string

const (
	// DPModeInternalLB represents the Internal Load Balancing DP mode.
	// All DP rank engines run as core engine processes inside one vllm serve,
	// communicating with API server(s) via ZMQ. A single HTTP port is exposed.
	// The DPLBAsyncMPClient handles internal request distribution using
	// per-rank queue stats (waiting*4 + running).
	DPModeInternalLB DPMode = "internal-lb"

	// DPModeHybridLB represents the Hybrid Load Balancing DP mode.
	// Each node runs its own API server(s) that only route to local DP rank engines.
	// An external upstream LB (e.g., K8s Ingress) spreads traffic across per-node endpoints.
	// Each node's API server uses DPLBAsyncMPClient for local rank selection.
	DPModeHybridLB DPMode = "hybrid-lb"

	// DPModeExternalLB represents the External Load Balancing DP mode.
	// Each DP rank is a separate vllm serve process with its own HTTP port.
	// Typically each rank is a separate K8s pod. An external router (the scheduler)
	// balances requests across ranks directly.
	DPModeExternalLB DPMode = "external-lb"
)

// ParseDPScoringKey parses a DP-aware scoring key into its base pod identifier
// and data parallel rank. Scoring keys from the KV cache indexer use the
// format "<podIdentifier>@dp<rank>".
//
// Examples:
//
//	"10.0.0.1:8080"       -> ("10.0.0.1:8080", -1)
//	"10.0.0.1:8080@dp0"   -> ("10.0.0.1:8080", 0)
//	"10.0.0.1:8080@dp3"   -> ("10.0.0.1:8080", 3)
func ParseDPScoringKey(scoringKey string) (podIdentifier string, dpRank int) {
	idx := strings.LastIndex(scoringKey, DPRankSuffix)
	if idx < 0 {
		return scoringKey, NoDataParallelRank
	}

	rankStr := scoringKey[idx+len(DPRankSuffix):]
	rank, err := strconv.Atoi(rankStr)
	if err != nil {
		// Malformed suffix; treat the whole key as the pod identifier
		return scoringKey, NoDataParallelRank
	}

	return scoringKey[:idx], rank
}

// StripDPRankSuffix removes the "@dp<N>" suffix from a scoring key if present,
// returning just the base pod identifier.
func StripDPRankSuffix(scoringKey string) string {
	podID, _ := ParseDPScoringKey(scoringKey)
	return podID
}

// BuildDPScoringKey constructs a DP-aware scoring key from a pod identifier and rank.
// If rank is NoDataParallelRank, the pod identifier is returned as-is.
func BuildDPScoringKey(podIdentifier string, dpRank int) string {
	if dpRank == NoDataParallelRank {
		return podIdentifier
	}
	return podIdentifier + DPRankSuffix + strconv.Itoa(dpRank)
}

// EncodeWinningRanks serializes a winning ranks map to a JSON string for transport via HTTP headers.
func EncodeWinningRanks(ranks map[string]int) (string, error) {
	data, err := json.Marshal(ranks)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// DecodeWinningRanks deserializes a JSON string from an HTTP header back into a winning ranks map.
func DecodeWinningRanks(encoded string) (map[string]int, error) {
	var ranks map[string]int
	if err := json.Unmarshal([]byte(encoded), &ranks); err != nil {
		return nil, err
	}
	return ranks, nil
}
