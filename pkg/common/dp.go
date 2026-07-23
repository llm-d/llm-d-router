package common

import (
	"encoding/json"
	"errors"
	"fmt"
	"net"
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
	// map of pod address → winning rank, where the pod address is the canonical
	// "ip:port" form produced by PodAddress (e.g., {"10.0.0.1:8000":0,"10.0.0.2:8000":1}).
	// This header is removed by the dp-rank-header-handler PreRequest plugin before
	// the request is forwarded to the backend.
	DPWinningRanksHeader = "x-llm-d-dp-winning-ranks"
)

// ParseDPScoringKey parses a DP-aware scoring key into its base pod identifier
// and data parallel rank. Scoring keys from the KV cache indexer use the
// format "<podIdentifier>@dp<rank>" where <rank> is a non-negative decimal
// integer. Any suffix that fails to match this strict shape (non-digit chars,
// leading sign, negative number, empty digits) is treated as part of the pod
// identifier, so pod names that happen to contain "@dp" (e.g. "@dp-service")
// are not mis-parsed.
//
// Examples:
//
//	"10.0.0.1:8080"              -> ("10.0.0.1:8080", -1)
//	"10.0.0.1:8080@dp0"          -> ("10.0.0.1:8080", 0)
//	"10.0.0.1:8080@dp3"          -> ("10.0.0.1:8080", 3)
//	"pod@dp-service:8080"        -> ("pod@dp-service:8080", -1)
//	"pod@dp-3"                   -> ("pod@dp-3", -1)
//	"pod@dp"                     -> ("pod@dp", -1)
func ParseDPScoringKey(scoringKey string) (podIdentifier string, dpRank int) {
	idx := strings.LastIndex(scoringKey, DPRankSuffix)
	if idx < 0 {
		return scoringKey, NoDataParallelRank
	}

	rankStr := scoringKey[idx+len(DPRankSuffix):]
	if !isAllDigits(rankStr) {
		return scoringKey, NoDataParallelRank
	}

	rank, err := strconv.Atoi(rankStr)
	if err != nil || rank < 0 {
		return scoringKey, NoDataParallelRank
	}

	return scoringKey[:idx], rank
}

// isAllDigits reports whether s is non-empty and composed entirely of ASCII
// decimal digits.
func isAllDigits(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	return true
}

// StripDPRankSuffix removes the "@dp<N>" suffix from a scoring key if present,
// returning just the base pod identifier.
func StripDPRankSuffix(scoringKey string) string {
	podID, _ := ParseDPScoringKey(scoringKey)
	return podID
}

// BuildDPScoringKey constructs a DP-aware scoring key from a pod identifier and rank.
// Negative ranks other than NoDataParallelRank are rejected: valid DP ranks are
// always non-negative, and silently encoding them would produce keys that
// ParseDPScoringKey would then reject, giving the caller misleading round-trips.
func BuildDPScoringKey(podIdentifier string, dpRank int) (string, error) {
	if dpRank == NoDataParallelRank {
		return podIdentifier, nil
	}
	if dpRank < 0 {
		return "", fmt.Errorf("invalid negative DP rank %d", dpRank)
	}
	return podIdentifier + DPRankSuffix + strconv.Itoa(dpRank), nil
}

// PodAddress returns the canonical "<ip>:<port>" key used to address a pod
// across the scorer and pre-request plugins. Keeping this in one place prevents
// format drift between the scorer (which uses this as a scoring key) and the
// PreRequest handler (which uses it to look up the winning rank for the
// selected pod). It uses net.JoinHostPort so IPv6 literals are bracketed
// correctly (e.g. "[2001:db8::1]:8000"), matching the format the KV cache
// indexer emits and preventing scorer/lookup mismatches.
func PodAddress(ipAddress, port string) string {
	return net.JoinHostPort(ipAddress, port)
}

// ErrEmptyWinningRanks is returned when Encode/DecodeWinningRanks is asked to
// handle an empty or nil ranks map. Callers should skip emitting the header
// in that case rather than transport an empty JSON object.
var ErrEmptyWinningRanks = errors.New("winning ranks map is empty")

// EncodeWinningRanks serializes a winning ranks map to a JSON string for
// transport via HTTP headers. Returns ErrEmptyWinningRanks if ranks is empty.
// All ranks must be non-negative; a negative rank is rejected.
func EncodeWinningRanks(ranks map[string]int) (string, error) {
	if len(ranks) == 0 {
		return "", ErrEmptyWinningRanks
	}
	for pod, rank := range ranks {
		if rank < 0 {
			return "", fmt.Errorf("invalid negative DP rank %d for pod %q", rank, pod)
		}
	}
	data, err := json.Marshal(ranks)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// DecodeWinningRanks deserializes a JSON string from an HTTP header back into
// a winning ranks map. Returns ErrEmptyWinningRanks if the decoded map is empty
// or the encoded input is empty, and rejects any negative rank.
func DecodeWinningRanks(encoded string) (map[string]int, error) {
	if encoded == "" {
		return nil, ErrEmptyWinningRanks
	}
	var ranks map[string]int
	if err := json.Unmarshal([]byte(encoded), &ranks); err != nil {
		return nil, err
	}
	if len(ranks) == 0 {
		return nil, ErrEmptyWinningRanks
	}
	for pod, rank := range ranks {
		if rank < 0 {
			return nil, fmt.Errorf("invalid negative DP rank %d for pod %q", rank, pod)
		}
	}
	return ranks, nil
}
