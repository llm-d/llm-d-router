package kvcache

import (
	"fmt"
)

// PodScore couples a pod identifier (IP) with a score.
type PodScore struct {
	Name  string
	Score float64
}

// KVScoringStrategy defines the strategy used to score pods for KV cache block reuse.
type KVScoringStrategy string

const (
	// LongestPrefixMatch Score by longest consecutive match from start.
	LongestPrefixMatch KVScoringStrategy = "LongestPrefix"
	// HighestBlockHit Score by highest block index hit.
	HighestBlockHit KVScoringStrategy = "HighestBlockHit"
	// CoverageBasedMatching Score by total number of blocks hit.
	CoverageBasedMatching KVScoringStrategy = "CoverageBased"
)

// KVBlockScorerConfig holds the configuration for the KVBlockScorer.
type KVBlockScorerConfig struct {
	ScoringStrategy KVScoringStrategy
}

// DefaultKVBlockScorerConfig returns the default configuration for the KVBlockScorer.
func DefaultKVBlockScorerConfig() *KVBlockScorerConfig {
	return &KVBlockScorerConfig{
		ScoringStrategy: LongestPrefixMatch,
	}
}

// KVBlockScorer defines the interface for implementing a KV block scoring
// strategy.
type KVBlockScorer interface {
	// Strategy returns the scoring strategy type.
	Strategy() KVScoringStrategy
	// Score scores the blocks based on the scoring strategy.
	// TODO: make keyToPods after fixing LMCache.
	Score(blockKeys []string, keyToPod map[string]string) ([]PodScore, error)
}

// NewKVBlockScorer creates a new KVBlockScorer based on the provided strategy.
func NewKVBlockScorer(config *KVBlockScorerConfig) (KVBlockScorer, error) {
	switch config.ScoringStrategy {
	case LongestPrefixMatch:
		return &LongestPrefixScorer{}, nil
	case HighestBlockHit:
		return &HighestBlockHitScorer{}, nil
	case CoverageBasedMatching:
		return &CoverageBasedScorer{}, nil
	default:
		return nil, fmt.Errorf("unsupported scoring strategy: %s", config.ScoringStrategy)
	}
}

// LongestPrefixScorer scores based on longest consecutive block matches count
// starting from block 0.
type LongestPrefixScorer struct{}

// Strategy returns the strategy type: LongestPrefixMatch.
func (s *LongestPrefixScorer) Strategy() KVScoringStrategy {
	return LongestPrefixMatch
}

// Score implements the longest prefix scoring logic.
func (s *LongestPrefixScorer) Score(blockKeys []string, keyToPod map[string]string) ([]PodScore, error) {
	longestIndex := make(map[string]int)

	if len(blockKeys) == 0 {
		return nil, nil
	}

	prevPod, ok := keyToPod[blockKeys[0]]
	if !ok {
		return nil, nil
	}
	count := 1

	for i := 1; i < len(blockKeys); i++ {
		pod, ok := keyToPod[blockKeys[i]]
		if !ok {
			break
		}

		if pod == prevPod {
			count++
		} else {
			if count > longestIndex[prevPod] {
				longestIndex[prevPod] = count
			}
			prevPod = pod
			count = 1
		}
	}

	if count > longestIndex[prevPod] {
		longestIndex[prevPod] = count
	}

	return convertScoresToPods(longestIndex), nil
}

// HighestBlockHitScorer scores based on the highest-indexed block hit for each
// pod.
type HighestBlockHitScorer struct{}

// Strategy returns the strategy type: HighestBlockHit.
func (s *HighestBlockHitScorer) Strategy() KVScoringStrategy {
	return HighestBlockHit
}

// Score implements the highest block hit scoring logic.
func (s *HighestBlockHitScorer) Score(blockKeys []string, keyToPod map[string]string) ([]PodScore, error) {
	maxIndex := make(map[string]int)

	for idx, k := range blockKeys {
		pod, ok := keyToPod[k]
		if !ok {
			continue
		}
		maxIndex[pod] = idx
	}

	return convertScoresToPods(maxIndex), nil
}

// CoverageBasedScorer scores based on total number of blocks hit (coverage).
type CoverageBasedScorer struct{}

// Strategy returns the strategy type: CoverageBasedMatching.
func (s *CoverageBasedScorer) Strategy() KVScoringStrategy {
	return CoverageBasedMatching
}

// Score implements the coverage-based scoring logic.
func (s *CoverageBasedScorer) Score(blockKeys []string, hitmap map[string]string) ([]PodScore, error) {
	coverage := make(map[string]int)

	for _, k := range blockKeys {
		pod, ok := hitmap[k]
		if !ok {
			continue
		}
		coverage[pod]++
	}

	return convertScoresToPods(coverage), nil
}

// convertScoresToPods converts a map of pod name to score into a slice of Pod structs.
func convertScoresToPods(scoreMap map[string]int) []PodScore {
	scored := make([]PodScore, 0, len(scoreMap))
	for pod, score := range scoreMap {
		scored = append(scored, PodScore{
			Name:  pod,
			Score: float64(score),
		})
	}

	return scored
}
