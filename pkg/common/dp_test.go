package common

import (
	"testing"
)

func TestParseDPScoringKey(t *testing.T) {
	tests := []struct {
		name       string
		scoringKey string
		wantPodID  string
		wantDPRank int
	}{
		{
			name:       "no DP suffix",
			scoringKey: "10.0.0.1:8080",
			wantPodID:  "10.0.0.1:8080",
			wantDPRank: NoDataParallelRank,
		},
		{
			name:       "DP rank 0",
			scoringKey: "10.0.0.1:8080@dp0",
			wantPodID:  "10.0.0.1:8080",
			wantDPRank: 0,
		},
		{
			name:       "DP rank 3",
			scoringKey: "10.0.0.1:8080@dp3",
			wantPodID:  "10.0.0.1:8080",
			wantDPRank: 3,
		},
		{
			name:       "malformed rank suffix",
			scoringKey: "10.0.0.1:8080@dpabc",
			wantPodID:  "10.0.0.1:8080@dpabc",
			wantDPRank: NoDataParallelRank,
		},
		{
			name:       "empty string",
			scoringKey: "",
			wantPodID:  "",
			wantDPRank: NoDataParallelRank,
		},
		{
			name:       "pod identifier without port",
			scoringKey: "pod-1@dp2",
			wantPodID:  "pod-1",
			wantDPRank: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotPodID, gotDPRank := ParseDPScoringKey(tt.scoringKey)
			if gotPodID != tt.wantPodID {
				t.Errorf("ParseDPScoringKey(%q) podID = %q, want %q", tt.scoringKey, gotPodID, tt.wantPodID)
			}
			if gotDPRank != tt.wantDPRank {
				t.Errorf("ParseDPScoringKey(%q) dpRank = %d, want %d", tt.scoringKey, gotDPRank, tt.wantDPRank)
			}
		})
	}
}

func TestStripDPRankSuffix(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"10.0.0.1:8080@dp0", "10.0.0.1:8080"},
		{"10.0.0.1:8080", "10.0.0.1:8080"},
		{"pod-1@dp5", "pod-1"},
	}

	for _, tt := range tests {
		got := StripDPRankSuffix(tt.input)
		if got != tt.want {
			t.Errorf("StripDPRankSuffix(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestBuildDPScoringKey(t *testing.T) {
	tests := []struct {
		podID string
		rank  int
		want  string
	}{
		{"10.0.0.1:8080", NoDataParallelRank, "10.0.0.1:8080"},
		{"10.0.0.1:8080", 0, "10.0.0.1:8080@dp0"},
		{"10.0.0.1:8080", 3, "10.0.0.1:8080@dp3"},
	}

	for _, tt := range tests {
		got := BuildDPScoringKey(tt.podID, tt.rank)
		if got != tt.want {
			t.Errorf("BuildDPScoringKey(%q, %d) = %q, want %q", tt.podID, tt.rank, got, tt.want)
		}
	}
}
