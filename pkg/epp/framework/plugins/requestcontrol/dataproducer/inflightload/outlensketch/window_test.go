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

package outlensketch

import (
	"strings"
	"testing"
	"unicode/utf8"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

func msg(role, text string) fwkrh.Message {
	return fwkrh.Message{Role: role, Content: fwkrh.Content{Raw: text}}
}

func TestKeptTurnsSelection(t *testing.T) {
	kept := keptTurns([]fwkrh.Message{
		msg("system", "sys"),
		msg("user", "u1"),
		msg("assistant", "a1"),
		msg("tool", "t1"),
		msg("user", "u2"),
		msg("tool", "t2"),
	})
	// Expect system, first user, last user, last tool -- assistant dropped.
	want := []keptTurn{
		{shareSystem, "sys"},
		{shareFirstUser, "u1"},
		{shareLastUser, "u2"},
		{shareLastTool, "t2"},
	}
	if len(kept) != len(want) {
		t.Fatalf("kept %d turns, want %d: %+v", len(kept), len(want), kept)
	}
	for i := range want {
		if kept[i] != want[i] {
			t.Errorf("kept[%d] = %+v, want %+v", i, kept[i], want[i])
		}
	}
}

func TestKeptTurnsSingleUserDedup(t *testing.T) {
	kept := keptTurns([]fwkrh.Message{msg("user", "only")})
	if len(kept) != 1 || kept[0] != (keptTurn{shareFirstUser, "only"}) {
		t.Errorf("single user turn kept = %+v, want one first_user", kept)
	}
}

func TestKeptTurnsDropsAssistant(t *testing.T) {
	// A trailing assistant turn must not become the last user turn.
	kept := keptTurns([]fwkrh.Message{msg("user", "u1"), msg("assistant", "a1")})
	if len(kept) != 1 || kept[0] != (keptTurn{shareFirstUser, "u1"}) {
		t.Errorf("user+assistant kept = %+v, want only first_user u1 (assistant dropped)", kept)
	}
	// Assistants between users are dropped; first and last user are the user turns.
	kept = keptTurns([]fwkrh.Message{msg("user", "u1"), msg("assistant", "a1"), msg("user", "u2")})
	want := []keptTurn{{shareFirstUser, "u1"}, {shareLastUser, "u2"}}
	if len(kept) != len(want) || kept[0] != want[0] || kept[1] != want[1] {
		t.Errorf("user/assistant/user kept = %+v, want %+v", kept, want)
	}
}

func TestKeptTurnsUnlabeledIsUser(t *testing.T) {
	kept := keptTurns([]fwkrh.Message{msg("", "blob")})
	if len(kept) != 1 || kept[0].shareIdx != shareFirstUser {
		t.Errorf("unlabeled turn kept = %+v, want first_user", kept)
	}
}

func TestWindowStaysWithinBudget(t *testing.T) {
	long := strings.Repeat("a", 5000)
	out := applyWindow([]fwkrh.Message{msg("user", long)})
	if n := utf8.RuneCountInString(out); n > windowBudget {
		t.Errorf("window length %d exceeds budget %d", n, windowBudget)
	}
}

func TestHeadTailSkipsMiddle(t *testing.T) {
	text := "START" + strings.Repeat("x", 1000) + "END"
	out := headTail(text, 10, 0.5)
	if utf8.RuneCountInString(out) != 10 {
		t.Fatalf("headTail length = %d, want 10", utf8.RuneCountInString(out))
	}
	if !strings.HasPrefix(out, "START") || !strings.HasSuffix(out, "END") {
		t.Errorf("headTail = %q, want head START.. and tail ..END", out)
	}
}

func TestHeadTailShortTextUnchanged(t *testing.T) {
	if got := headTail("short", 100, 0.5); got != "short" {
		t.Errorf("headTail of short text = %q, want unchanged", got)
	}
}

func TestHeadTailHeadOnly(t *testing.T) {
	// headFrac >= 1.0 keeps only the head.
	if got := headTail("abcdefghij", 4, 1.0); got != "abcd" {
		t.Errorf("head-only headTail = %q, want %q", got, "abcd")
	}
}

func TestWindowRunesNotBytes(t *testing.T) {
	// Multi-byte runes must be counted as single characters, so a budget-length
	// slice never splits a rune.
	out := headTail(strings.Repeat("é", 100), 10, 0.5)
	if utf8.RuneCountInString(out) != 10 {
		t.Errorf("rune count = %d, want 10", utf8.RuneCountInString(out))
	}
	if !utf8.ValidString(out) {
		t.Error("headTail split a multi-byte rune")
	}
}

func TestWindowEmptyForNonChat(t *testing.T) {
	if got := windowedPrompt(&fwkrh.InferenceRequestBody{}); got != "" {
		t.Errorf("non-chat body window = %q, want empty", got)
	}
}
