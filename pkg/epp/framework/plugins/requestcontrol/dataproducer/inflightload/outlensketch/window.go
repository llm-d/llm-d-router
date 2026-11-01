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
	"math"
	"strings"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

// W1 windowing (the validated W1-1024 recipe): keep the structure-bearing turns --
// system, first user, last user, and the most recent tool result -- drop bulky tool
// bodies and intermediate turns, and allocate a bounded character budget across the
// kept turns, taking head+tail of any that overflow. The length-predictive
// instruction cues cluster at these turns, and the bound keeps the hot-path
// featurization cost independent of prompt size.
//
// Chat requests carry structured messages, so the kept turns come straight from the
// message roles -- no marker re-segmentation. The budget is measured in characters
// (Unicode code points), matching the offline study.
const (
	windowBudget   = 1024 // total character budget across kept turns
	windowHeadFrac = 0.5  // fraction of an overflowing turn's slice taken from its head
	windowSep      = "\n\n"
)

// windowShares are the relative budget weights for {system, first user, last user,
// last tool}. The last-tool weight is zero: the turn is kept for structure but the
// budget goes to the user turns that carry the request.
var windowShares = [4]float64{0.15, 0.55, 0.30, 0.0}

const (
	shareSystem = iota
	shareFirstUser
	shareLastUser
	shareLastTool
)

// keptTurn is one turn retained by the window, tagged with its share index.
type keptTurn struct {
	shareIdx int
	text     string
}

// windowedPrompt returns the W1 window of a chat request's messages. A request
// with no chat messages yields the empty string (the caller abstains).
func windowedPrompt(body *fwkrh.InferenceRequestBody) string {
	if body == nil || body.ChatCompletions == nil {
		return ""
	}
	return applyWindow(body.ChatCompletions.Messages)
}

func applyWindow(messages []fwkrh.Message) string {
	kept := keptTurns(messages)
	if len(kept) == 0 {
		var all strings.Builder
		for _, m := range messages {
			all.WriteString(m.Content.PlainText())
			all.WriteByte(' ')
		}
		return headTail(strings.TrimSpace(all.String()), windowBudget, windowHeadFrac)
	}

	// Reserve the separators so the joined output stays within budget; keep at
	// least one character per kept turn.
	sepCost := len([]rune(windowSep)) * (len(kept) - 1)
	bodyBudget := windowBudget - sepCost
	if bodyBudget < len(kept) {
		bodyBudget = len(kept)
	}

	weights := make([]float64, len(kept))
	var wsum float64
	for i, k := range kept {
		w := windowShares[k.shareIdx]
		if w < 0 {
			w = 0
		}
		weights[i] = w
		wsum += w
	}
	if wsum <= 0 { // every kept turn had zero share -> even split
		for i := range weights {
			weights[i] = 1
		}
		wsum = float64(len(kept))
	}

	// A turn shorter than its slice donates the slack to later turns.
	pieces := make([]string, 0, len(kept))
	slack := 0
	for i, k := range kept {
		slice := int(float64(bodyBudget) * weights[i] / wsum)
		if slice < 1 {
			slice = 1
		}
		allot := slice + slack
		piece := headTail(k.text, allot, windowHeadFrac)
		slack = allot - len([]rune(piece))
		if piece != "" {
			pieces = append(pieces, piece)
		}
	}
	return strings.Join(pieces, windowSep)
}

// keptTurns selects {system, first user, last user, last tool} from the messages.
// Messages with no role count as user. A single user turn is kept once.
func keptTurns(messages []fwkrh.Message) []keptTurn {
	var system string
	haveSystem := false
	var users []string
	var lastTool string
	haveTool := false
	for _, m := range messages {
		text := strings.TrimSpace(m.Content.PlainText())
		switch m.Role {
		case "system":
			if !haveSystem { // first system turn
				system, haveSystem = text, true
			}
		case "user", "": // an unlabeled turn counts as user, matching the study
			users = append(users, text)
		case "tool":
			lastTool, haveTool = text, true
		default: // assistant and any other role carry no request cue; dropped
		}
	}

	var kept []keptTurn
	if haveSystem {
		kept = append(kept, keptTurn{shareSystem, system})
	}
	if len(users) > 0 {
		kept = append(kept, keptTurn{shareFirstUser, users[0]})
		if len(users) > 1 {
			kept = append(kept, keptTurn{shareLastUser, users[len(users)-1]})
		}
	}
	if haveTool {
		kept = append(kept, keptTurn{shareLastTool, lastTool})
	}
	return kept
}

// headTail keeps at most n characters: round(n*headFrac) from the front and the
// rest from the back, skipping the middle. Round-half-to-even matches the offline
// study's integer rounding.
func headTail(text string, n int, headFrac float64) string {
	if n <= 0 {
		return ""
	}
	r := []rune(text)
	if len(r) <= n {
		return text
	}
	if headFrac >= 1.0 {
		return string(r[:n])
	}
	h := int(math.RoundToEven(float64(n) * headFrac))
	t := n - h
	if t <= 0 {
		return string(r[:n])
	}
	return string(r[:h]) + string(r[len(r)-t:])
}
