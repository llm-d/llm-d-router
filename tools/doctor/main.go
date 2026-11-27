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

// Command doctor analyzes router profiler captures offline.
//
//	doctor analyze --fleet-kv-blocks N [--profile NAME] CAPTURE.ndjson [CAPTURE.ndjson ...]
//
// analyze merges the captures of all router replicas in time order and prints
// the prefix-cache hit-rate ladder: the reuse the workload allows, what the
// fleet's KV capacity keeps of it, what perfect routing over per-endpoint
// caches keeps, and what the router's own picks keep, followed by the hit
// rates the router predicted and the engines reported. It replays one
// scheduling profile, since each profile picks from its own endpoints; a
// capture with several profiles, such as prefill and decode, needs --profile.
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"

	"github.com/llm-d/llm-d-router/pkg/epp/scheduling/capture"
)

const usage = "usage: doctor analyze --fleet-kv-blocks N [--profile NAME] CAPTURE.ndjson [CAPTURE.ndjson ...]"

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(args []string, out io.Writer) error {
	if len(args) == 0 || args[0] != "analyze" {
		return errors.New(usage)
	}
	fs := flag.NewFlagSet("analyze", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	fleetBlocks := fs.Int("fleet-kv-blocks", 0, "KV-cache blocks across the profile's endpoints, in the capture's block size")
	profile := fs.String("profile", "", "scheduling profile to replay; required when the captures hold more than one")
	if err := fs.Parse(args[1:]); err != nil {
		return fmt.Errorf("%w\n%s", err, usage)
	}
	if fs.NArg() == 0 || *fleetBlocks < 1 {
		return errors.New(usage)
	}
	records, err := readRecords(fs.Args())
	if err != nil {
		return err
	}
	records, err = profileRecords(records, *profile)
	if err != nil {
		return err
	}
	l, err := computeLadder(records, *fleetBlocks)
	if err != nil {
		return err
	}
	return l.write(out)
}

// readRecords reads NDJSON captures and merges them in time order.
func readRecords(paths []string) ([]capture.Record, error) {
	var records []capture.Record
	for _, path := range paths {
		fileRecords, err := readCapture(path)
		if err != nil {
			return nil, err
		}
		records = append(records, fileRecords...)
	}
	slices.SortStableFunc(records, func(a, b capture.Record) int { return a.Time.Compare(b.Time) })
	return records, nil
}

// profileRecords keeps the records of one scheduling profile: profile, or the
// only profile the records hold when profile is empty.
func profileRecords(records []capture.Record, profile string) ([]capture.Record, error) {
	if profile == "" {
		var profiles []string
		for _, r := range records {
			if !slices.Contains(profiles, r.Profile) {
				profiles = append(profiles, r.Profile)
			}
		}
		if len(profiles) > 1 {
			slices.Sort(profiles)
			return nil, fmt.Errorf("the captures hold the scheduling profiles %q: pick one with --profile", profiles)
		}
		return records, nil
	}
	records = slices.DeleteFunc(records, func(r capture.Record) bool { return r.Profile != profile })
	if len(records) == 0 {
		return nil, fmt.Errorf("no record of scheduling profile %q", profile)
	}
	return records, nil
}

// readCapture reads one NDJSON capture. Blank lines are skipped.
func readCapture(path string) ([]capture.Record, error) {
	f, err := os.Open(filepath.Clean(path))
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	var records []capture.Record
	scanner := bufio.NewScanner(f)
	scanner.Buffer(nil, 64<<20)
	for line := 1; scanner.Scan(); line++ {
		b := bytes.TrimSpace(scanner.Bytes())
		if len(b) == 0 {
			continue
		}
		var r capture.Record
		if err := json.Unmarshal(b, &r); err != nil {
			return nil, fmt.Errorf("%s:%d: %w", path, line, err)
		}
		records = append(records, r)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return records, nil
}
