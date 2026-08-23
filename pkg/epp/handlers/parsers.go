/*
Copyright 2025 The Kubernetes Authors.

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

package handlers

import (
	"fmt"
	"strings"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

type parserEntry struct {
	parser          fwkrh.Parser
	normalizedPaths []string
}

// ParserRegistry handles the resolution of incoming requests to the appropriate Parser.
type ParserRegistry struct {
	entries        []parserEntry
	fallbackParser fwkrh.Parser
	parsers        []fwkrh.Parser
}

// NewParserRegistry builds a central resolution table from a list of active parsers.
// Resolve uses longest-suffix matching on each parser's claimed paths (longest match wins).
func NewParserRegistry(parsers []fwkrh.Parser, logger logr.Logger) *ParserRegistry {
	registry := &ParserRegistry{}
	seenTypes := make(map[string]bool)
	logger = logger.WithName("ParserRegistry")

	for i, parser := range parsers {
		paths := parser.Claims().Paths
		typeName := parser.TypedName().Type
		if len(paths) == 0 {
			// A parser with no paths acts as a fallback. Stop processing subsequent
			// parsers once a fallback is registered, as it will capture all remaining traffic.
			registry.fallbackParser = parser
			if !seenTypes[typeName] {
				seenTypes[typeName] = true
				registry.parsers = append(registry.parsers, parser)
			} else {
				logger.Info("Parser type is already registered, skipping duplicate type configuration", "severity", "warning", "type", typeName, "parser", parser.TypedName().Name)
			}
			for _, skipped := range parsers[i+1:] {
				logger.Info("Parser is skipped because it is configured after fallback parser", "severity", "warning", "skippedParser", skipped.TypedName().Name, "fallbackParser", parser.TypedName().Name)
			}
			break
		}
		if !seenTypes[typeName] {
			seenTypes[typeName] = true
			registry.parsers = append(registry.parsers, parser)
			normalizedPaths := make([]string, 0, len(paths))
			for _, suffix := range paths {
				normalizedPaths = append(normalizedPaths, normalizeSuffix(suffix))
			}
			registry.entries = append(registry.entries, parserEntry{
				parser:          parser,
				normalizedPaths: normalizedPaths,
			})
		} else {
			logger.Info("Parser type is already registered, skipping duplicate type configuration", "severity", "warning", "type", typeName, "parser", parser.TypedName().Name)
		}
	}
	return registry
}

// Parsers returns all unique active parsers registered in the registry.
func (pr *ParserRegistry) Parsers() []fwkrh.Parser {
	return pr.parsers
}

// Resolve resolves an incoming request path to the matching Parser using suffix matching.
// When multiple parsers match, the longest matching suffix wins to ensure more-specific
// parsers take priority regardless of registration order.
func (pr *ParserRegistry) Resolve(path string) (fwkrh.Parser, error) {
	var longestMatched fwkrh.Parser
	var longestMatchLen int
	for _, entry := range pr.entries {
		for _, suffix := range entry.normalizedPaths {
			if request.MatchPathSuffix(path, suffix) && len(suffix) > longestMatchLen {
				longestMatched = entry.parser
				longestMatchLen = len(suffix)
			}
		}
	}
	if longestMatched != nil {
		return longestMatched, nil
	}
	if pr.fallbackParser != nil {
		return pr.fallbackParser, nil
	}

	return nil, fmt.Errorf("no parser registered matching path suffix for: %s", path)
}

// normalizeSuffix cleans up a suffix by removing leading and trailing slashes.
func normalizeSuffix(suffix string) string {
	return strings.Trim(strings.TrimSpace(suffix), "/")
}
