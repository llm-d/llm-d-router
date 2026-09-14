// Package disagg provides profile handler plugins for the epp.
package disagg

import (
	"context"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// deciderPlugin decides whether the disaggregated stage should run for the request.
type deciderPlugin interface {
	plugin.Plugin
	disaggregate(ctx context.Context, request *scheduling.InferenceRequest, endpoint scheduling.Endpoint) bool
}

// prefixMatchInfoConsumer is implemented by deciders that read PrefixCacheMatchInfo
// from endpoint attributes. The profile handler declares the decider's key in its own
// Consumes() so the data layer wires the producer the decider reads.
type prefixMatchInfoConsumer interface {
	prefixMatchInfoDataKey() plugin.DataKey
}
