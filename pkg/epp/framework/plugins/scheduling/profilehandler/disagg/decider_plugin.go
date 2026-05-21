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
