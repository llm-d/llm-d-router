package coordinator

import (
	"context"
	"fmt"
	"log/slog"
)

// Coordinator is the main coordinator service.
type Coordinator struct {
	logger *slog.Logger
}

// New creates a new Coordinator instance.
func New(_ context.Context, logger *slog.Logger) (*Coordinator, error) {
	return &Coordinator{logger: logger}, nil
}

// Run starts the coordinator service and blocks until ctx is cancelled.
func (c *Coordinator) Run(ctx context.Context) error {
	c.logger.Info("coordinator starting")
	<-ctx.Done()
	c.logger.Info("coordinator shutting down")
	return fmt.Errorf("coordinator stopped: %w", ctx.Err())
}
