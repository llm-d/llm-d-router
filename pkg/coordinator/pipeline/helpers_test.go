package pipeline_test

// renderAware is the optional contract a step satisfies to receive the render
// service address during test pipeline wiring.
type renderAware interface{ SetServiceAddress(string) }
