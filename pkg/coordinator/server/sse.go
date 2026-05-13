package server

import (
	"bufio"
	"fmt"
	"io"
	"net/http"
)

// SetSSEHeaders configures response headers for Server-Sent Events streaming.
func SetSSEHeaders(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
}

// StreamSSE reads SSE events from src and writes them to the client ResponseWriter,
// flushing after each complete event (blank line).
func StreamSSE(w http.ResponseWriter, flusher http.Flusher, src io.Reader) error {
	scanner := bufio.NewScanner(src)
	for scanner.Scan() {
		line := scanner.Text()
		if _, err := fmt.Fprintf(w, "%s\n", line); err != nil {
			return err
		}
		if line == "" {
			flusher.Flush()
		}
	}
	return scanner.Err()
}
