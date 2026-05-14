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
	reader := bufio.NewReader(src)
	for {
		line, err := reader.ReadString('\n')
		if err != nil && err != io.EOF {
			return err
		}
		if len(line) > 0 && line[len(line)-1] == '\n' {
			line = line[:len(line)-1]
		}
		if len(line) > 0 && line[len(line)-1] == '\r' {
			line = line[:len(line)-1]
		}
		if len(line) > 0 || err != io.EOF {
			if _, writeErr := fmt.Fprintf(w, "%s\n", line); writeErr != nil {
				return writeErr
			}
			if line == "" {
				flusher.Flush()
			}
		}
		if err == io.EOF {
			return nil
		}
	}
}
