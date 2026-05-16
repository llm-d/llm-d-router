package proxy

import (
	"context"
	"net"
	"net/http"
	"net/url"
	"os"
	"strconv"

	"golang.org/x/sync/errgroup"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/routing"
)

// dataParallelHandler checks if Data Parallel handling is needed.
// Returns true if Data Parallel processing was needed
func (s *Server) dataParallelHandler(w http.ResponseWriter, r *http.Request) bool {
	dataParallelPodHostPort := r.Header.Get(routing.DataParallelEndpointHeader)
	if dataParallelPodHostPort != "" {
		s.logger.Info("The use of the x-data-parallel-host-port is deprecated. Use Istio >= 1.28.1.")
		handler := s.dataParallelProxies[dataParallelPodHostPort]
		if handler != nil {
			s.logger.V(4).Info("Data parallel routing", "to", dataParallelPodHostPort)
			handler.ServeHTTP(w, r)
		} else {
			// Shouldn't happen, send to default server
			s.logger.V(4).Info("Didn't find the Data Parallel Proxy", "for", dataParallelPodHostPort)
			w.WriteHeader(http.StatusBadRequest)
		}
		return true
	}

	s.logger.V(4).Info("skip data parallel")
	return false
}

func (s *Server) startDataParallel(ctx context.Context, grp *errgroup.Group) error {
	podIP := os.Getenv("POD_IP")
	basePort, err := strconv.Atoi(s.config.Port)
	if err != nil {
		return err
	}
	baseDecoderPort, err := strconv.Atoi(s.config.DecoderURL.Port())
	if err != nil {
		return err
	}
	decoderScheme := s.config.DecoderURL.Scheme // capture before goroutines launch
	s.dataParallelProxies[net.JoinHostPort(podIP, s.config.Port)] = s.decoderProxy

	// Fill in map of proxies, thus avoiding locks
	for idx := range s.config.DataParallelSize - 1 {
		decoderPort := strconv.Itoa(baseDecoderPort + idx + 1)
		rankPort := strconv.Itoa(basePort + idx + 1)
		hostPort := net.JoinHostPort(podIP, rankPort)
		decoderURL, err := url.Parse(decoderScheme + "://localhost:" + decoderPort)
		if err != nil {
			return err
		}
		handler := s.createDecoderProxyHandler(decoderURL, s.config.InsecureSkipVerifyForDecoder)
		s.dataParallelProxies[hostPort] = handler
	}

	for idx := range s.config.DataParallelSize - 1 {
		rankPort := strconv.Itoa(basePort + idx + 1)
		decoderPort := strconv.Itoa(baseDecoderPort + idx + 1)
		decoderURL, err := url.Parse(decoderScheme + "://localhost:" + decoderPort)
		if err != nil {
			return err
		}

		clone := s.Clone()
		clone.config.Port = rankPort
		clone.config.DecoderURL = decoderURL
		clone.forwardDataParallel = false

		grp.Go(func() error {
			clone.logger = log.FromContext(ctx).WithName("proxy server on port " + rankPort)
			// Configure handlers
			clone.handler = clone.createRoutes()
			clone.setKVConnector()

			return clone.startHTTP(ctx)
		})
	}
	return nil
}
