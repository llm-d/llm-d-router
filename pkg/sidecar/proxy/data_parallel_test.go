package proxy

import (
	"context"
	"net/http/httptest"
	"net/url"
	"os"
	"strconv"

	. "github.com/onsi/ginkgo/v2" // nolint:revive
	. "github.com/onsi/gomega"    // nolint:revive
	"golang.org/x/sync/errgroup"

	"github.com/llm-d/llm-d-router/pkg/common/routing"
	sidecarmock "github.com/llm-d/llm-d-router/test/sidecar/mock"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

const (
	testDataParallelSize = 2
)

var _ = Describe("Data Parallel support", func() {
	When("configured with --data-parallel-size > 1", func() {
		It("should create an extra proxy", func() {
			ctx := newTestContext()
			ctx, cancel := context.WithCancel(ctx)
			grp, ctx := errgroup.WithContext(ctx)

			// proxy.startDataParallel starts listeners on the ports following
			// the proxy's main port. To avoid problems, get a free port and
			// tell the proxy that it is listening on that port minus one.
			freePort, err := testutils.GetFreePort()
			Expect(err).ToNot(HaveOccurred())
			tmpPort, err := strconv.Atoi(freePort)
			fakeProxyPort := tmpPort - 1
			Expect(err).ToNot(HaveOccurred())

			// The data parallel support, assumes that the decoders are
			// listening on a set of contiguous ports. Get a free port
			// and fake the first one by being the free port minus one.
			rank1Handler := sidecarmock.GenericHandler{}
			rank1Server := httptest.NewServer(&rank1Handler)
			tempURL, err := url.Parse(rank1Server.URL)
			Expect(err).ToNot(HaveOccurred())
			tmpPort, err = strconv.Atoi(tempURL.Port())
			Expect(err).ToNot(HaveOccurred())
			fakeDecodePort := tmpPort - 1

			DeferCleanup(os.Setenv, "POD_IP", os.Getenv("POD_IP"))
			err = os.Setenv("POD_IP", "127.0.0.1")
			Expect(err).ToNot(HaveOccurred())

			decodeURL, err := url.Parse("http://localhost:" + strconv.Itoa(fakeDecodePort))
			Expect(err).ToNot(HaveOccurred())
			cfg := Config{
				Port:             strconv.Itoa(fakeProxyPort),
				DecoderURL:       decodeURL,
				KVConnector:      KVConnectorNIXLV2,
				DataParallelSize: testDataParallelSize,
			}
			theProxy := NewProxy(cfg)
			theProxy.allowlistValidator, err = NewAllowlistValidator(false, DefaultPoolGroup, "", "")
			Expect(err).ToNot(HaveOccurred())

			err = theProxy.startDataParallel(ctx, grp)
			Expect(err).ToNot(HaveOccurred())

			Expect(theProxy.dataParallelProxies).To(HaveLen(testDataParallelSize))
			handler := theProxy.dataParallelProxies["127.0.0.1:"+strconv.Itoa(fakeProxyPort+1)]
			Expect(handler).ToNot(BeNil())

			rank0Handler := sidecarmock.GenericHandler{}
			rank0Server := httptest.NewServer(&rank0Handler)
			tempURL, err = url.Parse(rank0Server.URL)
			Expect(err).ToNot(HaveOccurred())
			theProxy.config.DecoderURL = tempURL

			proxyHandler := theProxy.createRoutes()
			req := httptest.NewRequest("POST", "/v1/completions", nil)
			resp := httptest.NewRecorder()
			proxyHandler.ServeHTTP(resp, req)
			Expect(int(rank0Handler.RequestCount.Load())).To(Equal(1))
			Expect(int(rank1Handler.RequestCount.Load())).To(Equal(0))

			req.Header.Add(routing.DataParallelEndpointHeader, "127.0.0.1:"+strconv.Itoa(fakeProxyPort+1))
			resp = httptest.NewRecorder()
			proxyHandler.ServeHTTP(resp, req)
			Expect(int(rank0Handler.RequestCount.Load())).To(Equal(1))
			Expect(int(rank1Handler.RequestCount.Load())).To(Equal(1))

			rank0Server.Close()
			rank1Server.Close()

			cancel()
			err = grp.Wait()
			Expect(err).ToNot(HaveOccurred())
		})
	})
})
