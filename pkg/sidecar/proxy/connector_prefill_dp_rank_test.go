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

package proxy

import (
	. "github.com/onsi/ginkgo/v2" //nolint:revive
	. "github.com/onsi/gomega"    //nolint:revive

	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
	"github.com/llm-d/llm-d-router/test/sidecar/mock"
)

// prefillRankPath is a running proxy on one of the paths that send a prefill request.
type prefillRankPath struct {
	baseAddr        string
	prefillHostPort string
	prefill, decode *mock.ChatCompletionHandler
}

// startPrefillRankPath starts a proxy with connector and its default config.
func startPrefillRankPath(connector string) prefillRankPath {
	testInfo := sidecarConnectionTestSetup(connector)
	baseAddr := testInfo.startProxy()
	DeferCleanup(func() {
		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})
	return prefillRankPath{
		baseAddr:        baseAddr,
		prefillHostPort: testInfo.prefillBackend.URL[len("http://"):],
		prefill:         testInfo.prefillHandler,
		decode:          testInfo.decodeHandler,
	}
}

func offloadingRankPath() prefillRankPath {
	return startPrefillRankPath(constants.KVConnectorOffloading)
}

func sharedStorageRankPath() prefillRankPath {
	return startPrefillRankPath(constants.KVConnectorSharedStorage)
}

// nixlWriteParallelRankPath has a data-parallel size of 1, so the sidecar picks no
// rank of its own.
func nixlWriteParallelRankPath() prefillRankPath {
	env := startMoRIProxy(func(c *Config) {
		c.MoRIIOParallelDispatch = true
	})
	return prefillRankPath{
		baseAddr:        env.baseAddr,
		prefillHostPort: env.prefillBackend.URL[len("http://"):],
		prefill:         env.prefillHandler,
		decode:          env.decodeHandler,
	}
}

// Each path builds its prefill request from the client request, which carries the
// decode rank. The NIXL v2 serial, Mooncake, SGLang, and vllm.Inference paths have
// rank tests of their own.
var _ = DescribeTable("prefill data-parallel rank",
	func(start func() prefillRankPath, decodeRank, prefillRankHeader, wantPrefillRank string) {
		path := start()
		resp := postChatCompletions(path.baseAddr, chatCompletionsRequestBody, map[string]string{
			routing.PrefillEndpointHeader:         path.prefillHostPort,
			routing.KVCachePrefillHeader:          prefillP2PHostPort(path.prefillHostPort),
			routing.DataParallelRankHeader:        decodeRank,
			routing.PrefillDataParallelRankHeader: prefillRankHeader,
		})
		Expect(resp.Body.Close()).To(Succeed())

		Eventually(func() int { return len(path.prefill.GetCompletionRequests()) }).Should(Equal(1))
		Expect(dpRankHeader(path.prefill, 0)).To(Equal(wantPrefillRank))
		Expect(dpRankHeader(path.decode, 0)).To(Equal(decodeRank))
	},
	Entry("offloading sends the EPP's prefill rank", offloadingRankPath, "0", "1", "1"),
	Entry("offloading sends no rank when only decode has ranks", offloadingRankPath, "1", routing.NoDataParallelRank, ""),
	Entry("shared storage sends the EPP's prefill rank", sharedStorageRankPath, "0", "1", "1"),
	Entry("shared storage sends no rank when only decode has ranks", sharedStorageRankPath, "1", routing.NoDataParallelRank, ""),
	Entry("NIXL v2 write-parallel sends the EPP's prefill rank", nixlWriteParallelRankPath, "0", "1", "1"),
	Entry("NIXL v2 write-parallel sends no rank when only decode has ranks", nixlWriteParallelRankPath, "1", routing.NoDataParallelRank, ""),
)
