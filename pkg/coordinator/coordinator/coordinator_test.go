package coordinator

import (
	"context"
	"log/slog"
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func TestCoordinator(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Coordinator Suite")
}

var _ = Describe("Coordinator", func() {
	Describe("New", func() {
		It("should create a coordinator without error", func() {
			c, err := New(context.Background(), slog.Default())
			Expect(err).NotTo(HaveOccurred())
			Expect(c).NotTo(BeNil())
		})
	})

	Describe("Run", func() {
		It("should return when context is cancelled", func() {
			c, err := New(context.Background(), slog.Default())
			Expect(err).NotTo(HaveOccurred())

			ctx, cancel := context.WithCancel(context.Background())
			done := make(chan error, 1)
			go func() {
				done <- c.Run(ctx)
			}()

			cancel()
			Eventually(done).Should(Receive(HaveOccurred()))
		})
	})
})
