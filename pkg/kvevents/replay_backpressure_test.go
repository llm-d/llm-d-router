package kvevents //nolint:testpackage // exercises replay worker acknowledgement

import (
	"context"
	"encoding/binary"
	"errors"
	"testing"
	"time"

	zmq4 "github.com/go-zeromq/zmq4"
	"github.com/stretchr/testify/require"
)

type replayBlockingAdapter struct {
	entered chan struct{}
	release chan struct{}
	err     error
}

func (a *replayBlockingAdapter) ShardingKey(*RawMessage) string { return "source" }
func (a *replayBlockingAdapter) ParseMessage(*RawMessage) (string, string, EventBatch, error) {
	select {
	case a.entered <- struct{}{}:
	default:
	}
	<-a.release
	return "source", "model", EventBatch{}, a.err
}

func TestReplayWaitsForProcessing(t *testing.T) {
	for _, failure := range []bool{false, true} {
		t.Run(map[bool]string{false: "success", true: "parse failure"}[failure], func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			pool, _, _ := newTestPool(t, 16)
			adapter := &replayBlockingAdapter{entered: make(chan struct{}, 1), release: make(chan struct{})}
			if failure {
				adapter.err = errors.New("invalid replay payload")
			}
			pool.adapter = adapter
			pool.Start(ctx)
			defer pool.Shutdown(ctx)
			router := zmq4.NewRouter(ctx)
			defer router.Close()
			endpoint := "inproc://" + t.Name()
			require.NoError(t, router.Listen(endpoint))
			go func() {
				request, err := router.Recv()
				if err != nil {
					return
				}
				seq := make([]byte, 8)
				_ = router.SendMulti(zmq4.NewMsgFrom(request.Frames[0], nil, []byte("topic"), seq, []byte("payload")))
				terminal := make([]byte, 8)
				binary.BigEndian.PutUint64(terminal, ^uint64(0))
				_ = router.SendMulti(zmq4.NewMsgFrom(request.Frames[0], nil, nil, terminal, nil))
			}()
			z := newZMQSubscriber(pool, "pod", "source", "", endpoint, "topic", true)
			result := make(chan bool, 1)
			go func() { result <- z.requestReplay(ctx, 0) }()
			select {
			case <-adapter.entered:
			case <-ctx.Done():
				t.Fatal("replay never reached worker")
			}
			select {
			case <-result:
				t.Fatal("replay completed before processing")
			case <-time.After(50 * time.Millisecond):
			}
			close(adapter.release)
			select {
			case ok := <-result:
				require.Equal(t, !failure, ok)
			case <-ctx.Done():
				t.Fatal("replay did not finish after processing")
			}
			require.Equal(t, !failure, z.hasLastSeq)
		})
	}
}
