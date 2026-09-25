package kvevents

import (
	"context"
	"time"

	zmq "github.com/go-zeromq/zmq4"
)

// FetchSnapshotForTest requests one snapshot with the socket options,
// transport and timeout that bootstrap uses, and returns the reply frames.
func FetchSnapshotForTest(ctx context.Context, endpoint string) ([][]byte, error) {
	if err := registerSnapshotTransport(); err != nil {
		return nil, err
	}
	reqCtx, cancel := context.WithTimeout(ctx, snapshotTimeout)
	defer cancel()
	req := zmq.NewReq(reqCtx, zmq.WithDialerMaxRetries(0), zmq.WithDialerTimeout(time.Second), zmq.WithTimeout(snapshotTimeout))
	defer req.Close()
	if err := req.Dial(snapshotTCP(endpoint)); err != nil {
		return nil, err
	}
	if err := req.Send(zmq.NewMsg([]byte("snapshot"))); err != nil {
		return nil, err
	}
	msg, err := req.Recv()
	if err != nil {
		return nil, err
	}
	return msg.Frames, nil
}
