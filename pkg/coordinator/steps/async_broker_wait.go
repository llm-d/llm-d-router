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

package steps

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/go-logr/logr"
	"github.com/redis/go-redis/v9"
)

// asyncResultWaiter implements the multiplexed wait-mode wake-up: one Redis
// pub/sub connection per coordinator replica, on which each waiting handler
// subscribes to its own result key's keyspace-notification channel before
// parking. Same-replica delivery is structural: the only replica subscribed
// to a request's channel is the one holding its connection. Notifications
// are fire and forget, so waiters keep a slow backup poll.
type asyncResultWaiter struct {
	rdb    *redis.Client
	pubsub *redis.PubSub
	prefix string // keyspace channel prefix, "__keyspace@<db>__:"

	mu sync.Mutex
	// waiters holds one wake signal per parked handler. Concurrent waits on
	// the same key are legal (a client may retry a wait by id), so each
	// channel name keeps a list and the pub/sub subscription is held until
	// the last waiter leaves.
	waiters map[string][]chan struct{}
}

// asyncNotificationsEnabled reports whether the Redis server's keyspace
// notifications cover list events on keyspace channels (flags K and l, or
// the A alias for all classes).
func asyncNotificationsEnabled(ctx context.Context, rdb *redis.Client) bool {
	res, err := rdb.ConfigGet(ctx, "notify-keyspace-events").Result()
	if err != nil {
		return false
	}
	flags := res["notify-keyspace-events"]
	return strings.Contains(flags, "K") && (strings.Contains(flags, "l") || strings.Contains(flags, "A"))
}

func newAsyncResultWaiter(ctx context.Context, rdb *redis.Client, db int, logger logr.Logger) *asyncResultWaiter {
	w := &asyncResultWaiter{
		rdb:     rdb,
		prefix:  fmt.Sprintf("__keyspace@%d__:", db),
		waiters: make(map[string][]chan struct{}),
	}
	w.pubsub = rdb.Subscribe(ctx)
	go w.receive(ctx, logger)
	return w
}

func (w *asyncResultWaiter) receive(ctx context.Context, logger logr.Logger) {
	ch := w.pubsub.Channel()
	for {
		select {
		case <-ctx.Done():
			_ = w.pubsub.Close()
			return
		case msg, ok := <-ch:
			if !ok {
				logger.Info("result waiter pub/sub channel closed")
				return
			}
			w.mu.Lock()
			for _, wake := range w.waiters[msg.Channel] {
				select {
				case wake <- struct{}{}:
				default: // waiter already signaled
				}
			}
			w.mu.Unlock()
		}
	}
}

// register subscribes to the key's notification channel and returns a wake
// channel plus a cleanup func. Callers must check the key AFTER register
// returns: a result that landed before the subscription is only found by
// that check.
func (w *asyncResultWaiter) register(ctx context.Context, key string) (<-chan struct{}, func(), error) {
	channel := w.prefix + key
	wake := make(chan struct{}, 1)
	w.mu.Lock()
	first := len(w.waiters[channel]) == 0
	w.waiters[channel] = append(w.waiters[channel], wake)
	w.mu.Unlock()
	if first {
		if err := w.pubsub.Subscribe(ctx, channel); err != nil {
			w.mu.Lock()
			w.removeWaiter(channel, wake)
			w.mu.Unlock()
			return nil, nil, err
		}
	}
	cleanup := func() {
		// Unsubscribe with a fresh context: cleanup runs on request exit,
		// when the request context may already be canceled. The lock is held
		// across the unsubscribe so a waiter registering concurrently either
		// sees remaining waiters (no unsubscribe) or an empty list (it
		// re-subscribes, ordered after this unsubscribe on the connection).
		unsubCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		w.mu.Lock()
		defer w.mu.Unlock()
		if w.removeWaiter(channel, wake) == 0 {
			_ = w.pubsub.Unsubscribe(unsubCtx, channel)
		}
	}
	return wake, cleanup, nil
}

// removeWaiter drops wake from the channel's list and reports how many
// waiters remain. The caller holds w.mu.
func (w *asyncResultWaiter) removeWaiter(channel string, wake chan struct{}) int {
	list := w.waiters[channel]
	for i, c := range list {
		if c == wake {
			list = append(list[:i], list[i+1:]...)
			break
		}
	}
	if len(list) == 0 {
		delete(w.waiters, channel)
		return 0
	}
	w.waiters[channel] = list
	return len(list)
}
