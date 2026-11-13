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

package asyncbroker

import (
	"context"
	"fmt"
	"time"

	"github.com/go-logr/logr"
	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"
)

// asyncClassification mirrors the redis-quota gate's reserved/overflow
// outcome.
type asyncClassification string

const (
	asyncClassificationReserved asyncClassification = "reserved"
	asyncClassificationOverflow asyncClassification = "overflow"
)

// asyncQuotaClassifier classifies passthrough requests against the same
// Redis counters and key scheme as the async processor's redis-quota gate
// (concurrency mode, classifying semantics: over-quota is labeled overflow,
// never blocked). Queued modes are classified by the AP's gates at dequeue,
// not here.
type asyncQuotaClassifier struct {
	rdb    *redis.Client
	cfg    asyncQuotaConfig
	logger logr.Logger
}

// quotaCallGuardTTL outlasts the retries of one script call. Under the
// go-redis defaults the last retry runs at most about 53s after the first
// attempt: the first attempt's 5s read timeout, then three retries of up to
// 16s each (pool wait, dial and write timeouts). A read_timeout above about
// 14s or a larger max_retries in the Redis URL can push the retries past
// this TTL.
const quotaCallGuardTTL = 2 * time.Minute

// quotaCallGuardKey marks that the call holding token has changed the counter
// at key. go-redis retries a command whose reply was lost, so a counter script
// can run twice for one call; the guard keeps the retry from counting again.
// The format matches the redis-quota gate's guard key, so one Redis ACL key
// pattern covers both.
func quotaCallGuardKey(key, token string) string {
	return fmt.Sprintf("quota-call:%s:%s", key, token)
}

// quotaAcquireScript matches the redis-quota gate's atomic
// check-and-increment, including the call guard and the TTL refresh on every
// acquire so counters cannot expire while requests are in flight.
var quotaAcquireScript = redis.NewScript(`
if redis.call("EXISTS", KEYS[2]) == 1 then
	return 1
end
local current = redis.call("GET", KEYS[1])
if current and tonumber(current) >= tonumber(ARGV[1]) then
	return 0
end
redis.call("SET", KEYS[2], "1", "PX", ARGV[3])
redis.call("INCR", KEYS[1])
redis.call("EXPIRE", KEYS[1], ARGV[2])
return 1
`)

//nolint:dupword // Each Lua "end" closes its own nested block.
var quotaReleaseScript = redis.NewScript(`
if not redis.call("SET", KEYS[2], "1", "NX", "PX", ARGV[2]) then
	return
end
local current = redis.call("GET", KEYS[1])
if current and tonumber(current) > 0 then
	local remaining = redis.call("DECR", KEYS[1])
	if remaining > 0 then
		redis.call("EXPIRE", KEYS[1], ARGV[1])
	end
end
`)

// classify returns the tenant's classification and, for reserved
// acquisitions, a release func that must be called when the request
// completes (nil otherwise). Redis errors fail open: the request is
// classified reserved with no release, so a Redis outage never blocks live
// traffic (quota briefly unenforced).
func (q *asyncQuotaClassifier) classify(ctx context.Context, tenant string) (asyncClassification, func(), error) {
	limit, ok := q.cfg.Limits[tenant]
	if !ok || limit <= 0 {
		return asyncClassificationReserved, nil, nil
	}

	key := fmt.Sprintf("%s%s:%s", q.cfg.Prefix, q.cfg.Attribute, tenant)
	keys := []string{key, quotaCallGuardKey(key, uuid.NewString())}
	res, err := quotaAcquireScript.Run(ctx, q.rdb, keys, limit, q.cfg.WindowSeconds, quotaCallGuardTTL.Milliseconds()).Result()
	if err != nil {
		return asyncClassificationReserved, nil, fmt.Errorf("quota check failed (failing open): %w", err)
	}
	if v, ok := res.(int64); !ok || v == 0 {
		return asyncClassificationOverflow, nil, nil
	}

	release := func() {
		// Background context: the release must run even if the request
		// context is already canceled. A failed release leaves the counter
		// elevated until its window TTL, skewing the tenant toward overflow,
		// so it must leave a trail.
		keys := []string{key, quotaCallGuardKey(key, uuid.NewString())}
		if err := quotaReleaseScript.Run(context.Background(), q.rdb, keys, q.cfg.WindowSeconds, quotaCallGuardTTL.Milliseconds()).Err(); err != nil {
			q.logger.Error(err, "quota release failed, counter stays elevated until the window ttl", "tenant", tenant)
		}
	}
	return asyncClassificationReserved, release, nil
}
