# SPDX-License-Identifier: Apache-2.0
"""Capture synthetic recovery traffic using the vLLM scheduler (see ../README.md)."""

import gzip
import itertools
import sys
from pathlib import Path
from unittest.mock import patch

import msgspec
from tests.v1.core.hybrid_checkpoint.test_recovery_unit_events import (
    UNIT_EVENTS,
    make_scheduler,
)
from tests.v1.core.hybrid_checkpoint.utils import add_request, prompt, run
from vllm.distributed.kv_events import KVEventBatch
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash


def capture_rank(rank, frames):
    scheduler, recorder = make_scheduler(
        UNIT_EVENTS, num_blocks=94, max_num_batched_tokens=1024
    )

    def turn(name, tokens, session="edit"):
        request = add_request(scheduler, name, tokens, max_tokens=5, session_id=session)
        run(scheduler, request)
        recorder.take_pending()
        for events in recorder.batches:
            batch = KVEventBatch(
                ts=float(len(frames)), events=events, data_parallel_rank=rank
            )
            frames.append(msgspec.msgpack.encode(batch))
        recorder.batches.clear()
        return list(request.all_token_ids)

    history = prompt(1, 1447)
    history[-50] += rank
    initial = turn("initial", history)
    grown = turn("own-return", initial + prompt(2, 70))
    divergent = grown.copy()
    divergent[-26] += 1
    turn("divergent", divergent + prompt(3, 65))
    turn("previous-branch", grown + prompt(4, 65))
    for i in range(8):
        turn(f"pressure-{i}", [100 + i] + history[1:], f"other-{i}")
    turn("after-pressure", grown + prompt(5, 50))


if __name__ == "__main__":
    init_none_hash(sha256)
    frames = []
    # Make eviction ranking reproducible, as in the producer's fast_clock fixture.
    with patch("time.monotonic", side_effect=itertools.count(1_000_000.0)):
        for rank in range(2):
            capture_rank(rank, frames)
    Path(sys.argv[1]).write_bytes(
        gzip.compress(msgspec.msgpack.encode(frames), compresslevel=9, mtime=0)
    )
