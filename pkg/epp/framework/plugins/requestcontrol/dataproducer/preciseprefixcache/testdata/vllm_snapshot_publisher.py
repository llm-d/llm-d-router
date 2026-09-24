# SPDX-License-Identifier: Apache-2.0
"""Actual vLLM publisher driven by the router's cross-language regression."""
import json
import os
import socket
import sys
import time

from vllm.distributed.kv_events import (
    AllBlocksCleared, BlockRemoved, BlockStored, KVEventBatch, ZmqEventPublisher,
)

engine_block_size = int(os.environ.get("VLLM_TEST_BLOCK_SIZE", "4"))
hashes = [101, 102] if engine_block_size == 4 else [101]
ports = None
publisher = None

def store(medium, tokens):
    return BlockStored(block_hashes=hashes, parent_block_hash=None,
                       token_ids=tokens, block_size=engine_block_size, lora_id=None,
                       medium=medium, lora_name=None)

def emit(events):
    publisher.publish(KVEventBatch(ts=time.time(), events=events))
    publisher._event_queue.join()

def start():
    global publisher, ports
    def free_port():
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]
    live, snapshot = ports or (free_port(), free_port())
    publisher = ZmqEventPublisher(0, endpoint=f"tcp://*:{live}",
                                 snapshot_endpoint=f"tcp://*:{snapshot}",
                                 topic="kv@127.0.0.1:8000@test-model")
    config = publisher.get_publisher_config()
    ports = (int(config.endpoint.rsplit(":", 1)[1]),
             int(config.snapshot_endpoint.rsplit(":", 1)[1]))
    emit([store("GPU", list(range(1, 9)))])

start()
print(json.dumps(dict(result="ready", live_port=ports[0], snapshot_port=ports[1])), flush=True)
try:
    for line in sys.stdin:
        action = line.strip()
        if action == "offload-reset":
            emit([store("CPU", []), store("CPU", []), AllBlocksCleared()])
        elif action in ("remove-one-copy", "remove-last-copy"):
            emit([BlockRemoved(block_hashes=hashes, medium="CPU")])
        elif action == "store-gpu":
            emit([store("GPU", list(range(1, 9)))])
        elif action == "fail-recorder":
            # The state a recorder reaches on any internal failure.
            publisher._snapshot_recorder._failed.set()
        elif action == "stop":
            publisher.shutdown()
            publisher = None
        elif action == "restart":
            if publisher:
                publisher.shutdown()
            start()
        else:
            raise ValueError(action)
        print(json.dumps(dict(result="ok")), flush=True)
finally:
    if publisher:
        publisher.shutdown()
