# SPDX-License-Identifier: Apache-2.0

"""Queue-backed KV event emitter for the connector-sink path.

Duck-types :class:`DynamoKvEventEmitter` (``publish_stored`` /
``publish_removed`` / ``close``) but instead of a ZMQ ``PUB`` socket it
appends :class:`KvEventDrainRecord`s to a bounded in-process deque. The
vLLM-side ``LMCacheMPConnector.take_events()`` drains the queue over the MP
protocol (``DRAIN_KV_EVENTS``) and hands the events to vLLM's own KV-event
publisher, so CPU-tier events ride the engine's existing ZMQ stream (one
port, one seq space, ring-buffer + /kv_recover recovery included).

Overflow policy: drop-oldest with a rate-limited warning. Losing the oldest
records under sustained backpressure degrades external indexers gracefully
(stale CPU entries linger until churn) without ever blocking the store path.
"""

# Future
from __future__ import annotations

# Standard
from collections import deque
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.custom_types import KvEventDrainRecord

logger = init_logger(__name__)

_WARN_INTERVAL_S = 30.0


class QueueKvEventEmitter:
    """Bounded, thread-safe queue of KV event records for connector drains."""

    def __init__(
        self,
        medium: str | None = None,
        max_records: int = 10_000,
    ) -> None:
        """Create the emitter.

        Args:
            medium: Storage-medium tag stamped on every record (e.g.
                ``"CPU_PINNED"``). ``None`` means the consumer's default.
            max_records: Maximum queued records before drop-oldest kicks in.
        """
        self._medium = medium
        self._queue: deque[KvEventDrainRecord] = deque(maxlen=max_records)
        self._lock = threading.Lock()
        self._dropped = 0
        self._last_warn = 0.0

    def publish_stored(
        self,
        token_ids: list[int],
        block_hashes: list[int],
        parent_hash: int | None,
        block_size: int,
    ) -> None:
        """Queue a stored record (same signature as the ZMQ emitter)."""
        rec = KvEventDrainRecord(
            kind="stored",
            block_hashes=block_hashes,
            token_ids=token_ids,
            parent_hash=parent_hash,
            block_size=block_size,
            medium=self._medium,
        )
        self._append(rec)

    def publish_removed(self, block_hashes: list[int]) -> None:
        """Queue a removed record (same signature as the ZMQ emitter)."""
        rec = KvEventDrainRecord(
            kind="removed",
            block_hashes=block_hashes,
            token_ids=[],
            parent_hash=None,
            block_size=0,
            medium=self._medium,
        )
        self._append(rec)

    def drain(self) -> list[KvEventDrainRecord]:
        """Return and clear all queued records (oldest first)."""
        with self._lock:
            records = list(self._queue)
            self._queue.clear()
            return records

    def close(self) -> None:
        """Nothing to release; kept for emitter interface parity."""

    def _append(self, rec: KvEventDrainRecord) -> None:
        with self._lock:
            if len(self._queue) == self._queue.maxlen:
                self._dropped += 1
                now = time.monotonic()
                if now - self._last_warn > _WARN_INTERVAL_S:
                    self._last_warn = now
                    logger.warning(
                        "Dynamo KV event queue full; dropped %d oldest "
                        "records so far (connector not draining fast enough?)",
                        self._dropped,
                    )
            self._queue.append(rec)
