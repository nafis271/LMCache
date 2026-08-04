# SPDX-License-Identifier: Apache-2.0
"""Chunk selection for store-time retention stamping."""

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    _retained_chunk_keys,
)

CHUNK = 256


def _keys(n: int) -> list[ObjectKey]:
    return [
        ObjectKey(
            chunk_hash=i.to_bytes(8, "little"),
            model_name="test_model",
            kv_rank=0,
        )
        for i in range(n)
    ]


def test_full_attention_unbounded_retains_all():
    keys = _keys(8)
    retained = _retained_chunk_keys(
        keys,
        keys,
        is_full_attention=True,
        start_token=0,
        chunk_size=CHUNK,
        bound_tokens=0,
    )
    assert retained == keys


def test_full_attention_bound_floors_to_whole_chunks():
    keys = _keys(8)
    # bound 1500 covers chunks ending at 256..1280; chunk 5 ends at 1536.
    retained = _retained_chunk_keys(
        keys,
        keys,
        is_full_attention=True,
        start_token=0,
        chunk_size=CHUNK,
        bound_tokens=1500,
    )
    assert retained == keys[:5]


def test_full_attention_bound_on_chunk_boundary_is_inclusive():
    keys = _keys(8)
    retained = _retained_chunk_keys(
        keys,
        keys,
        is_full_attention=True,
        start_token=0,
        chunk_size=CHUNK,
        bound_tokens=1024,
    )
    assert retained == keys[:4]


def test_incremental_store_uses_absolute_positions():
    keys = _keys(4)
    # Second installment of a long request: covers tokens [2048, 3072).
    # A bound of 2560 covers this store's first two chunks only.
    retained = _retained_chunk_keys(
        keys,
        keys,
        is_full_attention=True,
        start_token=2048,
        chunk_size=CHUNK,
        bound_tokens=2560,
    )
    assert retained == keys[:2]


def test_incremental_store_entirely_past_bound_retains_nothing():
    keys = _keys(4)
    retained = _retained_chunk_keys(
        keys,
        keys,
        is_full_attention=True,
        start_token=2048,
        chunk_size=CHUNK,
        bound_tokens=1024,
    )
    assert retained == []


def test_windowed_group_retains_its_tail_unconditionally():
    keys = _keys(8)
    tail = keys[6:]
    retained = _retained_chunk_keys(
        keys,
        tail,
        is_full_attention=False,
        start_token=0,
        chunk_size=CHUNK,
        bound_tokens=1024,
    )
    assert retained == tail
