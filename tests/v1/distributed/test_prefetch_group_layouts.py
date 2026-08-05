# SPDX-License-Identifier: Apache-2.0
"""L2 prefetch must allocate L1 buffers with each object group's own layout.

Hybrid models split KV into object groups with different chunk layouts.
A prefetch whose keys span groups used to reserve every L1 buffer with the
single request layout, so every other group's load got a wrong-sized
buffer (truncated onboard or failed load; observed on DeepSeek-V4-Flash as
retrieve-time size mismatches and half-restored KV).
"""

# Standard
import os
import select

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    AttnWindowDesc,
    MemoryLayoutDesc,
    ObjectKey,
    TrimPolicy,
)
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)

G0_FLOATS = 128
G1_FLOATS = 2048
G0_BYTES = G0_FLOATS * 4
G1_BYTES = G1_FLOATS * 4


def _key(chunk_id: int, group_id: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=0,
        object_group_id=group_id,
    )


def _memory_obj(n_floats: int, fill: float) -> TensorMemoryObj:
    raw = torch.empty(n_floats, dtype=torch.float32)
    raw.fill_(fill)
    metadata = MemoryObjMetadata(
        shape=torch.Size([n_floats]),
        dtype=torch.float32,
        address=0,
        phy_size=n_floats * 4,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw, metadata, parent_allocator=None)


def _wait_fd(fd: int, timeout: float = 5.0) -> bool:
    poll = select.poll()
    poll.register(fd, select.POLLIN)
    events = poll.poll(timeout * 1000)
    if not events:
        return False
    try:
        os.eventfd_read(fd)
    except BlockingIOError:
        pass
    return True


def _store_sync(adapter: MockL2Adapter, key: ObjectKey, obj: TensorMemoryObj):
    adapter.submit_store_task([key], [obj])
    assert _wait_fd(adapter.get_store_event_fd()), "store event timed out"
    adapter.pop_completed_store_tasks()


@pytest.fixture
def storage_manager_with_mock_l2():
    adapter_config = MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0)
    config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=16 * 1024 * 1024,
                use_lazy=False,
                init_size_in_bytes=16 * 1024 * 1024,
                align_bytes=0x1000,
            ),
            write_ttl_seconds=600,
            read_ttl_seconds=300,
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
        l2_adapter_config=L2AdaptersConfig([adapter_config]),
    )
    sm = StorageManager(config)
    adapter = sm.l2_adapters()[0][1]
    yield sm, adapter
    sm.close()


def test_prefetch_reserves_per_group_layouts(storage_manager_with_mock_l2):
    """Keys spanning two groups with different layouts onboard correctly."""
    sm, adapter = storage_manager_with_mock_l2

    # Group 0 stored for every chunk; group 1 tail-only (chunks 2, 3) --
    # the shape a tail-only windowed group leaves behind.
    num_chunks = 4
    g1_stored = {2, 3}
    for c in range(num_chunks):
        _store_sync(adapter, _key(c, 0), _memory_obj(G0_FLOATS, float(c)))
    for c in g1_stored:
        _store_sync(adapter, _key(c, 1), _memory_obj(G1_FLOATS, float(100 + c)))

    # Chunk-major key order, matching the lookup module's layout.
    keys = []
    for c in range(num_chunks):
        keys.append(_key(c, 0))
        keys.append(_key(c, 1))

    g0_layout = MemoryLayoutDesc(
        shapes=[torch.Size([G0_FLOATS])], dtypes=[torch.float32]
    )
    g1_layout = MemoryLayoutDesc(
        shapes=[torch.Size([G1_FLOATS])], dtypes=[torch.float32]
    )

    handle = sm.submit_prefetch_task(
        keys,
        g0_layout,
        policy=TrimPolicy.SPARSE,
        attn_desc=AttnWindowDesc(num_chunks_in_sw=[-1, 2]),
        layout_descs=[g0_layout, g1_layout],
    )
    assert sm.wait_prefetch_status(handle, timeout=10.0), "prefetch timed out"
    found = sm.query_prefetch_status(handle)
    assert found is not None
    assert found.popcount() == num_chunks + len(g1_stored)

    good_keys, good_objs = sm.unsafe_read(keys)
    pairs = list(zip(good_keys, good_objs, strict=True))
    sizes = {key: obj.get_size() for key, obj in pairs}
    for c in range(num_chunks):
        assert sizes[_key(c, 0)] == G0_BYTES
    for c in g1_stored:
        assert sizes[_key(c, 1)] == G1_BYTES

    # Content survives the round trip with the correct per-group sizes.
    values = {key: obj.tensor[0].item() for key, obj in pairs}
    assert values[_key(1, 0)] == 1.0
    assert values[_key(3, 1)] == 103.0


def test_prefetch_single_layout_unchanged(storage_manager_with_mock_l2):
    """Uniform-layout callers that pass no per-group list behave as before."""
    sm, adapter = storage_manager_with_mock_l2

    for c in range(3):
        _store_sync(adapter, _key(c, 0), _memory_obj(G0_FLOATS, float(c)))
    keys = [_key(c, 0) for c in range(3)]
    layout = MemoryLayoutDesc(shapes=[torch.Size([G0_FLOATS])], dtypes=[torch.float32])

    handle = sm.submit_prefetch_task(keys, layout)
    assert sm.wait_prefetch_status(handle, timeout=10.0)
    found = sm.query_prefetch_status(handle)
    assert found is not None
    assert found.popcount() == 3
