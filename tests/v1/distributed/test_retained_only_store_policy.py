# SPDX-License-Identifier: Apache-2.0
"""retained_only store policy: only retention-shielded keys reach L2."""

# Standard
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.retention_manager import RetentionManager
from lmcache.v1.distributed.storage_controllers.store_policy import (
    AdapterDescriptor,
    RetainedOnlyStorePolicy,
    create_store_policy,
)
from lmcache.v1.distributed.storage_manager import StorageManager

N_FLOATS = 128
OBJ_BYTES = N_FLOATS * 4


def _key(chunk_id: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=0,
    )


def test_policy_filters_to_retained_keys():
    retention = RetentionManager(max_retained_bytes=10_000)
    retention.note_stored([_key(1), _key(3)], [100, 100], ttl_sec=300)
    policy = RetainedOnlyStorePolicy(retention)

    adapter_config = MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=1.0)
    adapters = [AdapterDescriptor(index=0, config=adapter_config)]

    keys = [_key(i) for i in range(4)]
    targets = policy.select_store_targets(keys, adapters)
    assert targets == {0: [_key(1), _key(3)]}

    assert policy.select_store_targets([_key(9)], adapters) == {}
    assert policy.select_l1_deletions(keys) == []


def test_factory_requires_retention_manager():
    with pytest.raises(ValueError, match="requires a retention manager"):
        create_store_policy("retained_only")
    retention = RetentionManager(max_retained_bytes=1)
    policy = create_store_policy("retained_only", retention_manager=retention)
    assert isinstance(policy, RetainedOnlyStorePolicy)
    # Policies that don't need it ignore the argument.
    create_store_policy("default", retention_manager=retention)


@pytest.fixture
def retained_only_sm():
    adapter_config = MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0)
    adapter_config.eviction_config = EvictionConfig(
        eviction_policy="LRU",
        trigger_watermark=0.8,
        eviction_ratio=0.2,
    )
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
        retention_max_fraction=0.5,
        store_policy="retained_only",
    )
    sm = StorageManager(config)
    adapter = sm.l2_adapters()[0][1]
    yield sm, adapter
    sm.close()


def _wait_l2_count(adapter, expected: int, timeout: float = 5.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(adapter._memory_objects) >= expected:
            break
        time.sleep(0.05)
    return len(adapter._memory_objects)


def test_retained_only_requires_budget():
    adapter_config = MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=1.0)
    config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=4 * 1024 * 1024,
                use_lazy=False,
                init_size_in_bytes=4 * 1024 * 1024,
                align_bytes=0x1000,
            ),
            write_ttl_seconds=600,
            read_ttl_seconds=300,
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
        l2_adapter_config=L2AdaptersConfig([adapter_config]),
        store_policy="retained_only",
    )
    with pytest.raises(ValueError, match="retained_only"):
        StorageManager(config)


def test_only_retained_keys_reach_l2(retained_only_sm):
    sm, adapter = retained_only_sm
    layout = MemoryLayoutDesc(shapes=[torch.Size([N_FLOATS])], dtypes=[torch.float32])

    keys = [_key(i) for i in range(4)]
    reserved = sm.reserve_write(keys, layout, "new")
    assert len(reserved) == 4
    for obj in reserved.values():
        obj.tensor.fill_(1.0)

    # Stamp two keys BEFORE finish_write publishes them to the controller --
    # the ordering the retention store path guarantees.
    sm.retention_manager.note_stored(
        [_key(0), _key(2)], [OBJ_BYTES, OBJ_BYTES], ttl_sec=300
    )
    sm.finish_write(keys)

    assert _wait_l2_count(adapter, 2) == 2
    assert _key(0) in adapter._memory_objects
    assert _key(2) in adapter._memory_objects
    assert _key(1) not in adapter._memory_objects


def test_adopted_keys_reach_l2_via_request_store(retained_only_sm):
    sm, adapter = retained_only_sm
    layout = MemoryLayoutDesc(shapes=[torch.Size([N_FLOATS])], dtypes=[torch.float32])

    keys = [_key(10), _key(11)]
    reserved = sm.reserve_write(keys, layout, "new")
    for obj in reserved.values():
        obj.tensor.fill_(2.0)
    sm.finish_write(keys)

    # Unstamped writes stay off L2.
    time.sleep(0.5)
    assert _key(10) not in adapter._memory_objects

    # A later retention request adopts them: stamp, then nudge.
    sm.retention_manager.note_stored(keys, [OBJ_BYTES, OBJ_BYTES], ttl_sec=300)
    sm.request_l2_store(keys)

    assert _wait_l2_count(adapter, 2) == 2
    assert _key(10) in adapter._memory_objects
    assert _key(11) in adapter._memory_objects
