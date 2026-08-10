# SPDX-License-Identifier: Apache-2.0
"""
Observability protocol definitions.

This module defines protocols for:
- REPORT_BLOCK_ALLOCATION: Report vLLM GPU block allocation events
  (fire-and-forget, no response)
"""

# First Party
from lmcache.v1.multiprocess.custom_types import (
    BlockAllocationRecord,
    KvEventDrainRecord,
)
from lmcache.v1.multiprocess.protocols.base import HandlerType, ProtocolDefinition

# Define request names for this protocol group
REQUEST_NAMES = [
    "REPORT_BLOCK_ALLOCATION",
    "DRAIN_KV_EVENTS",
]


def get_protocol_definitions() -> dict[str, ProtocolDefinition]:
    """
    Returns protocol definitions for observability operations.

    Returns:
        Dictionary mapping request names to their protocol definitions
    """
    return {
        # Report vLLM block allocation
        # Payload:
        #   - instance_id: int - scheduler instance ID
        #   - model_name: str - model name from the adapter
        #   - records: list[BlockAllocationRecord] - allocation records
        # Returns: None (fire-and-forget)
        "REPORT_BLOCK_ALLOCATION": ProtocolDefinition(
            payload_classes=[int, str, list[BlockAllocationRecord]],
            response_class=None,
            handler_type=HandlerType.BLOCKING,
        ),
        # Drain queued Dynamo KV events (connector-sink mode)
        # Payload: [] (none)
        # Returns: list[KvEventDrainRecord] - queued events, oldest first
        "DRAIN_KV_EVENTS": ProtocolDefinition(
            payload_classes=[],
            response_class=list[KvEventDrainRecord],
            handler_type=HandlerType.SYNC,
        ),
    }
