# SPDX-License-Identifier: Apache-2.0

"""Pure block-building logic for Dynamo KV cache events.

Slices a token sequence into full blocks and chain-hashes each block (via
:class:`~lmcache.v1.multiprocess.token_hasher.TokenHasher`), producing the
``(parent_hash, blocks)`` shape used to build Dynamo ``BlockStored`` events.
The chain is carried in the hasher's *native* form (``bytes`` for blake3,
``int`` for sha256_cbor/builtin) to avoid losing entropy; only the per-block
value returned to the caller is reduced to Dynamo's signed-i64 block hash.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.token_hasher import TokenHasher

# A hasher's native hash value: ``bytes`` (blake3) or ``int`` (sha256_cbor /
# builtin). Opaque; only ever fed back into the hasher's own methods.
NativeHash = object


def _to_signed_i64(native_hash: NativeHash, hasher: TokenHasher) -> int:
    """Reduce a native hash to a signed 64-bit integer.

    Takes the leading 8 bytes of ``hasher.hash_to_bytes(native_hash)`` so the
    result fits Dynamo's ``i64`` regardless of digest width (blake3 yields 32
    bytes; sha256_cbor / builtin already yield 8).

    Args:
        native_hash: A hash value in the hasher's native form.
        hasher: The hasher that produced ``native_hash``.

    Returns:
        The hash reduced to a signed 64-bit integer.
    """
    raw = hasher.hash_to_bytes(native_hash)
    # vLLM's maybe_convert_block_hash() reduces a digest with
    # int.from_bytes(all_bytes) & (2**64 - 1) — i.e. the TRAILING 8 bytes —
    # so we must match that window or engine and LMCache hashes diverge.
    val = int.from_bytes(raw, byteorder="big") & ((1 << 64) - 1)
    return val - 2**64 if val >= 2**63 else val


def build_blocks(
    token_ids: list[int],
    kv_block_size: int,
    prefix_hash: NativeHash,
    hasher: TokenHasher,
    hash_block_size: int | None = None,
) -> tuple[int | None, list[tuple[list[int], int]]]:
    """Slice ``token_ids`` into full blocks and chain-hash each block.

    Trailing tokens that do not fill a block are discarded. Block 0 is hashed
    against ``prefix_hash``, each later block against the previous block's
    native hash, forming the same prefix chain the storage path computes -- so
    two calls sharing a token prefix produce identical hashes for the shared
    blocks.

    Args:
        token_ids: Token sequence to slice. Only full blocks are used.
        kv_block_size: Tokens per block. Must be positive.
        prefix_hash: Native hash of the block preceding ``token_ids``; pass
            ``hasher.none_hash`` at the sequence start.
        hasher: Hasher used for all hashing.

    Returns:
        ``(parent_hash, blocks)``. ``parent_hash`` is the signed-i64 form of
        ``prefix_hash``, or ``None`` at the sequence start. ``blocks`` is an
        ordered list of ``(block_token_ids, block_hash_i64)``, one per block.

    Raises:
        ValueError: If ``kv_block_size`` is not positive.
    """
    if kv_block_size <= 0:
        raise ValueError(f"kv_block_size must be positive (got {kv_block_size})")
    # vLLM chain-hashes at its RESOLVED kernel block size (e.g. 4 for hybrid
    # DSv4), then events expose every (kv_block_size/hash_block_size)-th chain
    # value. Chaining here at the same micro granularity is required for the
    # emitted hashes to equal the engine's.
    micro = hash_block_size or kv_block_size
    if micro <= 0 or kv_block_size % micro != 0:
        raise ValueError(
            f"hash_block_size {micro} must divide kv_block_size {kv_block_size}"
        )

    if prefix_hash == hasher.none_hash:
        parent_hash: int | None = None
    else:
        parent_hash = _to_signed_i64(prefix_hash, hasher)

    blocks: list[tuple[list[int], int]] = []
    prev_native = prefix_hash
    num_full_blocks = len(token_ids) // kv_block_size
    for i in range(num_full_blocks):
        block_tokens = token_ids[i * kv_block_size : (i + 1) * kv_block_size]
        block_native = prev_native
        for j in range(0, kv_block_size, micro):
            block_native = hasher.hash_tokens(block_tokens[j : j + micro], block_native)
        blocks.append((block_tokens, _to_signed_i64(block_native, hasher)))
        prev_native = block_native

    return parent_hash, blocks
