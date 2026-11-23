# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared transient declaration-tensor arena for mutually exclusive Kimi preparation namespaces."""

from math import prod

import torch

from ignition.preparation.examples import declaration_tensor as _base_declaration_tensor


class ExampleArena:
    """Back large disjoint declaration namespaces on one shared transient byte slab.

    Operators in mutually exclusive model subsystems (for example, vision
    encoder vs. text decoder, or KDA vs. DSA attention) never execute
    concurrently during preparation or tuning. Sharing one byte slab across
    those namespaces bounds peak declaration memory to the largest active
    namespace while keeping every role within an operator 256-byte aligned and
    disjoint.

    Each namespace must be declared explicitly in `namespace_bases` as a
    `(base, limit)` byte span within `[0, nbytes]`. Allocations smaller than
    `min_bytes` or exceeding their namespace's remaining span return `None` so
    callers fall back explicitly to standalone tensors rather than aliasing
    across unrelated spans.
    """

    def __init__(self, device, nbytes, *, min_bytes=2 << 20, namespace_bases):
        if not namespace_bases:
            raise ValueError("ExampleArena requires at least one declared namespace span")
        total = int(nbytes)
        if total <= 0:
            raise ValueError("ExampleArena byte size must be positive")
        validated = {}
        for name, span in namespace_bases.items():
            if not isinstance(name, str) or not name:
                raise ValueError("ExampleArena namespace keys must be non-empty strings")
            base, limit = span
            if not (0 <= base <= limit <= total):
                raise ValueError(f"ExampleArena span {(base, limit)!r} for {name!r} exceeds slab size {total}")
            validated[name] = (int(base), int(limit))
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.slab = torch.zeros((total,), dtype=torch.uint8, device=self.device)
        self.min_bytes = int(min_bytes)
        self.namespace_bases = validated
        self.cursors = {}

    def allocate(self, namespace, shape, dtype):
        if self.slab is None:
            raise RuntimeError("ExampleArena has already been released")
        if namespace not in self.namespace_bases:
            raise KeyError(f"undeclared ExampleArena namespace: {namespace!r}")
        nbytes = prod(shape) * dtype.itemsize
        if nbytes < self.min_bytes:
            return None
        base, limit = self.namespace_bases[namespace]
        cursor = self.cursors.get(namespace, base)
        offset = (cursor + 255) // 256 * 256
        end = offset + nbytes
        if end > limit:
            return None
        self.cursors[namespace] = end
        return torch.empty(0, dtype=dtype, device=self.device).set_(
            self.slab.untyped_storage(), offset // dtype.itemsize, tuple(shape)
        )

    def namespace_for(self, role):
        if isinstance(role, str) and "." in role:
            prefix = role.partition(".")[0]
            if prefix in self.namespace_bases:
                return prefix
        if "common" in self.namespace_bases:
            return "common"
        raise KeyError(f"role {role!r} does not match any declared ExampleArena namespace")

    def release(self):
        """Free the backing slab immediately after preparation and tuning complete."""
        import gc

        if self.slab is None:
            return
        slab = self.slab
        self.slab = None
        slab.untyped_storage().resize_(0)
        del slab
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def declaration_tensor(examples, role, shape, dtype, device):
    """Allocate or reuse a declaration tensor, backing large Kimi roles in an attached ExampleArena."""
    arena = examples.get("__arena__")
    if arena is None:
        return _base_declaration_tensor(examples, role, shape, dtype, device)
    device = torch.device(device)
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    key = (role, shape, dtype, device)
    if key not in examples:
        storage_key = ("storage", role, dtype, device)
        storage = examples.get(storage_key)
        elements = prod(shape)
        if storage is None or storage.numel() < elements:
            allocated = arena.allocate(arena.namespace_for(role), shape, dtype) if arena.device == device else None
            storage = examples[storage_key] = (
                torch.zeros(shape, dtype=dtype, device=device) if allocated is None else allocated
            )
        examples[key] = storage.view(-1)[:elements].view(shape)
    return examples[key]
