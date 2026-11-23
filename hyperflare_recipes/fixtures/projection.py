# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Finite strided operands shared by a projection's serial tuning candidates."""

from contextlib import nullcontext

import torch

from .resources import candidate_inputs, restore_seed


def _tensor_byte_span(tensor):
    elems = 1 + sum((s - 1) * st for s, st in zip(tensor.shape, tensor.stride(), strict=True)) if tensor.numel() else 0
    start = tensor.storage_offset() * tensor.element_size()
    end = start + elems * tensor.element_size()
    base = tensor.untyped_storage().data_ptr()
    return base + start, base + end, start, end


class Inputs:
    def __init__(self, source, output, *, reuse_disposable=False):
        borrow = False
        if reuse_disposable and source.is_cuda and output.is_cuda and source.device == output.device:
            s_lo, s_hi, s_off, s_end = _tensor_byte_span(source)
            o_lo, o_hi, o_off, o_end = _tensor_byte_span(output)
            need_bytes = (s_end - s_off) + (o_end - o_off)
            if s_hi <= o_lo or o_hi <= s_lo:
                try:
                    free_bytes, _ = torch.cuda.mem_get_info(source.device)
                except Exception:
                    free_bytes = None
                if free_bytes is not None and free_bytes < need_bytes + (256 << 20):
                    borrow = True
        if borrow:
            src_storage = source.untyped_storage()[s_off:s_end]
            out_storage = output.untyped_storage()[o_off:o_end]
            self.source = torch.empty(0, dtype=source.dtype, device=source.device).set_(
                src_storage, 0, source.shape, source.stride()
            )
            self.output = torch.empty(0, dtype=output.dtype, device=output.device).set_(
                out_storage, 0, output.shape, output.stride()
            )
            self.seed = torch.empty_strided(source.shape, source.stride(), dtype=source.dtype,
                                           device="cpu", pin_memory=True)
            if source.is_contiguous():
                generator = torch.Generator(device=source.device).manual_seed(0xB12)
                self.source.normal_(generator=generator).mul_(0.125)
                self.seed.copy_(self.source)
            else:
                generator = torch.Generator(device="cpu").manual_seed(0xB12)
                self.seed.copy_(torch.randn(source.shape, dtype=source.dtype, generator=generator).mul_(0.125))
                self.source.copy_(self.seed)
            return
        generator = torch.Generator(device=source.device).manual_seed(0xB12)
        values = torch.randn(source.shape, device=source.device, dtype=source.dtype,
                             generator=generator).mul_(0.125)
        self.seed = torch.empty_strided(source.shape, source.stride(), dtype=source.dtype,
                                       device="cpu", pin_memory=source.is_cuda)
        self.seed.copy_(values)
        # Contiguous inputs can keep the generated allocation; strided inputs
        # preserve their ABI without changing the random sequence's logical order.
        if source.is_contiguous():
            self.source = values.as_strided(source.shape, source.stride())
        else:
            self.source = torch.empty_strided(source.shape, source.stride(),
                                             dtype=source.dtype, device=source.device).copy_(values)
        del values
        self.output = torch.empty_strided(output.shape, output.stride(),
                                         dtype=output.dtype, device=output.device)

    def produce(self):
        restore_seed(self.source, self.seed)


def input_factory(source, output, *, storage=None, session=None):
    def allocate():
        with (nullcontext() if storage is None else storage.transient()):
            return Inputs(source, output, reuse_disposable=storage is not None)

    return candidate_inputs(allocate, session=session)
