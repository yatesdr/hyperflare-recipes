# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Kimi EXL3 replacement inspection, ExampleArena, and full 65,536-patch vision catalog."""

import json

import pytest
import torch

from ignition import planning, registry
from ignition.declaration import Speculation
from ignition.errors import UsageError
from hyperflare_recipes.recipes.kimi3.examples import ExampleArena, declaration_tensor
from hyperflare_recipes.recipes.kimi3.vision import PATCH_CAPACITIES
from hyperflare_recipes.recipes.kimi31.recipe import RECIPE as KIMI31


def _write_exl3_dir(directory, *, weight_file_bytes=796_967_521_440, bad_manifest=False):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps({
        "architectures": ["KimiK31ForConditionalGeneration"],
        "text_config": {
            "quantization_config": {
                "quant_method": "exl3",
                "exl3": {
                    "manifest": "exl3-manifest.json",
                    "weight_file_bytes": weight_file_bytes,
                },
            },
            "hidden_size": 7168,
            "num_hidden_layers": 93,
            "num_attention_heads": 64,
            "q_lora_rank": 1536,
            "index_n_heads": 32,
            "index_head_dim": 128,
            "first_k_dense_replace": 1,
            "num_shared_experts": 1,
            "moe_intermediate_size": 1024,
            "linear_attn_config": {
                "num_heads": 64,
                "kda_layers": list(range(69)),
            },
        },
    }))
    (directory / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 300_000_000_000},
        "weight_map": {"model.embed_tokens.weight": "model-00001-of-00094.safetensors"},
    }))
    if bad_manifest:
        (directory / "exl3-manifest.json").write_text(json.dumps({"schema": "bad-v0"}))
    else:
        (directory / "exl3-manifest.json").write_text(json.dumps({
            "schema": "exl3-v1",
            "kind": "exl3-manifest",
            "layers": {"1": {"file": "exl3-experts-layer-001.safetensors"}},
        }))


def test_kimi31_exl3_replacement_inspection_and_csf_rejection(tmp_path):
    ckpt_dir = tmp_path / "kimi31-exl3"
    _write_exl3_dir(ckpt_dir)

    resolved = registry.resolve(str(ckpt_dir), recipes=(KIMI31,))
    assert resolved.recipe is KIMI31
    assert resolved.checkpoint.stored_bytes == 796_967_521_440
    assert resolved.checkpoint.scale_compressed_device_bytes == {}
    assert set(resolved.checkpoint.online_mxfp8_scope_savings_bytes) == {"kda-input", "kda-all", "all"}
    assert resolved.checkpoint.online_mxfp8_scope_savings_bytes["kda-all"] > resolved.checkpoint.online_mxfp8_scope_savings_bytes["kda-input"]

    gpus = tuple(
        planning.Gpu(i, "NVIDIA RTX PRO 6000 Blackwell", f"GPU-{i:04d}", 96 << 30, 95 << 30, (12, 0))
        for i in range(8)
    )
    # EXL3 checkpoints reject --expert-scale-compression csf cleanly.
    with pytest.raises(UsageError, match="--expert-scale-compression csf"):
        planning.fit(
            KIMI31, resolved.checkpoint, gpus, concurrency=4, reserve_bytes=0,
            speculation=Speculation(draft_tokens=7, method="dspark"),
            expert_scale_compression="csf",
        )

    # TP8/DCP8 EXL3 planning succeeds with kda-all MXFP8 and DSpark draft-body MXFP8.
    plan = planning.fit(
        KIMI31, resolved.checkpoint, gpus, concurrency=4, reserve_bytes=0,
        speculation=Speculation(draft_tokens=7, method="dspark", draft_body_quantization="mxfp8"),
        online_mxfp8=True, online_mxfp8_scope="kda-all",
        context_parallel=True, context_parallel_size=8,
    )
    assert plan.world_size == 8 and plan.pages > 0

    # Malformed EXL3 manifest raises UsageError rather than bare ValueError.
    bad_dir = tmp_path / "kimi31-exl3-bad"
    _write_exl3_dir(bad_dir, bad_manifest=True)
    with pytest.raises(UsageError, match="unsupported EXL3 expert manifest"):
        registry.resolve(str(bad_dir), recipes=(KIMI31,))


def test_example_arena_disjoint_namespaces_validation_and_fallback():
    arena = ExampleArena(
        "cpu",
        64 << 10,
        min_bytes=1024,
        namespace_bases={
            "vision": (0, 64 << 10),
            "common": (0, 32 << 10),
            "kda": (32 << 10, 64 << 10),
            "dsa": (32 << 10, 64 << 10),
        },
    )
    examples = {"__arena__": arena}

    v = declaration_tensor(examples, "vision.hidden", (1024,), torch.bfloat16, "cpu")
    c = declaration_tensor(examples, "misc.buf", (1024,), torch.bfloat16, "cpu")
    k = declaration_tensor(examples, "kda.q", (1024,), torch.bfloat16, "cpu")
    d = declaration_tensor(examples, "dsa.q", (1024,), torch.bfloat16, "cpu")

    # vision aliases common at offset 0; kda and dsa share the upper half disjoint from common.
    assert v.data_ptr() == arena.slab.data_ptr()
    assert c.data_ptr() == arena.slab.data_ptr()
    assert k.data_ptr() == arena.slab.data_ptr() + (32 << 10)
    assert d.data_ptr() == arena.slab.data_ptr() + (32 << 10)
    assert k.data_ptr() >= c.data_ptr() + c.numel() * c.element_size()

    # Allocations exceeding namespace span fall back cleanly to standalone tensors.
    overflow = declaration_tensor(examples, "kda.huge", (32 << 10,), torch.bfloat16, "cpu")
    assert not (arena.slab.data_ptr() <= overflow.data_ptr() < arena.slab.data_ptr() + arena.slab.nbytes)

    # Undeclared namespaces are rejected explicitly rather than silently aliasing offset 0.
    with pytest.raises(KeyError, match="undeclared ExampleArena namespace"):
        arena.allocate("unknown", (1024,), torch.bfloat16)

    arena.release()
    assert arena.slab is None
    with pytest.raises(RuntimeError, match="already been released"):
        arena.allocate("kda", (1024,), torch.bfloat16)


def test_vision_patch_capacities_include_full_65536():
    assert PATCH_CAPACITIES == (256, 1024, 4096, 16384, 65536)
