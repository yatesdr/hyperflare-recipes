# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3 MXFP4-CSF preparation capacities and serving defaults."""

from dataclasses import replace
import json
from pathlib import Path
from types import MappingProxyType

from ignition.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults, SamplingRange, Speculation


def _exl3_quantization(metadata: Path) -> tuple[dict, dict] | None:
    config_path = metadata / "config.json"
    if not config_path.is_file():
        return None
    raw = json.loads(config_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("checkpoint configuration must be an object")
    text = raw.get("text_config", {})
    quant = text.get("quantization_config") if isinstance(text, dict) else None
    if not isinstance(quant, dict) or quant.get("quant_method") != "exl3":
        return None
    if not isinstance(quant.get("exl3"), dict) or quant["exl3"].get("manifest") != "exl3-manifest.json":
        raise ValueError("EXL3 requires exl3-manifest.json")
    return raw, quant


def _exl3_files(metadata: Path, quant: dict) -> tuple[str, ...]:
    manifest_name = quant["exl3"]["manifest"]
    manifest = json.loads((metadata / manifest_name).read_text())
    if (not isinstance(manifest, dict) or manifest.get("schema") != "exl3-v1"
            or manifest.get("kind") != "exl3-manifest"):
        raise ValueError("unsupported EXL3 expert manifest")
    layers = manifest.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise ValueError("EXL3 manifest has no expert layers")
    if not all(isinstance(ref, dict) for ref in layers.values()):
        raise ValueError("EXL3 layer references must be objects")
    files = tuple(ref.get("file") for ref in layers.values())
    if any(not isinstance(name, str) or not name.endswith(".safetensors") or Path(name).name != name
           for name in files):
        raise ValueError("EXL3 expert filenames must be local safetensors basenames")
    if len(set(files)) != len(files):
        raise ValueError("EXL3 expert layer files must be distinct")
    return files


def _exl3_stored_bytes(metadata: Path, quant: dict, base_stored_bytes: int) -> int:
    total = quant["exl3"].get("weight_file_bytes")
    if total is not None:
        if type(total) is not int or total <= base_stored_bytes:
            raise ValueError("EXL3 weight_file_bytes must be an integer larger than safetensors total_size")
        return total
    index = json.loads((metadata / "model.safetensors.index.json").read_text())
    files = set(index["weight_map"].values()) | set(_exl3_files(metadata, quant))
    return sum((metadata / name).stat().st_size for name in files)


def _exl3_mxfp8_savings(raw: dict) -> dict[str, int]:
    """Estimate Kimi's aligned KDA/attention/shared projection savings by scope for EXL3 checkpoints."""
    c = raw.get("text_config", {})
    if not isinstance(c, dict):
        return {}
    hidden, layers = c["hidden_size"], c["num_hidden_layers"]
    linear = c["linear_attn_config"]
    kda_columns = linear["num_heads"] * 128
    attention_columns = c["num_attention_heads"] * 128
    kda_layers = len(linear["kda_layers"])
    kda_input_elements = kda_layers * hidden * (4 * kda_columns + 128 + linear["num_heads"])
    kda_all_elements = kda_input_elements + kda_layers * hidden * kda_columns
    all_elements = hidden * (3 * kda_layers * kda_columns
        + 2 * (kda_layers * kda_columns + (layers - kda_layers) * attention_columns)
        + 3 * (layers - c["first_k_dense_replace"]) * c["num_shared_experts"] * c["moe_intermediate_size"])
    return {
        "kda-input": 2 * kda_input_elements * 7 // 16,
        "kda-all": 2 * kda_all_elements * 7 // 16,
        "all": 2 * all_elements * 3 // 8,
    }


def _replicated_dsa_indexer_bytes(raw: dict) -> int:
    """Per-rank BF16 byte size of replicated DSA indexer weights, or 0 for non-DSA checkpoints."""
    c = raw.get("text_config", {})
    if not isinstance(c, dict) or not c.get("index_n_heads"):
        return 0
    linear = c.get("linear_attn_config", {})
    kda_layers = len(linear.get("kda_layers", ())) if isinstance(linear, dict) else 0
    dsa_layers = max(0, int(c["num_hidden_layers"]) - kda_layers)
    heads, dim = int(c["index_n_heads"]), int(c["index_head_dim"])
    hidden, q_lora = int(c["hidden_size"]), int(c["q_lora_rank"])
    return dsa_layers * 2 * (heads * dim * q_lora + dim * hidden + heads * hidden + 2 * dim)


def _inspect_exl3_replacement(metadata: Path, replacement: Checkpoint) -> Checkpoint | None:
    """Size an unregistered Kimi EXL3 checkpoint and reject incompatible expert scale compression."""
    parsed = _exl3_quantization(metadata)
    if parsed is None:
        return None
    raw, quant = parsed
    manifest_path = metadata / quant["exl3"]["manifest"]
    if manifest_path.is_file():
        _exl3_files(metadata, quant)
    stored = _exl3_stored_bytes(metadata, quant, replacement.stored_bytes)
    savings = _exl3_mxfp8_savings(raw)
    # Indexer tensors are replicated on every rank without sharding; stored counts one copy.
    extra_replicated_indexer = 7 * _replicated_dsa_indexer_bytes(raw)
    return replace(
        replacement,
        stored_bytes=stored,
        device_bytes=stored + extra_replicated_indexer + replacement.draft_bytes,
        mapped_host_bytes=0,
        expansion_bytes=0,
        scale_compressed_device_bytes={},
        online_mxfp8_scope_savings_bytes=savings,
    )


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            core_storage=None, ranks=None, recipe=None, draft_source=None, draft_directory=None):
    from ignition.preparation.rank_group import prepare_group
    from .bootstrap import PreparationConfig, prepare_rank
    from .config import DRAFT_DIRECTORY, DSPARK_REPOSITORY, DSPARK_REVISION

    recipe = RECIPE if recipe is None else recipe
    if recipe.speculation is None and plan.speculation is not None:
        raise ValueError(f"{recipe.served_model_name} preparation does not support speculation")
    if ranks is not None:
        raise ValueError("Kimi K3 preparation requires GPUs on one host")
    drafting = {}
    if plan.speculation is not None:
        drafting = dict(speculative_widths=plan.speculation.widths, speculative_fixed_width=plan.speculation.fixed_width,
                        speculative_token_capacity=plan.speculative_token_capacity,
                        speculative_graph_budget=plan.speculation.graph_budget or 64,
                        draft_head_quantization=plan.speculation.draft_head_quantization,
                        draft_body_quantization=plan.speculation.draft_body_quantization,
                        markov_head_quantization=plan.speculation.markov_head_quantization,
                        speculative_method=plan.speculation.method)
        directory = DRAFT_DIRECTORY if draft_directory is None else draft_directory
        if (Path(snapshot) / directory).is_dir():
            drafting["draft_checkpoint"] = str(Path(snapshot) / directory)
        else:
            from huggingface_hub import snapshot_download

            repository, revision = (DSPARK_REPOSITORY, DSPARK_REVISION) if draft_source is None else draft_source
            drafting["draft_checkpoint"] = snapshot_download(repository, revision=revision, cache_dir=plan.hub_cache,
                                                             allow_patterns=("config.json", "model.safetensors"))
    config = PreparationConfig(mtp_only=plan.speculation is not None and plan.speculation.mtp_only,
        benchmark_selection=plan.benchmark_selection,
        prompt_logprobs=plan.prompt_logprobs,
        compile_workers=plan.compile_workers, autotune_effort=plan.autotune_effort, tuning_replica=plan.tuning_replica,
        native_dense_activations=plan.native_dense_activations,
        lock_gpu_clocks=plan.lock_gpu_clocks and plan.autotune_effort != "off",
        devices=tuple(range(plan.world_size)), context=plan.recipe.context,
        max_active=plan.concurrency, image_concurrency=plan.image_concurrency, prefill_chunk_size=plan.prefill_chunk,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes, compile_decoder=compile_forwards,
        retain_graphs=retain_graphs, kernel_cache=kernel_cache, online_mxfp8=plan.online_mxfp8, online_mxfp8_scope=plan.online_mxfp8_scope,
        stream_overlap=plan.stream_overlap, context_parallel=plan.context_parallel,
        context_parallel_size=plan.context_parallel_size or plan.world_size,
        vision=not plan.language_model_only, sampling_defaults=recipe.sampling_defaults.as_dict(),
        frontend_template=str(Path(__file__).parent.parent / recipe.name / "assets" / "chat_template.jinja"),
        **drafting)
    if plan.world_size == 1:
        return prepare_rank(snapshot, config, rank=0, group=None, progress=progress, core_storage=core_storage)
    return prepare_group(snapshot, config, prepare_rank, progress=progress, core_storage=core_storage,
                         checkpoint=plan.checkpoint)


def usage(prepared):
    report = prepared.memory_report
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in report.get("ranks", (report,)))


RECIPE = Recipe(
    name="kimi3", served_model_name="Kimi-K3", prepare=prepare, usage=usage,
    # Agentic defaults; top-p spans single-step/agentic presets: https://huggingface.co/moonshotai/Kimi-K3#3-evaluation-results.
    sampling_defaults=SamplingDefaults(temperature=1., top_p=1., top_k=-1,
        ranges={"top_p": SamplingRange(.95, 1.)}),
    checkpoints=(Checkpoint(alias="kimi-k3-mxfp4-csf", repository="/data/models/Kimi-K3-MXFP4-CSF",
        revision="local", architecture="KimiK3ForConditionalGeneration",
        stored_bytes=1_491_026_420_236, device_bytes=1_498_926_420_236, draft_bytes=7_900_000_000,
        scale_compressed_device_bytes={"csf": 1_498_926_420_236}),
        # The native checkpoint's block scales are compressed while loading, as resident as the CSF copy's.
        Checkpoint(alias="kimi-k3", repository="moonshotai/Kimi-K3",
        revision="2496450e92e425c886db095102a52a6682ca3970", architecture="KimiK3ForConditionalGeneration",
        stored_bytes=1_560_860_324_864, device_bytes=1_498_926_420_236, draft_bytes=7_900_000_000,
        scale_compressed_device_bytes={"csf": 1_498_926_420_236}),
        # A local target checkpoint carrying Red Hat's DSpark weights in dspark/.
        Checkpoint(alias="kimi-k3-dspark", repository="/data/kimi-k3-dspark", revision="local",
        architecture="KimiK3ForConditionalGeneration", stored_bytes=1_500_516_228_062,
        device_bytes=1_498_926_420_236, draft_bytes=7_900_000_000,
        scale_compressed_device_bytes={"csf": 1_498_926_420_236})),
    context=1048576, page_tokens=128, concurrency=32, max_concurrency=64, prefill_chunk=4096,
    speculation=Speculation(draft_tokens=7, token_capacity=128, packed_verification=True, graph_budget=64), max_draft_tokens=7, max_images=4, image_bytes=32 << 20,
    verification_rows=0,
    # KDA, MLA except batched key/value, shared experts, latent MoE projections, and the DSpark
    # draft's context projection, attention outputs and MLPs may load as MXFP8.
    online_mxfp8=True, online_mxfp8_scopes=("all", "kda-input", "kda-all"),
    inspect_replacement=_inspect_exl3_replacement,
    # Decode and verification run each MoE layer's shared experts beside its routing and routed experts.
    stream_overlap=True,
    # Text-only plans, as vLLM serves Kimi K3 with --language-model-only, leave the MoonViT encoder unloaded.
    language_model_only=True,
    # vLLM's Kimi K3 launch settings for b12x's PCIe all-reduce (island reduce-scatter, deferred slab
    # consumption, 256 threads), its decode context exchange and its small-M W4A16 experts.
    b12x_environment=MappingProxyType({"B12X_PCIE_ALLREDUCE_ALGORITHM": "island_rs",
        "B12X_PCIE_HIERARCHICAL_DEFERRED_CONSUMPTION": "1", "B12X_PCIE_HIERARCHICAL_THREADS": "256",
        "B12X_PCIE_DCP_THREADS": "512", "B12X_PCIE_DCP_BLOCK_LIMIT": "8", "B12X_W4A16_SMALL_M_HOST_BARRIER_RESET": "0"}),
    # Target MLA pages are context-parallel; DSpark keeps its sliding KV windows in continuation rows.
    ranks=MappingProxyType({16: RankCosts(state_row_bytes=34_716_928, page_bytes=1_769_472,
                                        graph_bytes=9 << 29, page_tokens=128 * 16)}),
    plain_ranks=MappingProxyType({16: RankCosts(state_row_bytes=29_869_824, page_bytes=1_769_472, graph_bytes=9 << 29,
                                              page_tokens=128 * 16)}),
)
