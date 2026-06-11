import atexit
import gc
import importlib
import importlib.util
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

# These are performance-only microbenchmarks for the DeepSeek-V4 DSA CP
# compressor path. Defaults are taken from ../prefill_0.log:
#   4k prefill: hidden_states=(4096, 4096), ratio-4 output=(1027, dim),
#               ratio-128 output=(33, 512)
#   8k prefill: hidden_states=(8192, 4096), ratio-4 output=(2049, dim),
#               ratio-128 output=(65, 512)
# Tune request shapes with:
#   DSV4_COMPRESSOR_SEQ_LENS=4096,8192
#   DSV4_COMPRESSOR_BATCH_SIZE=1
#   DSV4_COMPRESSOR_CACHE_SEGMENTS=1,2,4
# This file intentionally does not import vllm_ascend.utils.enable_custom_op.
# If the ops are not already registered, set DSV4_ASCEND_OPS_LIB to the
# vllm_ascend_C shared library path. If ACLNN cannot find Compressor regInfo,
# set DSV4_ASCEND_OPP_PATH to _cann_ops_custom/vendors/custom_transformer.
DTYPE = torch.bfloat16
HIDDEN_SIZE = int(os.getenv("DSV4_COMPRESSOR_HIDDEN_SIZE", "4096"))
ROPE_HEAD_DIM = int(os.getenv("DSV4_COMPRESSOR_ROPE_HEAD_DIM", "64"))
KV_CACHE_BLOCK_SIZE = int(os.getenv("DSV4_COMPRESSOR_KV_CACHE_BLOCK_SIZE", "128"))
BATCH_SIZE = int(os.getenv("DSV4_COMPRESSOR_BATCH_SIZE", "1"))
KV_CACHE_BLOCKS = int(os.getenv("DSV4_COMPRESSOR_KV_CACHE_BLOCKS", "8632"))
STATE_CACHE_BLOCKS = int(os.getenv("DSV4_COMPRESSOR_STATE_CACHE_BLOCKS", "8632"))
MAX_SEQ_LEN = int(os.getenv("DSV4_COMPRESSOR_MAX_SEQ_LEN", "1048576"))
WARMUP_ITERS = int(os.getenv("DSV4_COMPRESSOR_WARMUP_ITERS", "5"))
BENCH_ITERS = int(os.getenv("DSV4_COMPRESSOR_BENCH_ITERS", "20"))
RANDOM_LAYOUT_SEED = int(os.getenv("DSV4_COMPRESSOR_RANDOM_LAYOUT_SEED", "20260609"))
SLOT_MAPPING_PREVIEW_ROWS = int(os.getenv("DSV4_COMPRESSOR_SLOT_MAPPING_PREVIEW_ROWS", "8"))
REQUIRED_ASCEND_OPS = ("compressor", "npu_scatter_nd_update_v2")
CUSTOM_OP_VENDOR_DIR = "custom_transformer"


def _prepend_env_path(env_name: str, path: str) -> None:
    current_value = os.environ.get(env_name, "")
    path_entries = [entry for entry in current_value.split(":") if entry]
    if path not in path_entries:
        path_entries.insert(0, path)
        os.environ[env_name] = ":".join(path_entries)


def _candidate_package_dirs() -> list[Path]:
    package_dirs: list[Path] = []

    ops_lib = os.getenv("DSV4_ASCEND_OPS_LIB")
    if ops_lib:
        package_dirs.append(Path(ops_lib).resolve().parent)

    spec = importlib.util.find_spec("vllm_ascend")
    if spec and spec.submodule_search_locations:
        package_dirs.extend(Path(path).resolve() for path in spec.submodule_search_locations)

    return package_dirs


def _bootstrap_custom_opp_env() -> None:
    custom_opp_path = os.getenv("DSV4_ASCEND_OPP_PATH")
    if custom_opp_path:
        _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", custom_opp_path)
        vendor_lib_path = Path(custom_opp_path) / "op_api" / "lib"
        if vendor_lib_path.exists():
            _prepend_env_path("LD_LIBRARY_PATH", str(vendor_lib_path))
        return

    for package_dir in _candidate_package_dirs():
        vendor_path = package_dir / "_cann_ops_custom" / "vendors" / CUSTOM_OP_VENDOR_DIR
        if not vendor_path.exists():
            continue
        _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", str(vendor_path))
        vendor_lib_path = vendor_path / "op_api" / "lib"
        if vendor_lib_path.exists():
            _prepend_env_path("LD_LIBRARY_PATH", str(vendor_lib_path))
        return


def _required_ascend_ops_registered() -> bool:
    return all(hasattr(torch.ops._C_ascend, op_name) for op_name in REQUIRED_ASCEND_OPS)


def _ensure_ascend_ops_loaded() -> None:
    _bootstrap_custom_opp_env()

    if _required_ascend_ops_registered():
        return

    ops_lib = os.getenv("DSV4_ASCEND_OPS_LIB")
    if ops_lib:
        torch.ops.load_library(ops_lib)
    else:
        try:
            importlib.import_module("vllm_ascend.vllm_ascend_C")
        except ImportError as exc:
            pytest.skip(
                "DSV4 custom ops are not registered. Set DSV4_ASCEND_OPS_LIB to "
                "the vllm_ascend_C shared library path, or run in an installed "
                f"vllm-ascend environment. Import error: {exc}",
                allow_module_level=True,
            )

    if not _required_ascend_ops_registered():
        pytest.skip(
            f"Required _C_ascend ops are missing: {REQUIRED_ASCEND_OPS}",
            allow_module_level=True,
        )


_ensure_ascend_ops_loaded()


@dataclass(frozen=True)
class DSV4CompressorCase:
    name: str
    compress_ratio: int
    head_dim: int
    coff: int
    state_block_size: int
    scatter_dtype: torch.dtype = DTYPE


@dataclass(frozen=True)
class DSV4PrefillWorkload:
    name: str
    seq_len: int


@dataclass(frozen=True)
class DSV4CacheLayout:
    name: str
    num_segments: int
    random_blocks: bool = False


DSV4_PRO_COMPRESSOR_CASES = [
    DSV4CompressorCase("c4_dsa_compressor", compress_ratio=4, head_dim=512, coff=2, state_block_size=8),
    DSV4CompressorCase(
        "c4_indexer_compressor",
        compress_ratio=4,
        head_dim=128,
        coff=2,
        state_block_size=8,
        scatter_dtype=torch.int8,
    ),
    DSV4CompressorCase("c128_dsa_compressor", compress_ratio=128, head_dim=512, coff=1, state_block_size=32),
]


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _parse_int_list(env_name: str, default: str) -> list[int]:
    values = os.getenv(env_name, default)
    return [int(value.strip()) for value in values.split(",") if value.strip()]


def _parse_bool_env(env_name: str, default: bool) -> bool:
    value = os.getenv(env_name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _make_workloads() -> list[DSV4PrefillWorkload]:
    seq_lens = _parse_int_list("DSV4_COMPRESSOR_SEQ_LENS", "4096,8192")
    return [DSV4PrefillWorkload(f"seq{seq_len}_bs{BATCH_SIZE}", seq_len=seq_len) for seq_len in seq_lens]


DSV4_PRO_PREFILL_WORKLOADS = _make_workloads()


def _make_cache_layouts() -> list[DSV4CacheLayout]:
    segment_counts = _parse_int_list("DSV4_COMPRESSOR_CACHE_SEGMENTS", "1,2,4")
    layouts = [
        DSV4CacheLayout(
            name={1: "contiguous", 2: "two_segments", 4: "four_segments"}.get(
                num_segments, f"{num_segments}_segments"
            ),
            num_segments=num_segments,
        )
        for num_segments in segment_counts
    ]
    if _parse_bool_env("DSV4_COMPRESSOR_INCLUDE_RANDOM_LAYOUT", True):
        layouts.append(DSV4CacheLayout(name="random_blocks", num_segments=0, random_blocks=True))
    return layouts


DSV4_CACHE_LAYOUTS = _make_cache_layouts()
LAYOUT_PERF_RESULTS: list[dict[str, object]] = []


def _slot_mapping_summary(slot_mapping: torch.Tensor) -> dict[str, object]:
    slot_mapping_cpu = slot_mapping.detach().cpu()
    if slot_mapping_cpu.numel() == 0:
        return {
            "rows": 0,
            "num_blocks": 0,
            "block_ids": [],
            "linear_min": None,
            "linear_max": None,
            "head": [],
            "tail": [],
            "full": [],
        }

    block_ids = torch.unique(slot_mapping_cpu[:, 0], sorted=True).tolist()
    linear_indices = slot_mapping_cpu[:, 0].long() * KV_CACHE_BLOCK_SIZE + slot_mapping_cpu[:, 1].long()
    preview_rows = min(SLOT_MAPPING_PREVIEW_ROWS, slot_mapping_cpu.shape[0])
    full_slot_mapping = (
        slot_mapping_cpu.tolist()
        if _parse_bool_env("DSV4_COMPRESSOR_PRINT_FULL_SLOT_MAPPING", False)
        else None
    )
    return {
        "rows": slot_mapping_cpu.shape[0],
        "num_blocks": len(block_ids),
        "block_ids": block_ids,
        "linear_min": int(linear_indices.min().item()),
        "linear_max": int(linear_indices.max().item()),
        "head": slot_mapping_cpu[:preview_rows].tolist(),
        "tail": slot_mapping_cpu[-preview_rows:].tolist(),
        "full": full_slot_mapping,
    }


def _format_preview(values: list, max_items: int = 16) -> str:
    if len(values) <= max_items:
        return str(values)
    head_items = max_items // 2
    tail_items = max_items - head_items
    return f"{values[:head_items]} ... {values[-tail_items:]}"


def _format_slot_mapping_summary(summary: dict[str, object]) -> str:
    block_ids = summary["block_ids"]
    assert isinstance(block_ids, list)
    details = (
        f"rows={summary['rows']}, blocks={summary['num_blocks']}, "
        f"linear_range=[{summary['linear_min']}, {summary['linear_max']}], "
        f"block_ids={_format_preview(block_ids)}, "
        f"head={summary['head']}, tail={summary['tail']}"
    )
    full_slot_mapping = summary["full"]
    if full_slot_mapping is not None:
        details += f", full={full_slot_mapping}"
    return details


def _record_layout_perf(
    case: DSV4CompressorCase,
    workload: DSV4PrefillWorkload,
    cache_layout: DSV4CacheLayout,
    compressor_ms: float,
    scatter_ms: float,
    chain_ms: float,
    slot_mapping: torch.Tensor,
) -> None:
    LAYOUT_PERF_RESULTS.append({
        "case_name": case.name,
        "workload_name": workload.name,
        "layout_name": cache_layout.name,
        "compressor_ms": compressor_ms,
        "scatter_ms": scatter_ms,
        "chain_ms": chain_ms,
        "slot_mapping": _slot_mapping_summary(slot_mapping),
    })


def _print_layout_perf_summary() -> None:
    if not LAYOUT_PERF_RESULTS:
        return

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for result in LAYOUT_PERF_RESULTS:
        key = (str(result["case_name"]), str(result["workload_name"]))
        grouped.setdefault(key, []).append(result)

    layout_order = {layout.name: idx for idx, layout in enumerate(DSV4_CACHE_LAYOUTS)}
    print("\nDSV4 cache layout perf summary:")
    for case_name, workload_name in sorted(grouped):
        rows = sorted(
            grouped[(case_name, workload_name)],
            key=lambda row: layout_order.get(str(row["layout_name"]), 999),
        )
        baseline_chain_ms = next(
            (float(row["chain_ms"]) for row in rows if row["layout_name"] == "contiguous"),
            float(rows[0]["chain_ms"]),
        )
        print(f"\ncase={case_name}, workload={workload_name}")
        print("layout          compressor_ms  scatter_ms  chain_ms  chain_vs_contiguous")
        for row in rows:
            layout_name = str(row["layout_name"])
            compressor_ms = float(row["compressor_ms"])
            scatter_ms = float(row["scatter_ms"])
            chain_ms = float(row["chain_ms"])
            ratio = chain_ms / baseline_chain_ms if baseline_chain_ms > 0 else 0.0
            print(
                f"{layout_name:<15}"
                f"{compressor_ms:>13.3f}"
                f"{scatter_ms:>12.3f}"
                f"{chain_ms:>10.3f}"
                f"{ratio:>21.3f}x"
            )
        print("slot_mapping:")
        for row in rows:
            slot_mapping = row["slot_mapping"]
            assert isinstance(slot_mapping, dict)
            print(f"  {str(row['layout_name']):<15} {_format_slot_mapping_summary(slot_mapping)}")


atexit.register(_print_layout_perf_summary)


def _compressed_output_rows(num_tokens: int, batch_size: int, compress_ratio: int) -> int:
    return min(num_tokens, num_tokens // compress_ratio + batch_size)


def _balanced_segment_lengths(num_blocks: int, num_segments: int) -> list[int]:
    base = num_blocks // num_segments
    remain = num_blocks % num_segments
    return [base + (1 if idx < remain else 0) for idx in range(num_segments)]


def _make_random_block_ids(num_blocks: int, total_blocks: int) -> torch.Tensor:
    if num_blocks > total_blocks:
        raise ValueError(f"num_blocks={num_blocks} exceeds total_blocks={total_blocks}")
    if num_blocks == 0:
        return torch.empty((0,), dtype=torch.int32, device="npu")

    rng = random.Random(RANDOM_LAYOUT_SEED + num_blocks * 1009 + total_blocks)
    block_ids = rng.sample(range(total_blocks), num_blocks)
    return torch.tensor(block_ids, dtype=torch.int32, device="npu")


def _make_segmented_block_ids(num_blocks: int, total_blocks: int, cache_layout: DSV4CacheLayout) -> torch.Tensor:
    if num_blocks > total_blocks:
        raise ValueError(f"num_blocks={num_blocks} exceeds total_blocks={total_blocks}")
    if num_blocks == 0:
        return torch.empty((0,), dtype=torch.int32, device="npu")
    if cache_layout.random_blocks:
        return _make_random_block_ids(num_blocks, total_blocks)

    num_segments = min(cache_layout.num_segments, num_blocks)
    segment_lengths = _balanced_segment_lengths(num_blocks, num_segments)
    gap = 0 if num_segments == 1 else max((total_blocks - num_blocks) // (num_segments - 1), 1)

    block_ids: list[int] = []
    start = 0
    for segment_len in segment_lengths:
        block_ids.extend(range(start, start + segment_len))
        start += segment_len + gap

    if block_ids[-1] >= total_blocks:
        raise ValueError(
            f"cache layout {cache_layout.name} needs block id {block_ids[-1]}, "
            f"but total_blocks={total_blocks}"
        )
    return torch.tensor(block_ids, dtype=torch.int32, device="npu")


def _make_cu_seqlens(seq_lens: list[int]) -> torch.Tensor:
    values = [0]
    for seq_len in seq_lens:
        values.append(values[-1] + seq_len)
    return torch.tensor(values, dtype=torch.int32, device="npu")


def _make_state_block_table(
    seq_lens: list[int],
    max_seq_len: int,
    state_block_size: int,
    state_cache_blocks: int,
    cache_layout: DSV4CacheLayout,
) -> torch.Tensor:
    batch_size = len(seq_lens)
    max_blocks_per_batch = _ceil_div(max_seq_len, state_block_size)
    block_table = torch.zeros((batch_size, max_blocks_per_batch), dtype=torch.int32, device="npu")
    required_blocks = sum(_ceil_div(seq_len, state_block_size) for seq_len in seq_lens)
    block_ids = _make_segmented_block_ids(required_blocks, state_cache_blocks, cache_layout)
    block_offset = 0
    for batch_idx, seq_len in enumerate(seq_lens):
        num_blocks = _ceil_div(seq_len, state_block_size)
        block_table[batch_idx, :num_blocks] = block_ids[block_offset : block_offset + num_blocks]
        block_offset += num_blocks
    return block_table.contiguous()


def _make_slot_mapping(
    num_rows: int,
    cache_layout: DSV4CacheLayout,
    block_size: int = KV_CACHE_BLOCK_SIZE,
    total_blocks: int = KV_CACHE_BLOCKS,
) -> torch.Tensor:
    num_blocks = _ceil_div(num_rows, block_size)
    block_ids = _make_segmented_block_ids(num_blocks, total_blocks, cache_layout)
    positions = torch.arange(num_rows, dtype=torch.int32, device="npu")
    logical_block_ids = (positions // block_size).long()
    physical_block_ids = block_ids[logical_block_ids]
    return torch.stack([physical_block_ids, positions % block_size], dim=-1).contiguous()


def _make_update_tensor(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.int8:
        return torch.randint(-128, 127, shape, dtype=dtype, device="npu")
    return torch.randn(shape, dtype=dtype, device="npu")


def _make_compressor_inputs(
    case: DSV4CompressorCase,
    workload: DSV4PrefillWorkload,
    cache_layout: DSV4CacheLayout,
):
    torch.manual_seed(1024)
    seq_lens = [workload.seq_len] * BATCH_SIZE
    num_tokens = workload.seq_len * BATCH_SIZE
    compressed_rows = _compressed_output_rows(num_tokens, BATCH_SIZE, case.compress_ratio)
    state_dim = 2 * case.coff * case.head_dim
    required_state_blocks = sum(_ceil_div(seq_len, case.state_block_size) for seq_len in seq_lens)
    state_cache_blocks = max(STATE_CACHE_BLOCKS, required_state_blocks)
    state_block_table = _make_state_block_table(
        seq_lens,
        MAX_SEQ_LEN,
        case.state_block_size,
        state_cache_blocks,
        cache_layout,
    )

    hidden_states = torch.randn((num_tokens, HIDDEN_SIZE), dtype=DTYPE, device="npu")
    wkv = torch.randn((case.coff * case.head_dim, HIDDEN_SIZE), dtype=DTYPE, device="npu")
    wgate = torch.randn((case.coff * case.head_dim, HIDDEN_SIZE), dtype=DTYPE, device="npu")
    state_cache = torch.randn(
        (state_cache_blocks, case.state_block_size, state_dim), dtype=torch.float32, device="npu"
    )
    ape = torch.randn((case.compress_ratio, case.coff * case.head_dim), dtype=torch.float32, device="npu")
    norm_weight = torch.randn((case.head_dim,), dtype=DTYPE, device="npu")
    rope_sin = torch.randn((compressed_rows, ROPE_HEAD_DIM), dtype=torch.float32, device="npu")
    rope_cos = torch.randn((compressed_rows, ROPE_HEAD_DIM), dtype=torch.float32, device="npu")
    cu_seqlens = _make_cu_seqlens(seq_lens)
    start_pos = torch.zeros((BATCH_SIZE,), dtype=torch.int32, device="npu")

    return {
        "hidden_states": hidden_states,
        "wkv": wkv,
        "wgate": wgate,
        "state_cache": state_cache,
        "ape": ape,
        "norm_weight": norm_weight,
        "rope_sin": rope_sin,
        "rope_cos": rope_cos,
        "state_block_table": state_block_table,
        "cu_seqlens": cu_seqlens,
        "start_pos": start_pos,
        "expected_rows": compressed_rows,
    }


def _run_compressor(case: DSV4CompressorCase, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.ops._C_ascend.compressor(
        inputs["hidden_states"],
        inputs["wkv"],
        inputs["wgate"],
        inputs["state_cache"],
        inputs["ape"],
        inputs["norm_weight"],
        inputs["rope_sin"],
        inputs["rope_cos"],
        state_block_table=inputs["state_block_table"],
        cu_seqlens=inputs["cu_seqlens"],
        seqused=None,
        start_pos=inputs["start_pos"],
        rope_head_dim=ROPE_HEAD_DIM,
        cmp_ratio=case.compress_ratio,
        coff=case.coff,
        norm_eps=1e-6,
        rotary_mode=2,
        cache_mode=1,
    )


def _run_fused_compressor(
    case: DSV4CompressorCase,
    inputs: dict[str, torch.Tensor],
    slot_mapping: torch.Tensor,
    paged_kv_cache: torch.Tensor,
) -> torch.Tensor:
    return torch.ops._C_ascend.compressor(
        inputs["hidden_states"],
        inputs["wkv"],
        inputs["wgate"],
        inputs["state_cache"],
        inputs["ape"],
        inputs["norm_weight"],
        inputs["rope_sin"],
        inputs["rope_cos"],
        state_block_table=inputs["state_block_table"],
        cu_seqlens=inputs["cu_seqlens"],
        seqused=None,
        start_pos=inputs["start_pos"],
        slot_mapping=slot_mapping,
        paged_kv_cache=paged_kv_cache,
        rope_head_dim=ROPE_HEAD_DIM,
        cmp_ratio=case.compress_ratio,
        coff=case.coff,
        norm_eps=1e-6,
        rotary_mode=2,
        cache_mode=1,
        block_size=KV_CACHE_BLOCK_SIZE,
    )


def _benchmark_ms(fn, warmup_iters: int = WARMUP_ITERS, bench_iters: int = BENCH_ITERS) -> float:
    for _ in range(warmup_iters):
        fn()
    torch.npu.synchronize()

    start = time.perf_counter()
    for _ in range(bench_iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1000 / bench_iters


def _cleanup():
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize("workload", DSV4_PRO_PREFILL_WORKLOADS, ids=lambda workload: workload.name)
@pytest.mark.parametrize("cache_layout", DSV4_CACHE_LAYOUTS, ids=lambda cache_layout: cache_layout.name)
@pytest.mark.parametrize("case", DSV4_PRO_COMPRESSOR_CASES, ids=lambda case: case.name)
@torch.inference_mode()
def test_dsv4_pro_compressor_perf(
    case: DSV4CompressorCase,
    cache_layout: DSV4CacheLayout,
    workload: DSV4PrefillWorkload,
):
    inputs = _make_compressor_inputs(case, workload, cache_layout)
    compressed_kv = _run_compressor(case, inputs)
    torch.npu.synchronize()

    expected_shape = (inputs["expected_rows"], case.head_dim)
    assert tuple(compressed_kv.shape) == expected_shape

    elapsed_ms = _benchmark_ms(lambda: _run_compressor(case, inputs))
    print(
        f"\nDSV4 {case.name} compressor perf: {elapsed_ms:.3f} ms, "
        f"workload={workload.name}, layout={cache_layout.name}, "
        f"x={tuple(inputs['hidden_states'].shape)}, output={tuple(compressed_kv.shape)}, "
        f"state_cache={tuple(inputs['state_cache'].shape)}, "
        f"block_table={tuple(inputs['state_block_table'].shape)}"
    )
    _cleanup()


@pytest.mark.parametrize("workload", DSV4_PRO_PREFILL_WORKLOADS, ids=lambda workload: workload.name)
@pytest.mark.parametrize("cache_layout", DSV4_CACHE_LAYOUTS, ids=lambda cache_layout: cache_layout.name)
@pytest.mark.parametrize("case", DSV4_PRO_COMPRESSOR_CASES, ids=lambda case: case.name)
@torch.inference_mode()
def test_dsv4_pro_scatter_nd_update_v2_perf(
    case: DSV4CompressorCase,
    cache_layout: DSV4CacheLayout,
    workload: DSV4PrefillWorkload,
):
    update_rows = _compressed_output_rows(workload.seq_len * BATCH_SIZE, BATCH_SIZE, case.compress_ratio)
    compressed_kv = _make_update_tensor((update_rows, case.head_dim), case.scatter_dtype)
    cache = torch.zeros(
        (KV_CACHE_BLOCKS, KV_CACHE_BLOCK_SIZE, 1, case.head_dim), dtype=case.scatter_dtype, device="npu"
    )
    slot_mapping = _make_slot_mapping(update_rows, cache_layout)

    torch.ops._C_ascend.npu_scatter_nd_update_v2(cache, slot_mapping, compressed_kv)
    torch.npu.synchronize()

    elapsed_ms = _benchmark_ms(
        lambda: torch.ops._C_ascend.npu_scatter_nd_update_v2(cache, slot_mapping, compressed_kv)
    )
    print(
        f"\nDSV4 {case.name} scatter_nd_update_v2 perf: {elapsed_ms:.3f} ms, "
        f"workload={workload.name}, layout={cache_layout.name}, "
        f"cache={tuple(cache.shape)}, slot_mapping={tuple(slot_mapping.shape)}, "
        f"update={tuple(compressed_kv.shape)}"
    )
    _cleanup()


@pytest.mark.parametrize("workload", DSV4_PRO_PREFILL_WORKLOADS, ids=lambda workload: workload.name)
@pytest.mark.parametrize("cache_layout", DSV4_CACHE_LAYOUTS, ids=lambda cache_layout: cache_layout.name)
@pytest.mark.parametrize("case", DSV4_PRO_COMPRESSOR_CASES, ids=lambda case: case.name)
@torch.inference_mode()
def test_dsv4_pro_compressor_scatter_chain_perf(
    case: DSV4CompressorCase,
    cache_layout: DSV4CacheLayout,
    workload: DSV4PrefillWorkload,
):
    inputs = _make_compressor_inputs(case, workload, cache_layout)
    slot_mapping = _make_slot_mapping(inputs["expected_rows"], cache_layout)
    cache = torch.zeros((KV_CACHE_BLOCKS, KV_CACHE_BLOCK_SIZE, 1, case.head_dim), dtype=DTYPE, device="npu")

    def _chain():
        compressed_kv = _run_compressor(case, inputs)
        torch.ops._C_ascend.npu_scatter_nd_update_v2(cache, slot_mapping, compressed_kv)
        return compressed_kv

    compressed_kv = _chain()
    torch.npu.synchronize()
    assert tuple(compressed_kv.shape) == (inputs["expected_rows"], case.head_dim)

    compressor_ms = _benchmark_ms(lambda: _run_compressor(case, inputs))
    scatter_ms = _benchmark_ms(
        lambda: torch.ops._C_ascend.npu_scatter_nd_update_v2(cache, slot_mapping, compressed_kv)
    )
    chain_ms = _benchmark_ms(_chain)
    _record_layout_perf(case, workload, cache_layout, compressor_ms, scatter_ms, chain_ms, slot_mapping)
    print(
        f"\nDSV4 {case.name} compressor+scatter perf: {chain_ms:.3f} ms, "
        f"compressor={compressor_ms:.3f} ms, scatter={scatter_ms:.3f} ms, "
        f"workload={workload.name}, layout={cache_layout.name}, "
        f"x={tuple(inputs['hidden_states'].shape)}, output={tuple(compressed_kv.shape)}, "
        f"cache={tuple(cache.shape)}, slot_mapping={tuple(slot_mapping.shape)}"
    )
    _cleanup()


@pytest.mark.parametrize("workload", DSV4_PRO_PREFILL_WORKLOADS, ids=lambda workload: workload.name)
@pytest.mark.parametrize("cache_layout", DSV4_CACHE_LAYOUTS, ids=lambda cache_layout: cache_layout.name)
@pytest.mark.parametrize("case", DSV4_PRO_COMPRESSOR_CASES, ids=lambda case: case.name)
@torch.inference_mode()
def test_dsv4_pro_fused_compressor_perf(
    case: DSV4CompressorCase,
    cache_layout: DSV4CacheLayout,
    workload: DSV4PrefillWorkload,
):
    """Benchmark the fused compressor (OPT2): compressor writes directly to paged cache."""
    inputs = _make_compressor_inputs(case, workload, cache_layout)
    slot_mapping = _make_slot_mapping(inputs["expected_rows"], cache_layout)
    cache = torch.zeros((KV_CACHE_BLOCKS, KV_CACHE_BLOCK_SIZE, 1, case.head_dim), dtype=DTYPE, device="npu")

    compressed_kv = _run_fused_compressor(case, inputs, slot_mapping, cache)
    torch.npu.synchronize()
    assert tuple(compressed_kv.shape) == (inputs["expected_rows"], case.head_dim)

    fused_ms = _benchmark_ms(lambda: _run_fused_compressor(case, inputs, slot_mapping, cache))

    # baseline: chain (compressor + separate scatter)
    cache_baseline = torch.zeros_like(cache)

    def _chain():
        kv = _run_compressor(case, inputs)
        torch.ops._C_ascend.npu_scatter_nd_update_v2(cache_baseline, slot_mapping, kv)
        return kv

    chain_ms = _benchmark_ms(_chain)
    speedup = chain_ms / fused_ms if fused_ms > 0 else 0.0

    print(
        f"\nDSV4 {case.name} fused vs chain: "
        f"fused={fused_ms:.3f} ms, chain={chain_ms:.3f} ms, speedup={speedup:.2f}x, "
        f"workload={workload.name}, layout={cache_layout.name}, "
        f"x={tuple(inputs['hidden_states'].shape)}, output={tuple(compressed_kv.shape)}, "
        f"cache={tuple(cache.shape)}, slot_mapping={tuple(slot_mapping.shape)}"
    )
    _cleanup()


@pytest.mark.parametrize("workload", DSV4_PRO_PREFILL_WORKLOADS, ids=lambda workload: workload.name)
@pytest.mark.parametrize("cache_layout", DSV4_CACHE_LAYOUTS, ids=lambda cache_layout: cache_layout.name)
@pytest.mark.parametrize("case", DSV4_PRO_COMPRESSOR_CASES, ids=lambda case: case.name)
@torch.inference_mode()
def test_dsv4_pro_fused_compressor_correctness(
    case: DSV4CompressorCase,
    cache_layout: DSV4CacheLayout,
    workload: DSV4PrefillWorkload,
):
    """Verify fused compressor writes identical data to paged cache as chain."""
    if case.scatter_dtype != DTYPE:
        pytest.skip(f"correctness test only for {DTYPE} scatter, got {case.scatter_dtype}")

    inputs = _make_compressor_inputs(case, workload, cache_layout)
    slot_mapping = _make_slot_mapping(inputs["expected_rows"], cache_layout)

    # run chain: compressor + scatter
    cache_chain = torch.zeros((KV_CACHE_BLOCKS, KV_CACHE_BLOCK_SIZE, 1, case.head_dim), dtype=DTYPE, device="npu")
    compressed_kv = _run_compressor(case, inputs)
    torch.ops._C_ascend.npu_scatter_nd_update_v2(cache_chain, slot_mapping, compressed_kv)
    torch.npu.synchronize()

    # run fused: compressor with built-in scatter
    # reset state_cache to same initial values (compressor modifies it in-place)
    inputs_fused = _make_compressor_inputs(case, workload, cache_layout)
    cache_fused = torch.zeros((KV_CACHE_BLOCKS, KV_CACHE_BLOCK_SIZE, 1, case.head_dim), dtype=DTYPE, device="npu")
    _run_fused_compressor(case, inputs_fused, slot_mapping, cache_fused)
    torch.npu.synchronize()

    # compare: only check slots that were written
    slot_mapping_cpu = slot_mapping.cpu()
    for i in range(slot_mapping_cpu.shape[0]):
        block_idx = int(slot_mapping_cpu[i, 0].item())
        offset = int(slot_mapping_cpu[i, 1].item())
        chain_val = cache_chain[block_idx, offset, 0, :].cpu()
        fused_val = cache_fused[block_idx, offset, 0, :].cpu()
        if not torch.allclose(chain_val, fused_val, atol=1e-3, rtol=1e-3):
            pytest.fail(
                f"Mismatch at slot [{block_idx}, {offset}] (token {i}): "
                f"chain={chain_val[:8]}..., fused={fused_val[:8]}..."
            )

    print(
        f"\nDSV4 {case.name} fused correctness PASSED: "
        f"{slot_mapping_cpu.shape[0]} slots verified, "
        f"workload={workload.name}, layout={cache_layout.name}"
    )