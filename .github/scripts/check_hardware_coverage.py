# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Reconcile each tutorial's declared `metadata.ci.hardware` against the hardware
its Prerequisites / Hardware section actually calls for.

The notebook regression runner routes a tutorial to a GPU machine purely from
`ci.hardware` (see the runner's tools/routing.py; this repo validates the same
vocabulary in check_ci_metadata.py). A wrong or too-narrow declaration sends a
notebook to the wrong machine, or to none, and a reader is told the wrong GPU.
`ci.hardware` is written by hand, so this gates PRs against a curated audit.

The source of truth is a human audit of every tutorial's Prerequisites /
Hardware section (and body GPU mentions):

    tutorial-gpu-requirements.md, compiled 2026-10-13.

That file is curated outside the repo, so its hardware column is baked in below
as SOURCE_OF_TRUTH — a dated snapshot, not a live read. Re-run the audit and
update this table when a tutorial's hardware section changes.

Usage:
    python3 .github/scripts/check_hardware_coverage.py        # reconcile every tutorial
    python3 .github/scripts/check_hardware_coverage.py --dir DIR
    python3 .github/scripts/check_hardware_coverage.py --json

Exit status is non-zero when any notebook has a hard disagreement (declared
hardware shares nothing with the tutorial's, a supported option is not declared,
or the declaration is not in the vocabulary), so it can gate CI.
"""

import argparse
import json
import sys
from pathlib import Path

from check_ci_metadata import HARDWARE, hardware_names

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_ROOT = REPO_ROOT / "docs" / "notebooks"

# ── Human hardware string -> the runner's canonical id ────────────────────────
# Keys are lowercased, whitespace-collapsed fragments of the tutorial's hardware
# text. Every value is a name hardware_names() understands.
HARDWARE_ALIASES = {
    "amd instinct mi300x": "mi300x", "instinct mi300x": "mi300x", "mi300x": "mi300x",
    "mi300": "mi300x",  # informal, no trailing X
    "mi308x": "mi308x",
    "amd instinct mi325x": "mi325x", "mi325x": "mi325x",
    "amd instinct mi350x": "mi350x", "mi350x": "mi350x",
    "amd instinct mi355x": "mi355x", "mi355x": "mi355x",
    "mi355": "mi355x",  # typo: missing the trailing X (mxfp4_quantization_quark_vllm)
    "amd instinct mi250x": "mi250x", "mi250x": "mi250x",
    "amd instinct mi210": "mi210", "mi210": "mi210",
    "amd instinct mi100": "mi100", "mi100": "mi100",
    "amd radeon pro w7900": "w7900", "radeon pro w7900": "w7900",
    "amd radeon w7900": "w7900", "radeon w7900": "w7900", "w7900": "w7900",
    "amd radeon rx 7900 xtx": "rx7900xtx", "radeon rx 7900 xtx": "rx7900xtx",
    "rx 7900 xtx": "rx7900xtx", "rx7900xtx": "rx7900xtx",
    "rx9070xt": "rx9070xt",
    "radeon 8060s": "radeon8060s",  # RDNA3.5 Strix Halo iGPU
    # architectures / families spoken directly
    "gfx942": "gfx942", "gfx950": "gfx950", "gfx90a": "gfx90a", "gfx908": "gfx908",
    "gfx1100": "gfx1100", "gfx1201": "gfx1201", "gfx1151": "gfx1151",
    "cdna1": "cdna1", "cdna2": "cdna2", "cdna3": "cdna3", "cdna4": "cdna4",
    "rdna3": "rdna3", "rdna4": "rdna4", "rdna35": "rdna35",
    "instinct": "instinct", "radeon": "radeon",
}

# Hardware a tutorial names that the vocabulary still cannot express. Empty now
# that MI100/MI210/MI250X and the RDNA3.5 iGPU have canonical ids; kept as the
# seam for any future GPU the audit names before the vocabulary catches up.
OUT_OF_VOCAB: dict[str, tuple[str, str]] = {}

# ── Source of truth: tutorial-gpu-requirements.md (2026-10-13 snapshot) ───────
# key: path under docs/notebooks (workshops/hackathons keys are repo-root).
# "hardware": the hardware the tutorial's Prerequisites/Hardware section names
# (human strings, resolved through the maps above).  "flag": the audit's
# inconsistency note, surfaced verbatim.  Notebooks the audit could not pin to
# hardware carry "hardware": [].
SOURCE_OF_TRUTH = [
    # Pre-training
    {"key": "pretrain/ddim_pretrain.ipynb", "hardware": ["MI300X"]},
    {"key": "pretrain/se3transform_intro.ipynb", "hardware": ["MI300X", "MI250X"],
     "flag": "code refs gfx940/941/942/90a/908 — broader CDNA than prereqs state"},
    {"key": "pretrain/setup_tutorial.ipynb", "hardware": ["MI300X"]},
    {"key": "pretrain/torchtitan_llama3.ipynb", "hardware": ["MI300X"]},
    {"key": "pretrain/torchtitan_deepseek.ipynb", "hardware": ["MI300X"],
     "flag": "image tag pins gfx942 (MI300X) — architecture-locked"},
    {"key": "pretrain/train_llama_mock_data.ipynb", "hardware": ["MI300X"]},
    {"key": "pretrain/training_with_primus.ipynb", "hardware": ["MI300X"],
     "flag": "image tag pins gfx942"},
    {"key": "pretrain/SpecForge_SGlang.ipynb", "hardware": ["MI300X"]},
    {"key": "pretrain/triton_inference_server_benchmark.ipynb", "hardware": ["MI300X", "MI325X"],
     "flag": "body also references MI355X — not in prerequisites"},
    # Fine-tuning
    {"key": "fine_tune/fine_tuning_lora_qwen2vl.ipynb", "hardware": ["MI300X"]},
    {"key": "fine_tune/llama_factory_llama3.ipynb", "hardware": ["MI300X"]},
    {"key": "fine_tune/qwen_image.ipynb", "hardware": ["MI300X"]},
    {"key": "fine_tune/torchtune_llama3.ipynb", "hardware": ["MI300X"]},
    {"key": "fine_tune/slime_qwen3_4B_GRPO.ipynb", "hardware": ["MI300X"]},
    {"key": "fine_tune/unsloth_Llama3_1_8B_GRPO.ipynb", "hardware": ["MI300X"]},
    # Inference
    {"key": "inference/1_inference_ver3_HF_transformers.ipynb", "hardware": ["MI300X"]},
    {"key": "inference/3_inference_ver3_HF_vllm.ipynb", "hardware": ["MI300X"]},
    {"key": "inference/SGlang_PD_Disagg_On_AMD_GPU.ipynb", "hardware": ["MI300X"]},
    {"key": "inference/agentic_ai_profiling.ipynb", "hardware": ["MI300X"]},
    {"key": "inference/build_airbnb_agent_mcp.ipynb", "hardware": ["MI300X"]},
    {"key": "inference/deepseek_janus_cpu_gpu.ipynb", "hardware": ["MI100", "MI210", "MI300X"]},
    {"key": "inference/deepseekr1_sglang.ipynb", "hardware": ["MI300X"]},
    {"key": "inference/graph_multiagent_openclaw.ipynb", "hardware": ["MI300X"]},
    {"key": "inference/mtp.ipynb", "hardware": ["MI300X"]},
    {"key": "inference/ocr_vllm.ipynb", "hardware": ["MI300X"]},
    {"key": "inference/opea_deployment_and_evaluation.ipynb", "hardware": ["MI300X"]},
    {"key": "inference/openclaw_vllm.ipynb", "hardware": ["MI300X"],
     "flag": "no prerequisites section — hardware inferred"},
    {"key": "inference/power-Google-ADK-on-AMD-platform-and-local-LLMs.ipynb",
     "hardware": ["MI100", "MI210", "MI300X"]},
    {"key": "inference/rag_ollama_llamaindex.ipynb", "hardware": ["Radeon PRO W7900"]},
    {"key": "inference/rapbot_vllm.ipynb", "hardware": ["MI300X"]},
    {"key": "inference/speculative_decoding_deep_dive.ipynb", "hardware": ["MI300X"]},
    {"key": "inference/t2v_comfyui_api_mode_instinct.ipynb", "hardware": ["MI300X", "MI355X"],
     "flag": "body comparison table also lists MI325X and MI350X — not in prerequisites"},
    {"key": "inference/t2v_comfyui_radeon.ipynb", "hardware": ["Radeon RX 7900 XTX"]},
    {"key": "inference/vllm_v1_DSR1.ipynb", "hardware": ["MI300X"]},
    {"key": "inference/voice_pipeline_rag_ollama.ipynb", "hardware": ["MI300X", "Radeon W7900"]},
    # GPU dev & optimize
    {"key": "gpu_dev_optimize/aiter_mla_decode_kernel.ipynb", "hardware": ["MI300X"],
     "flag": "no formal hardware section — GPU named only in body prose"},
    {"key": "gpu_dev_optimize/fp8_gemm_hip_cdna4.ipynb", "hardware": ["MI350X", "MI355X"]},
    {"key": "gpu_dev_optimize/fp8_quantization_quark_vllm.ipynb", "hardware": ["MI355X"],
     "flag": "no hardware section; notebook text says only 'Instinct' — MI355X is inferred"},
    {"key": "gpu_dev_optimize/helion_gpu_kernel_dev.ipynb", "hardware": ["MI300X"],
     "flag": "code refs gfx940/941/942/90a/908 — broader CDNA implied"},
    {"key": "gpu_dev_optimize/llama4_profiling_vllm.ipynb", "hardware": ["MI300X", "Radeon Pro W7900"]},
    {"key": "gpu_dev_optimize/mxfp4_quantization_quark_vllm.ipynb", "hardware": ["MI355"],
     "flag": "prerequisite label reads 'MI355' (missing X) — typo in the notebook"},
    {"key": "gpu_dev_optimize/triton_kernel_dev.ipynb", "hardware": ["MI300X"],
     "flag": "code refs gfx940/941/942/90a/908 — broader CDNA implied"},
    # Workshops / Hackathons — repo-root paths, not under docs/notebooks, not CI-tracked
    {"key": "workshops/airbnb_agent/build_airbnb_agent_mcp.ipynb", "hardware": ["MI300X"],
     "ci_tracked": False},
    {"key": "workshops/grpo_game2048-finetune-unsloth/gpt_oss_20B_Reinforcement_Learning_2048_Game_BF16.ipynb",
     "hardware": [], "ci_tracked": False,
     "flag": "no hardware section; code comment notes '64GB GPU or more (MI300)'"},
    {"key": "workshops/llama-3.2-vision-finetune-hf/tutorial_hf.ipynb", "hardware": ["MI300X", "MI325X"],
     "ci_tracked": False},
    {"key": "workshops/llama-3.2-vision-finetune-unsloth/tutorial_unsloth.ipynb", "hardware": ["MI300X"],
     "ci_tracked": False},
    {"key": "workshops/multi-agent-nutrionist/multi_agent_nutrition.ipynb", "hardware": [],
     "ci_tracked": False, "flag": "no hardware section; no GPU mentions anywhere in notebook"},
    {"key": "workshops/unsloth-gemma-4-finetuning-ryzen/Gemma4_(E2B)_Reinforcement_Learning_Sudoku_Game.ipynb",
     "hardware": ["Radeon 8060S"], "ci_tracked": False,
     "flag": "no hardware section; execution output reveals Radeon 8060S (gfx1151, RDNA3.5 laptop iGPU)"},
    {"key": "hackathons/robotics_2025/training-models-on-rocm.ipynb", "hardware": ["MI300X"],
     "ci_tracked": False},
]


def normalize(human):
    """(canonical ids, [(human, arch, why) out of vocab]) for a hardware list."""
    canonical = set()
    oov = []
    for name in human:
        key = " ".join(name.lower().split())
        if key in HARDWARE_ALIASES:
            canonical.add(HARDWARE_ALIASES[key])
        elif key in OUT_OF_VOCAB:
            arch, why = OUT_OF_VOCAB[key]
            oov.append((name, arch, why))
        else:
            oov.append((name, "?", "unrecognized hardware string"))
    return canonical, oov


def declared_hardware(nb_path):
    """The notebook's ci.hardware (lowercased), or None if it can't be read."""
    try:
        nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
        ci = nb.get("metadata", {}).get("ci") or {}
        return [h.lower() for h in ci.get("hardware", ["instinct"])]
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def path_for(record, notebook_root):
    """Where a source-of-truth key lives: ci-tracked under docs/notebooks, the
    rest at the repo root (workshops / hackathons)."""
    if record.get("ci_tracked", True):
        return notebook_root / record["key"]
    return REPO_ROOT / record["key"]


def reconcile(record, notebook_root):
    """Compare one source-of-truth row against the notebook's declaration."""
    key = record["key"]
    expected, oov = normalize(record.get("hardware", []))
    out = {"key": key, "expected": sorted(expected), "declared": None,
           "status": "ok", "issues": [], "notes": []}
    if record.get("flag"):
        out["notes"].append(record["flag"])
    for human, arch, why in oov:
        out["notes"].append(f"out_of_vocab: {human} ({arch}) — {why}")

    # Workshops / hackathons live outside docs/notebooks: they carry no ci
    # metadata and the runner never routes them, so there is nothing to
    # reconcile — whether or not the file is present in this checkout.
    if not record.get("ci_tracked", True):
        out["status"] = "not_ci_tracked"
        out["notes"].append("not under docs/notebooks — carries no ci metadata, not routed")
        return out

    path = path_for(record, notebook_root)
    if not path.exists():
        out["status"] = "missing_file"
        out["issues"].append(f"source-of-truth notebook not found at {path}")
        return out

    declared = declared_hardware(path)
    if declared is None:
        out["status"] = "unreadable"
        out["issues"].append("could not read metadata.ci")
        return out
    out["declared"] = declared

    unknown = [d for d in declared if d not in HARDWARE]
    if unknown:
        out["status"] = "invalid_declaration"
        out["issues"].append(f"declared hardware not in the vocabulary: {', '.join(unknown)}")

    if not expected:
        # Nothing authoritative to check against (missing/informal in the audit).
        if out["status"] == "ok":
            out["status"] = "no_source_hardware"
        out["notes"].append("source of truth states no routable hardware for this notebook")
        return out

    declared_names = set().union(*(hardware_names(d) for d in declared)) if declared else set()
    expected_names = {p: hardware_names(p) for p in expected}

    # A supported hardware option that the declaration cannot route to.
    uncovered = [p for p, names in expected_names.items() if not (names & declared_names)]
    all_expected_names = set().union(*expected_names.values())
    if not (declared_names & all_expected_names):
        out["status"] = "no_match" if out["status"] == "ok" else out["status"]
        out["issues"].append(
            f"declared {declared} shares no hardware with tutorial {sorted(expected)}")
    elif uncovered:
        out["status"] = "under_declared" if out["status"] == "ok" else out["status"]
        out["issues"].append(
            f"tutorial also supports {uncovered} but declaration {declared} cannot route there")
    return out


HARD_STATUSES = {"no_match", "under_declared", "invalid_declaration",
                 "missing_file", "unreadable"}


def report(notebook_root):
    return [reconcile(r, notebook_root) for r in SOURCE_OF_TRUTH]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=str(NOTEBOOK_ROOT),
                   help="Notebook root to read ci.hardware from (default: %(default)s).")
    p.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    args = p.parse_args()
    notebook_root = Path(args.dir)
    rows = report(notebook_root)

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for r in rows:
            decl = ",".join(r["declared"]) if r["declared"] else "—"
            exp = ",".join(r["expected"]) if r["expected"] else "—"
            print(f"[{r['status']:19}] {r['key']}")
            print(f"{'':22}tutorial={exp}  declared={decl}")
            for issue in r["issues"]:
                print(f"{'':22}! {issue}")
            for note in r["notes"]:
                print(f"{'':22}· {note}")

    hard = [r for r in rows if r["status"] in HARD_STATUSES]
    print(f"\n{len(rows)} notebooks, {len(hard)} with disagreements CI should act on.",
          file=sys.stderr)
    return 1 if hard else 0


if __name__ == "__main__":
    raise SystemExit(main())
