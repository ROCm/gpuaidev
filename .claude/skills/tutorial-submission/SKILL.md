---
name: tutorial-submission
description: Guides authors through contributing a new Jupyter Notebook tutorial to the AMD ROCm AI Developer Hub (gpuaidev-internal). Use when the user wants to submit, add, contribute, or prepare a new tutorial notebook, wire it into the README and Sphinx TOC, create the tutorial branch, or open the contribution PR. Enforces the eligibility checklist, required notebook structure, correct category placement, README table row, docs/sphinx/_toc.yml.in entry, and PR/review conventions.
---

# Tutorial Submission Skill

Use this skill to take an author from a draft notebook to a correctly wired, review-ready Pull Request for the AMD AI Developer Hub. Follow the steps in order. Do not skip the eligibility gate.

If the user invoked this skill with arguments, treat them as the tutorial path, title, or topic: $ARGUMENTS

## What you can and cannot verify

You CAN verify and edit repository wiring and notebook structure (headers, sections, category placement, README row, TOC entry, JSON validity, keyword scans).

You CANNOT execute the tutorial's code cells. Tutorial cells target specific AMD GPU hardware (Instinct MI300X, Radeon, CDNA4, etc.) that you do not have access to. End-to-end execution on AMD hardware is the author's responsibility and must be attested by the author, never claimed by you.

## Step 0: Identify the notebook and category

1. Confirm the tutorial is a single `.ipynb` file. If supplementary scripts/data are needed, they must be downloaded or generated from within the notebook, and explained in markdown cells. Multiple hand-assembled source files are not allowed.
2. Determine the content category. It MUST be exactly one of these, which map to real directories:
   - Inference -> `docs/notebooks/inference/`
   - Fine-tuning -> `docs/notebooks/fine_tune/`
   - Training / Pretraining -> `docs/notebooks/pretrain/`
   - GPU Development and Optimization -> `docs/notebooks/gpu_dev_optimize/`
   If the category is ambiguous, ask the user before proceeding.
3. Choose a descriptive `snake_case` filename ending in `.ipynb`. Place the file in the matching category directory above.
4. Images and static assets: if the notebook embeds images (architecture diagrams, screenshots, GIFs), place the files under `docs/notebooks/assets/` and reference them from markdown with a relative path `![alt](../assets/<file>)` (this is the repo convention: ~48 asset files, referenced via `../assets/`). These asset files MUST be committed as part of the PR. Do not hotlink external images for core diagrams.

## Step 1: Eligibility and content-alignment gate

Confirm ALL of the following before wiring the tutorial. If any fails, stop and tell the author what to fix.

- Single Jupyter Notebook; supplementary files fetched/generated inside the notebook and explained (what they contain, why needed, how used).
- Built for AMD GPUs using the ROCm stack; hardware/software environment clearly identified. Target hardware may be AMD Instinct AND/OR AMD Radeon depending on the tutorial (e.g., Radeon-targeted tutorials like ComfyUI-on-Radeon are valid); the Prerequisites must name the intended family.
- Recent, relevant, or emerging AI topic with practical value.
- Educational, step-by-step format: small focused code cells with markdown explaining what/why/expected output. Not one giant code dump.
- Uses upstream open-source projects, NOT AMD-specific forks, when upstream supports ROCm. Flag any `git clone` / `pip install` pointing at an AMD/ROCm fork of a mainstream project.
- Uses open-source, permissively licensed frameworks/libraries/models/datasets where possible, and links to the upstream project/model/dataset/docs so readers can learn more.
- Overlap check: confirm the topic is unique or offers a meaningfully different perspective/workflow/technology/depth vs the existing tutorials in the same category (see the README table and `docs/notebooks/<category>/`). If a related tutorial exists, state what this one adds.
- No unapproved performance claims. The concern is legal: any performance claim about AMD hardware on the official AMD page requires legal approval. Apply this distinction:
  - NOT allowed (blocking): specific/absolute performance figures presented as AMD hardware capability (e.g., "MI300X reaches X tokens/sec", "achieves Y ms latency", "Z% faster on AMD"), and any competitive comparison against other vendors/hardware (e.g., vs NVIDIA, A100/H100/H200). These are marketing performance claims that need legal review.
  - Allowed (educational): a pedagogical baseline-then-improve narrative within the tutorial, e.g., "start from this baseline, apply the optimization in this exercise, and observe the relative improvement." Relative, self-contained, tutorial-internal comparisons that teach a technique are fine, as long as they do not assert an absolute AMD hardware performance number or a cross-vendor comparison.
  - Naturally emitted framework timings are fine as long as they are not framed as an absolute AMD performance claim or a competitive comparison.
- Fits at least one AI Developer Hub category (see Step 0).
- Author attests the notebook was run end-to-end on the documented AMD GPU configuration with no unresolved errors (you cannot verify this; ask the author to confirm).

## Step 2: Verify required notebook structure

Read the notebook (parse the JSON) and confirm the conventions used by the published tutorials in this repo. These were derived from all 40 currently published notebooks; match them so the new tutorial is consistent with the hub.

### 2.1 Header block (first markdown cell)
1. Starts with a single `# H1` title (38/40 tutorials).
2. Immediately followed by:
   - `**Author**: <name>`
   - `**Knowledge level**: <Beginner|Intermediate|Advanced>` (100% of tutorials; use exactly one of these three values).
3. Introduction: goal of the tutorial, the technology/framework/model, the workflow demonstrated, and what the reader will learn.
4. Optional but common in newer tutorials: an in-notebook table of contents using anchor links (e.g., `[Environment setup](#env-setup)`) backed by `<a id="env-setup"></a>` anchors before the corresponding sections.

### 2.2 Prerequisites (required, `## Prerequisites` - present in 38/40)
Use the canonical sub-structure the hub standardizes on. Include a line like "This tutorial was developed and tested using the following setup." then:
- `### Operating system` - e.g., Ubuntu 22.04 / 24.04.
- `### Hardware` - AMD GPU model or family (AMD Instinct e.g. "AMD Instinct MI300X GPU", or AMD Radeon for Radeon-targeted tutorials) AND the explicit number of GPUs required. State minimum vs optional configs if multiple are supported. Note required GPU memory / system memory / storage where relevant. Link to the ROCm [system requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) (35/40 do this).
- `### Software` - ROCm version and a verify step (`amd-smi`, noting `rocm-smi` for ROCm 6.4 and earlier), plus Docker with the standard permission steps (`sudo usermod -aG docker $USER` / `newgrp docker` / `docker run hello-world`).
- `### Hugging Face API access` - ONLY when the tutorial uses gated/hosted models. State that the reader must obtain a token and have approval/permission for the specific checkpoints (18/40 include this).

Missing GPU model or missing GPU count in Prerequisites is a blocking gap.

### 2.3 Environment setup and Jupyter launch (required)
Tutorials must be runnable in Jupyter format by following the notebook alone. The hub's strong convention (33/40 use `docker run`; 39/40 include Jupyter launch steps) is a Docker-based, containerized setup. Include an environment setup section (commonly `## Environment setup with Docker and ROCm` or `## Prepare the <inference|training> environment`) that walks through:
1. Pull the Docker image - a ROCm-based image (e.g., `rocm/pytorch`, `rocm/pytorch-training`, `rocm/vllm-dev`, `lmsysorg/sglang:*-rocm*`, `vllm/vllm-openai-rocm`). Prefer a pinned tag over `latest` for reproducibility.
2. Launch the container with the canonical device/flag block:
   ```bash
   docker run -it --rm \
     --network=host \
     --device=/dev/kfd \
     --device=/dev/dri \
     --group-add=video \
     --ipc=host \
     --cap-add=SYS_PTRACE \
     --security-opt seccomp=unconfined \
     --shm-size 8G \
     -v $(pwd):/workspace \
     -w /workspace \
     <rocm-image>
   ```
3. Install and launch Jupyter inside the container:
   ```bash
   pip install jupyter
   jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
   ```
   (36/40 use this exact `jupyter-lab` invocation.)
4. Install the tutorial's required libraries (the notebook's `pip install` cells run INSIDE the container).
5. Provide the HF token when needed.

An acceptable alternative to Docker is an explicit isolated environment (conda or venv on top of a documented ROCm install), but a bare host-OS "just pip install" setup with no container/env isolation is NOT consistent with the hub and should be avoided. If not using Docker, justify why and still document a reproducible setup.

### 2.4 Models, datasets, and downloads
- Models, datasets, and any supplementary code/config must be fetched or generated FROM WITHIN the notebook, consistent with the single-notebook rule. The repo does this via Hugging Face (`snapshot_download` / `hf_hub_download` / `load_dataset`, ~20 tutorials), `wget`/`curl` (~13), or `git clone` for a supporting repo. Each download cell should be preceded by markdown explaining what is being downloaded and why.
- For gated models, the notebook should include the HF login/token step (`huggingface-cli login`, `notebook_login()`, or reading a token env var) and the Prerequisites must mention the required access.
- Prefer pinning a model revision/dataset version where practical for reproducibility.

### 2.5 GPU count consistency (critical for multi-GPU tutorials)
The number of GPUs actually used by the code MUST match the "GPUs required" stated in Prerequisites. Cross-check the mechanisms the repo uses to set parallelism:
- vLLM/SGLang: `tensor_parallel_size=N` or `--tp N`
- Distributed launchers: `nproc_per_node N`, `world_size`
- Device selection: `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES`
If Prerequisites says "8 GPUs" the launch commands should reflect 8 (e.g., `--tp 8` / `nproc_per_node 8`). A mismatch is a blocking issue. Mention GPU memory needs where relevant (e.g., model footprint vs per-GPU VRAM).

### 2.6 Body, outputs, and endings
- Step-by-step content: small, focused code cells interleaved with markdown explaining what/why/expected output. Use `**Note**:` callouts for caveats and permissions (near-universal in the repo).
- One logical task per code cell where practical. Prefer separating: imports, configuration, model loading, data preparation, inference/training setup, execution, and result inspection. Split large or hard-to-explain cells into smaller logical steps.
- Introduce a concept in markdown BEFORE showing the code that implements it.
- Explain important outputs: for cells that produce meaningful results (model responses, training progress, tensor shapes, GPU detection, generated files, validation results), add markdown telling the reader what to expect (the repo commonly uses "you should see" / "expected output" phrasing).
- Screenshots may be used when they meaningfully improve the tutorial, but prefer a textual explanation when the information can be represented clearly in text.
- Link to the upstream project, model, dataset, and relevant docs so readers can learn more.
- Recommended ending: a short Summary, Conclusion, Next steps, or Further reading section (varies across tutorials; not mandatory but valued).
- Do NOT add an in-notebook license footer or legal disclaimer; licensing is handled centrally via `docs/notebooks/licensing.md` (only 2/40 mention license, 0/40 have disclaimers).

### 2.7 Reproducible dependencies
Where practical, make the tutorial reproducible and easy to maintain: pin key `pip install` package versions, document the ROCm version validated against, identify the AMD GPU model/family and GPU count, pin the model revision/dataset version, and identify the Docker image (pinned tag). Avoid unnecessary dependencies.

Notebook metadata should be valid: `nbformat: 4`, a python3 kernelspec. Prefer clearing noisy execution counts, but keep meaningful outputs that help readers.

## Step 3: Wire the tutorial into the repo

Make the following edits precisely. Match existing indentation and formatting exactly. The three mandatory edits are the notebook file, the README row (3a), and the TOC entry (3b); also commit any assets (3d).

### 3a. README.md table row

Add a row under the correct category section in the root `README.md` "Current Notebooks" table. The table has 4 columns: Category, Title, GitHub Link, AMD Tutorial Page. Only the first row of a category shows the bold category label; continuation rows leave the first column blank.

- GitHub link points to the PUBLIC mirror: `https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/<category>/<file>.ipynb`
- AMD Tutorial link: `https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/<category>/<file>.html`

Example continuation row (Inference):
```
|                          | <Your tutorial title>        | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/inference/<file>.ipynb) | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/<file>.html) |
```

### 3b. docs/sphinx/_toc.yml.in entry

Add an entry under the matching `- caption:` block. Use 2-space indentation, `- file:` then `title:`. Path is relative to `docs/` (no leading `./`).

```yaml
  - caption: Inference tutorials
    entries:
    ...
    - file: notebooks/inference/<file>.ipynb
      title: <Your tutorial title>
```

Caption-to-directory mapping:
- `Inference tutorials` -> `notebooks/inference/`
- `Fine-tuning tutorials` -> `notebooks/fine_tune/`
- `Pretraining tutorials` -> `notebooks/pretrain/`
- `GPU development and optimization tutorials` -> `notebooks/gpu_dev_optimize/`

### 3c. Consistency

The title used in the README row and the TOC entry should match. The file path in README, TOC, and the actual file on disk must be identical.

### 3d. Assets

If the notebook references images via `../assets/<file>`, ensure each referenced file exists under `docs/notebooks/assets/` and is staged in the same commit. Every `![...](../assets/...)` reference must resolve to a committed file.

## Step 4: Pre-submission verification (author-attested)

Remind the author to, on the documented AMD GPU hardware, achieve error-free execution:
1. Restart the Jupyter kernel.
2. Run all cells from the beginning, top to bottom, in order.
3. Confirm no unresolved runtime errors and no hidden dependencies on previously executed cells (nothing relies on out-of-order state).
4. Confirm all required packages and files can be obtained using ONLY the documented setup (no undocumented local configuration).
5. Verify the workflow using the documented number of GPUs (for multi-GPU tutorials, the documented minimum count).

A developer following only the documented prerequisites should be able to reproduce the tutorial. You cannot perform these steps (no AMD hardware access); collect the author's explicit confirmation.

Documentation build compatibility: the notebook is rendered by Sphinx/nbsphinx for the docs site. Keep markdown/anchors valid and asset paths correct so the documentation build does not break. If a docs build is available locally, building it is a safe non-hardware check; otherwise flag build compatibility as something the docs team validates.

## Step 5: Notify the tutorial team

Before beginning the repo contribution process, the author emails Tanina Obasi and Mahdi Ghodsi with subject `New Tutorial - [Title]` including: title, brief description, author name(s), group/org, intended AMD GPU hardware, GPU count, and content category (one or more of Inference / Fine-tuning / Training / GPU Development and Optimization).

For status and planned release schedule, the author should review the AI Tutorials Tracker.

## Step 6: Branch and PR

Target repository: `AMD-ROCm-Internal/gpuaidev-internal`.

Prerequisite - GitHub org access: the author must have access to the AMD-ROCm-Internal GitHub organization before they can access the repo, push a branch, or open a PR. Apply for access EARLY via the internal "GitHub 101: Getting Access and Support" instructions: https://amd.atlassian.net/wiki/spaces/LC/pages/531791921/GitHub+101+Getting+Access+and+Support

1. Clone the internal repo: `https://github.com/AMD-ROCm-Internal/gpuaidev-internal`.
2. Create a branch from `main` named `tutorial/<short-descriptor>`.
3. Add the `.ipynb` to the correct category directory. If unsure which directory, contact the repository maintainers before opening the PR.
4. Commit the new `.ipynb`, the README.md change, the `_toc.yml.in` change, and any assets together.
5. Push the branch to `AMD-ROCm-Internal/gpuaidev-internal`.
6. Open a PR against `main`.
7. Add `Mahdi-CV` as a required reviewer (matches `.github/CODEOWNERS`).

Suggested commit message style (no AI attribution): `Add <category> tutorial: <title>`.

What happens after approval (inform the author): PR approved -> merged to internal `main` -> ROCm documentation team performs a final documentation review -> tutorial prepared for publication -> internal content synchronized and mirrored to the appropriate public repository (`ROCm/gpuaidev`).

## Step 7: Post-publication maintenance (set author expectations)

Tell the author they are the primary maintainer of the tutorial after publication and are responsible for:
- Regular maintenance: updating the tutorial when ROCm releases, Python packages, AI frameworks, models, APIs, Docker images, or other upstream dependencies change.
- Regression testing: re-running the complete notebook on the documented AMD GPU configuration (documented minimum GPU count for multi-GPU tutorials) when significant dependency/API changes occur; automated testing may flag failures but the author investigates and fixes.
- User support: helping address technical questions/issues raised through the repo or docs platform, and folding genuine problems into future updates.

## Final report to the author

Summarize: category and file path chosen, structural checks passed/failed, overlap finding vs existing tutorials, the README/TOC/asset edits made, remaining author actions (hardware run attestation, email notification + tracker, org-access), the branch/PR steps to run, and the post-publication maintenance responsibility.
