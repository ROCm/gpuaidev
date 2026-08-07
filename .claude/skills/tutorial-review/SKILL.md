---
name: tutorial-review
description: Reviews a Pull Request that adds or updates a Jupyter Notebook tutorial for the AMD ROCm AI Developer Hub (gpuaidev-internal). Use when the user asks to review a tutorial PR, check a tutorial notebook against the contribution guidelines, validate that a tutorial is wired into the README and Sphinx TOC, scan for performance claims or AMD forks, or produce a reviewer verdict. Runs automated repo-wiring and text checks only; does not execute hardware-specific notebook cells.
---

# Tutorial Review Skill

Use this skill to review a tutorial contribution PR against the AMD AI Developer Hub contribution guidelines and this repo's real conventions. Produce a structured, actionable verdict.

If invoked with arguments, treat them as the notebook path, PR number, or branch: $ARGUMENTS

## Scope: what to check automatically vs. what to defer

Tutorial code cells target specific AMD GPU hardware (Instinct MI300X, Radeon, CDNA4, etc.) that is not available in this environment. DO NOT attempt to execute notebook code cells and never claim you verified end-to-end execution.

Actively check (no hardware needed):
- Repo wiring: README table row, `docs/sphinx/_toc.yml.in` entry, correct category directory, path consistency.
- Notebook static structure: required header fields, Prerequisites content, JSON/nbformat validity.
- Text scans: performance-claim keywords, AMD-fork vs upstream signals, leaked secrets, absolute local paths.

Defer to the author (attestation, not verification):
- That the notebook runs end-to-end on the documented AMD GPU config with the documented GPU count and no errors.
State clearly in your verdict that hardware execution was NOT verified by you and relies on author attestation.

## Step 1: Identify the changed notebook(s)

Determine which `.ipynb` file(s) the PR adds or modifies (e.g., via `git diff --name-only main...HEAD` or the provided path). Establish for each: the filename, the category directory it lives in, and whether it is new or an update.

Valid category directories:
- `docs/notebooks/inference/`
- `docs/notebooks/fine_tune/`
- `docs/notebooks/pretrain/`
- `docs/notebooks/gpu_dev_optimize/`

A tutorial outside these directories is a blocking issue unless the maintainer intends a new category.

## Step 2: Repo-wiring checks (automated)

For each new tutorial file `docs/notebooks/<category>/<file>.ipynb`:

1. README.md: confirm a row exists in the "Current Notebooks" table under the correct category, that its GitHub link is `https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/<category>/<file>.ipynb` and its AMD Tutorial link is `https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/<category>/<file>.html`.
2. `docs/sphinx/_toc.yml.in`: confirm an entry `- file: notebooks/<category>/<file>.ipynb` with a `title:` exists under the matching `- caption:` block, using 2-space indentation.
3. Consistency: the file path in README, the TOC, and the actual file on disk must be identical. The title in README and TOC should match.
4. The `- file:` path in the TOC must resolve to a real file (no typos / stale paths).

Caption-to-directory mapping:
- `Inference tutorials` -> `notebooks/inference/`
- `Fine-tuning tutorials` -> `notebooks/fine_tune/`
- `Pretraining tutorials` -> `notebooks/pretrain/`
- `GPU development and optimization tutorials` -> `notebooks/gpu_dev_optimize/`

5. Assets: for every markdown image reference of the form `![...](../assets/<file>)` in the notebook, confirm the file exists under `docs/notebooks/assets/` and is included in the PR. A referenced-but-missing asset is a blocking issue (broken image in the published page).
6. Documentation build compatibility (Sphinx/nbsphinx): the notebook is rendered into the docs site by Sphinx/nbsphinx, so static rendering must not break. Statically verify (no execution): valid notebook JSON (`nbformat: 4`); every markdown image/link path resolves (assets via `../assets/<file>`, in-notebook anchor links like `[x](#id)` have a matching `<a id="id"></a>`); no malformed markdown tables or unbalanced code fences; the TOC `- file:` path matches the on-disk file exactly (a stale/typo path breaks the build). If a docs build is available locally, building it is a safe non-hardware check; otherwise flag build compatibility as the final call of the ROCm documentation team and list any static red flags you found.

You may use a short python/grep pass over `README.md`, `_toc.yml.in`, and `docs/notebooks/assets/` to confirm the file path appears where required and that referenced assets exist. This is repo metadata, not hardware code, so it is safe to run.

## Step 3: Notebook structure checks (automated, static)

Parse the notebook JSON (do not execute it) and confirm the conventions shared by the 40 published tutorials:

1. Valid `nbformat: 4`, python3 kernelspec present.
2. First markdown cell has a single `# H1` title.
3. Header block contains:
   - `**Author**: <name>`
   - `**Knowledge level**:` set to exactly one of `Beginner`, `Intermediate`, `Advanced` (100% of published tutorials include this).
4. Introduction present: explains goal, technology/model, workflow, and learning outcome.
5. `## Prerequisites` section present with the canonical sub-structure (`### Operating system`, `### Hardware`, `### Software`, and `### Hugging Face API access` when gated models are used). It must include, where applicable:
   - AMD GPU model or family (AMD Instinct or AMD Radeon - Radeon-targeted tutorials are valid; confirm the stated family matches the tutorial's intent)
   - Explicit GPU count required (minimum vs optional if multiple configs)
   - ROCm version + a verify step (`amd-smi` / `rocm-smi`), OS, Python version, key package/framework versions, Docker image, model/dataset, GPU/system memory + storage considerations, API tokens.
   A missing GPU model or missing GPU count is a blocking issue.
6. Environment setup + Jupyter launch present (the hub's core convention: 33/40 use `docker run`, 39/40 include Jupyter launch). Confirm the notebook contains:
   - A reproducible environment setup: a `docker run` block using a ROCm-based image (e.g., `rocm/pytorch`, `rocm/vllm-dev`, `lmsysorg/sglang:*-rocm*`, `vllm/vllm-openai-rocm`) with the standard device flags (`--device=/dev/kfd`, `--device=/dev/dri`, `--group-add=video`, `--ipc=host`, `--shm-size`), OR an explicit conda/venv isolated environment on a documented ROCm install.
   - Jupyter launch instructions (e.g., `pip install jupyter` + `jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root`).
   Flag as an issue for reviewer judgement: a bare host-OS "just pip install" setup with no container/env isolation and no Jupyter launch steps. Note: prefer a pinned image tag over `latest` (recommendation, not a blocker).
7. GPU count consistency (blocking on mismatch): the GPU count stated in Prerequisites must match what the code actually uses. Read the launch/config cells and compare against the documented count. Check the mechanisms the repo uses: `tensor_parallel_size=N` / `--tp N` (vLLM/SGLang), `nproc_per_node N` / `world_size` (distributed launchers), and `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES`. Example: Prerequisites says "8 GPUs" but code uses `--tp 8` or `nproc_per_node 8` -> consistent; if it says 8 but code uses `--tp 1`, flag it.
8. Downloads inside the notebook: models/datasets/supplementary files should be fetched or generated within the notebook (HF `snapshot_download`/`hf_hub_download`/`load_dataset`, `wget`/`curl`, or `git clone` for a supporting repo), each with a preceding markdown explanation. For gated models, confirm an HF login/token step exists and Prerequisites documents the required access. Flag references to local hand-authored files that are not downloaded/generated (also covered in Step 4 item 4).
9. Educational format and cell granularity: multiple focused code cells interleaved with markdown, not one or two giant code cells. Ideally one logical task per cell (imports / config / model loading / data prep / setup / execution / result inspection separated). Concepts should be introduced in markdown before the implementing code. Flag any single code cell that dominates the notebook without explanation.
10. Output explanations (recommendation): for cells producing meaningful results (model responses, training progress, tensor shapes, GPU detection, generated files, validation results), the notebook should tell the reader what to expect. Flag important cells with no explanation as a recommendation.
11. Upstream links and licensing (recommendation): the tutorial should link to the upstream project/model/dataset/docs so readers can learn more, and prefer permissively licensed open-source resources. Flag missing upstream references as a recommendation.
12. Reproducible dependencies (recommendation): check whether key `pip install` versions are pinned, the model revision/dataset version is pinned, the ROCm version is documented, and the Docker image uses a pinned tag. Unpinned deps are a maintainability recommendation, not a blocker.
13. Screenshots (recommendation): screenshots are acceptable when they add value, but flag cases where a screenshot replaces information that should be textual/reproducible in the notebook.
14. Callouts and endings (recommendations, not blockers): `**Note**:` callouts used for caveats/permissions (near-universal); a wrap-up section (Summary / Conclusion / Next steps / Further reading) is valued. Do NOT require an in-notebook license footer or disclaimer (licensing is centralized in `docs/notebooks/licensing.md`); flag it only if one was mistakenly added with legal/performance language.

## Step 4: Content-policy text scans (automated)

1. Performance / benchmarking claims. The underlying rule is legal: unapproved performance claims about AMD hardware on the official AMD page require legal review. Do NOT hard-block on the mere presence of a timing word. Scan markdown and comments (case-insensitive) for: `speedup`, `faster`, `x faster`, `throughput`, `latency`, `tokens/s`, `tokens per second`, `tok/s`, `training speed`, `efficiency`, `outperform`, `benchmark`, `vs nvidia`, `versus nvidia`, `compared to`, `A100`, `H100`, `H200`. For each hit, read the surrounding context and classify it:
   - BLOCKING (needs legal approval): an absolute/specific performance figure presented as AMD hardware capability (e.g., "MI300X reaches N tokens/sec", "achieves N ms latency", "N% faster on AMD"), OR any competitive/cross-vendor comparison (vs NVIDIA, A100/H100/H200, "faster than <other hardware>").
   - ALLOWED (educational, not a violation): a pedagogical baseline-then-improve narrative that is relative and self-contained within the tutorial (e.g., "start from this baseline, apply the optimization in this exercise, and observe the relative improvement"), and naturally emitted framework timings that are not framed as an absolute AMD claim or a competitive comparison.
   Report BLOCKING hits with cell reference and quoted context. Report ALLOWED-but-borderline hits as recommendations for the reviewer to confirm, not as blockers.
2. Untrusted / non-upstream sources (provenance check). The tutorial should pull code, packages, images, models, and data from official upstream or well-known trusted sources, never from a personal or throwaway account. Scan `git clone`, `pip install`, `docker run`/`docker pull`, and any URLs, and classify each source:
   - AMD fork vs upstream (blocking if a mainstream project is pulled from an AMD/ROCm fork that upstream already supports on ROCm). Look for `github.com/ROCm/<mainstream-project>` or other AMD-specific forks of major frameworks used in place of the upstream repo.
   - Personal / non-official Docker images (flag for reviewer judgement). The base image should be an official ROCm-ecosystem image (`rocm/*`, `vllm/*`, `lmsysorg/*`, or another recognized vendor namespace). Flag images from a personal Docker Hub namespace, an unpinned `:latest` on an unofficial repo, or a raw registry IP/host (e.g., `docker run <user>/<image>`, `myregistry.local/...`).
   - Personal / unofficial GitHub repos (flag for reviewer judgement). Flag `git clone` or `pip install git+https://...` pointing at an individual's account, a fork that is not the upstream project, a gist, or a repo that is not the recognized upstream / official org for that project. The reader should be sent to the canonical upstream.
   - Non-canonical package sources (flag). Flag `pip install` using a custom `--index-url`/`--extra-index-url` to a non-official index, a personal wheel URL, or an install from a personal fork instead of PyPI / the official ROCm index / the upstream project.
   For every hit, report the source URL/namespace, the cell reference, and whether an official upstream equivalent exists that should be used instead.
3. Leaked secrets (blocking). Scan for hardcoded tokens/keys: `hf_[A-Za-z0-9]`, `sk-`, `AKIA`, `Authorization: Bearer`, `password=`, private IPs, and personal absolute paths like `/home/<user>/` or `C:\\Users\\`.
4. Single-notebook packaging: flag references to local hand-authored source files that are not downloaded/generated within the notebook.

## Step 5: Category, overlap, and update-PR context

1. Category fit: confirm the tutorial fits its claimed category (Inference / Fine-tuning / Training / GPU Development and Optimization) and lives in the matching directory.
2. Overlap check: compare the topic against existing tutorials in the same category (README table and `docs/notebooks/<category>/`). If a closely related tutorial already exists, the PR should articulate what this one adds (different framework, workflow, model, depth, or perspective). A near-duplicate with no differentiation is a reviewer concern to raise, not an automatic block.
3. Update vs new: if this PR updates an existing tutorial (not a brand-new file), evaluate it against the author's post-publication maintenance responsibility: the change should keep the notebook current with ROCm releases, package/framework/model/API/Docker-image changes, and should not regress the documented setup, GPU count, or wiring. For updates, verify the README/TOC/title/path are still consistent if any of those changed.

## Step 6: Subjective editorial assessment (read the notebook narrative)

These are qualitative judgments, not mechanical checks. Actually read the notebook's markdown narrative end to end (you can read the cells even though you cannot execute the code) and assess:

### 6a. Pedagogical flow (does it teach well?)
- Does the tutorial have a clear, logical arc: motivation/goal -> prerequisites -> stepwise build-up -> working result -> wrap-up? Or does it jump around?
- Is each major code cell preceded or followed by markdown explaining what it does, why the step is needed, and what output to expect? Or are there unexplained code dumps?
- Are concepts introduced before the code that implements them?
- Are steps broken into small, understandable units rather than one or two giant cells?
- Does it teach the subject, or merely present a finished application to copy-paste?
- Is the writing clear, approachable, and free of unexplained jargon?

Rate the flow qualitatively (e.g., strong / acceptable / needs work) and give specific examples of where the narrative is strong or where it breaks down. Weak flow is normally a Recommendation, not a hard blocker, unless the notebook is essentially undocumented code (which also fails the educational-format check in Step 3).

### 6b. Topic relevance and recency on ROCm
- Is the subject a recent, relevant, or emerging AI topic (current models, frameworks, techniques), or is it dated/superseded?
- Is it genuinely demonstrated on AMD hardware via the ROCm stack (not a generic tutorial that happens to mention AMD)?
- Does it use upstream open-source projects rather than AMD-specific forks (cross-check Step 4 item 2)?
- Does it add value beyond existing tutorials in the same category?

Give a short judgment on whether the topic is worth publishing now and whether it showcases the AI ecosystem working on AMD/ROCm. Note that these are subjective calls; surface your reasoning so the human maintainer can override.

## Step 7: Produce the review verdict

Output a structured report:

- **Summary**: tutorial title, category, file path, new vs update.
- **Editorial assessment**: pedagogical flow rating (strong / acceptable / needs work) with specific examples, and a topic relevance/recency judgment (is it a current AI topic genuinely demonstrated on ROCm). Mark these clearly as subjective calls the human maintainer can override.
- **Overlap finding**: whether a related tutorial already exists in the category and what this one adds (from Step 5).
- **Blocking issues** (must fix before merge): each with file/cell reference and the guideline it violates.
- **Recommendations** (non-blocking improvements, including flow/clarity suggestions from Step 6 and any unpinned-dependency / missing-upstream-link notes).
- **Passed checks**: concise list of what was verified (wiring, structure, content-policy scans).
- **Documentation build compatibility**: static red flags found (broken asset/anchor paths, unbalanced fences, stale TOC path), or a note that final build validation is the ROCm documentation team's call.
- **Not verified (author attestation required)**: end-to-end execution on the documented AMD GPU hardware with the documented GPU count.
- **Reviewer actions**: confirm `Mahdi-CV` is set as reviewer (per `.github/CODEOWNERS`), and note that after merge the ROCm documentation team performs a final documentation review and the content is synchronized/mirrored to the public `ROCm/gpuaidev` repository.

Map every blocking issue back to a specific guideline so the author knows exactly what to change.
