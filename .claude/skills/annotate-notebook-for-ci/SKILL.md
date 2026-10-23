---
name: annotate-notebook-for-ci
description: Prepare a tutorial notebook under docs/notebooks so CI can run it unattended — declare its metadata.ci requirements, start servers from cells, make setup fail loudly, add assertions and CI parameters for shorter runs, and check the result. Use when adding a new tutorial, when a notebook fails or is blocked in the regression runs, or when asked to "annotate", "make CI-ready" or "make runnable" a notebook.
---

# Annotate a notebook so CI can run it

CI executes every tutorial top to bottom with papermill, in the Docker image
the notebook needs, on a machine that meets its declared requirements, and
judges it from the executed notebook: **pass** if every code cell ran without
an error, **fail** if one raised, **blocked** if it could not run or stopped
part-way. `.github/CI_METADATA.md` is the reference for every key and
convention; this skill is the workflow. The notebook is for readers first:
every change must keep it a good tutorial for someone running it by hand in the
documented container.

## 1. Read the whole notebook

Read every cell, code and markdown, before changing anything. Note:

- **Hardware and GPU count** from the Hardware section and the code
  (`-tp 8`, `--nproc_per_node`, `HIP_VISIBLE_DEVICES`, `TP=`, hard-coded GPU lists).
- **The container** the setup section tells readers to use, and what the cells
  import. That decides `ci.image`.
- **What a person has to do**: open a terminal, run a server in another
  window, paste a token, click in a browser, answer `input()`.
- **What the cells install**: `!pip install`, `git clone`, `apt`, `curl | sh`.
- **Gated models** (`meta-llama/*`, `google/paligemma*`, ...). CI's Hugging Face
  token needs access to each; check before relying on one.
- **Files outside the notebook**: `../assets/...`, files the reader must
  download, absolute paths such as `/workspace/notebooks/...` or `/root/...`.

## 2. Declare `metadata.ci`

Add the `ci` object (keys and defaults in `.github/CI_METADATA.md`). Rules
learned from running all the tutorials:

- `hardware` and `gpus` are required. `gpus` is what **one CI run** needs:
  the fewest GPUs that work, not the tutorial's showcase count (see step 6).
  The Prerequisites must state that same minimum as **GPUs required**, with the
  settings that need it and the default configuration as the optional one: the
  contribution guidelines require regression runs to use the documented
  minimum GPU count.
- Set `ci.image` whenever the image is not what the imports suggest: a notebook
  that only calls `vllm serve` has no `import vllm`, so declare
  `vllm/vllm-openai-rocm`. CI resolves the newest release tag; pin a tag
  (`rocm/primus:v25.9_gfx942`) only when the code needs that release, and
  say why in the notebook.
- Pin every third-party `git clone` the cells run to a **release tag**, never a
  moving branch (`main`, `release/v26.8`) or an unpinned clone: a floating ref
  silently changes under the tutorial when upstream moves, so a notebook that
  passed last week fails — or quietly produces a different result — with no
  change of its own. A bare commit SHA is reproducible but opaque; prefer the
  tag that points at it. When a tutorial pins a personal fork that upstream has
  since absorbed or abandoned, move to the maintained upstream and pin its
  newest release (and if it has no tags, a dated SHA on its release branch, with
  the date in a comment). Prefer the upstream project over an AMD fork whenever
  upstream supports ROCm.
- Before pinning a clone to a newer ref, confirm it runs on the **image's Python
  version**: a "newer is better" bump can break: e.g. a ref that imports
  `typing.override` needs Python 3.12 and fails on a 3.10 image. Match the ref to
  the interpreter the image ships, or bump the image too — and validate, don't
  assume. Run the clone/checkout in a `%%bash` cell with `set -e` (see step 4)
  so a bad ref fails the cell instead of silently reusing a prior checkout.
- Large checkpoints: vLLM maps the whole checkpoint into host memory while
  loading. Set `memory_gb` above the checkpoint size (the default 64 GB fails
  for a ~200 GB model) and `gpu_memory_gb` when the model fills a GPU.
- `system_packages` for every tool the cells use that a minimal image may lack
  (`git`, `wget`, `unzip`, `ffmpeg`, `zstd`, `cmake`). Prefer this to
  `apt install` cells.
- `requirements` for Python packages readers must install before the first
  cell: list them in the notebook's install cell and give the same list in
  `ci.requirements`, so readers and CI install the same thing. Don't add files
  next to the notebook: a tutorial is a single notebook, and anything else it
  needs is downloaded or generated from its cells.
- `sandbox: false` only when a tool launches its own Python processes with the
  image's interpreter (Ray workers); otherwise CI's virtual environment is
  right.
- `timeout_minutes` covers model downloads too; set it from a measured run.

## 3. Remove the person from the loop

- **Servers** (vLLM, SGLang, Ollama, ComfyUI): start them from a code cell with
  the `ensure_server` / `stop_server` helpers (copy the helper cell from
  `inference/build_airbnb_agent_mcp.ipynb`), poll a health URL, and stop the
  server at the end. Replace "open a terminal and run" text with "the cell
  below starts it"; keep the explanation of the command and flags.
- **No `docker` in code cells.** CI runs the notebook inside a container, and
  pods cannot start nested containers. Run the server as a background process
  in the notebook's own image and declare that image in `ci.image`; the setup
  markdown keeps one `docker run -it ... <image>` for readers.
- **Interactive setup** (onboarding wizards, `notebook_login()`): use the
  tool's non-interactive flags or environment variables. `notebook_login()`,
  `input()`, Gradio `launch()` and audio playback are handled by CI; leave them.
- Tag a cell `ci-skip` only when it genuinely needs a person or a browser.
  Never for slow or broken cells: CI is there to report those.

## 4. Make failures visible

- A `!command` line never fails its cell. Setup and the steps that produce the
  tutorial's results go in a `%%bash` cell starting with `set -e`, or
  `subprocess.run(cmd, shell=True, check=True)` when Python values are needed.
  A `!` line that fails silently can turn a broken notebook into a pass.
- End each stage that produces something with a short assertion on it: the
  file exists and is non-empty, the checkpoint was written, the video has
  frames, the server answered, training reported a final loss.
- Scripts that always exit 0 (some training launchers) need their log checked
  for the line that proves they finished.

## 5. Interpolation and paths

- IPython substitutes `{VAR}` and `$VAR` from Python into `!` lines, **not**
  into `%%bash` cells. For `%%bash`, export first:
  `os.environ["NUM_GPUS"] = str(NUM_GPUS)`, then use `$NUM_GPUS`.
- Markdown is never interpolated: write "`TP` is the number of GPUs
  (`NUM_GPUS`)", not `TP={NUM_GPUS}`.
- A package installed in editable mode (`pip install -e`) after the kernel
  started is not importable in that kernel. Add its source directory to
  `sys.path` in the next cell, and clone it under a name that differs from the
  package (`aiter_src`, not `aiter`), or the checkout shadows the package.
- `../assets/<file>` reads work in CI. Avoid absolute paths that only exist in
  the author's container layout; derive them from where the cell cloned or
  wrote things.

## 6. Shorter runs and fewer GPUs, without changing the tutorial

Put each knob in **one** code cell tagged `parameters`, with the reader's value
as the default, and give CI's value in `ci.parameters`:

```python
NUM_GPUS = 8          # GPUs for tensor parallelism / data parallelism
TRAIN_ITERS = 50      # training iterations
NUM_SAMPLES = None    # None uses the whole dataset
```

```json
"ci": {"gpus": 1, "parameters": {"NUM_GPUS": 1, "TRAIN_ITERS": 10, "NUM_SAMPLES": 64}}
```

- Thread the variables into every place the value appears (server flags,
  launchers, visible-device lists, config files the notebook edits).
- Choose CI values from memory, not hope: an MI300X has 192 GB. Weights in
  bf16 take 2 bytes per parameter; full fine-tuning with Adam needs about
  16 bytes per parameter plus activations.
- Keep `ci.gpus` equal to the CI value of the GPU-count parameter.
- Evaluation limits must stay valid: a perplexity evaluation needs enough
  tokens for one window, so do not shrink it to a handful of rows.

## 7. Keep cells readable

How the code is split into cells is part of the tutorial: a reader runs them one
at a time, so each cell should be one comprehensible step introduced by a line of
prose. When you add or change code, keep its cells to these rules.

- **Never leave code cells back to back with no explanation.** Two or more code
  cells in a row with nothing between them leave the reader guessing what each is
  for. Put at least a one-line markdown cell before a code cell that a nearby
  heading does not already explain — say what the cell does and why, not how.
- **Split a cell that does too much.** A code cell that runs several distinct
  steps (install, then configure, then launch) overwhelms the reader and hides
  which step failed. Break it along those boundaries into one cell per step, each
  with its own one-line lead-in.
- **Merge cells too small to run on their own.** A one-line import followed by a
  one-line call, run back to back with nothing between them, is two clicks for no
  reason — merge them into one cell. Do not merge across a markdown explanation
  the reader needs, or steps deliberately kept apart to show output in between.
- **Keep them readable in the one-click render too.** The notebooks.amd.com
  one-click launch serves a *stripped* notebook: it removes setup and marked
  cells, so a markdown or setup cell sitting between two code cells can vanish
  and leave them back to back even when the source reads fine. These rules must
  hold after that transform, not just in the repo. This is checked locally when
  the style guide is applied (the fan-out renders each notebook through the real
  one-click transform and rescans) — it is not a CI gate, so verify it before
  you finish.

## 8. Keep the text true

After changing code, update the markdown that describes it: image tags and
versions, GPU counts in the Prerequisites, commands in "expected output"
blocks, steps the cells now do (installing and starting servers). Remove
nothing that explains the technique.

- **No performance claims.** Don't add speedups, throughput, latency or
  tokens-per-second figures, or comparisons with other kernels or hardware,
  even ones you measured. Explain what a benchmark measures and how to read its
  output, and let readers compare the numbers their own run prints.
- **Describe the current setup, not its history.** Write "this tutorial uses
  Qwen2.5-VL", not "earlier versions of this tutorial used Llama 3.2 Vision";
  a changelog in the text is noise to a reader following the steps.
- **Write for readers, not for CI.** Markdown says what a step does for the
  reader ("stop the script from killing the notebook's kernel"), never that it
  makes the notebook "run unattended".

## 9. Check

1. `python3 .github/scripts/check_ci_metadata.py docs/notebooks/<path>.ipynb` —
   the metadata is well formed and every `hardware` name is in the vocabulary.
   List every GPU the Hardware section names, by product (e.g.
   `["mi300x", "w7900"]` for "MI300X or Radeon Pro W7900", `["mi100", "mi210",
   "mi300x"]` for a notebook that runs on all three). Use the product id, not the
   family (`instinct`/`radeon`), so the declaration states exactly what the
   tutorial supports; this check gates the PR.
2. Keep the notebook's JSON formatting: edit metadata in JupyterLab (Property
   Inspector) or VS Code (Edit Notebook Metadata), or with a script that writes
   it back with the file's own indent and `ensure_ascii=False`, so the diff
   shows only your change.
3. Run it the way CI does, in the declared image with the CI parameters:

   ```bash
   pip install papermill ipykernel
   papermill docs/notebooks/<path>.ipynb /tmp/out.ipynb \
     -p NUM_GPUS 1 -p TRAIN_ITERS 10      # the values from ci.parameters
   ```

   Then read the output of every `!` cell and every assertion, not just the
   final status: a pass with a hidden traceback is still a broken tutorial.
