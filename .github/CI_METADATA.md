# CI requirements

Every tutorial notebook is run regularly, end to end, on AMD GPUs to catch
breakage before readers do. The runner reads what a notebook needs from the
notebook itself and picks a machine that has it, so a notebook written for a
Radeon card runs on a Radeon card and one that needs eight GPUs runs on an
eight-GPU node.

Every tutorial under `docs/notebooks` declares at least the hardware it runs
on and how many GPUs it uses; the rest is optional. The declarations are
invisible on the documentation site and in Jupyter.

The [CI metadata](workflows/ci-metadata.yml) check validates them on every
pull request that changes a notebook.

## Notebook requirements

Add a `ci` object to the notebook's metadata. In JupyterLab, open the
**Property Inspector** (the gear icon in the right sidebar) and edit
**Notebook metadata**; in VS Code, use **Edit Notebook Metadata** from the
"..." menu.

```json
"metadata": {
  "ci": {
    "hardware": ["instinct"],
    "gpus": 1
  }
}
```

| Key | Meaning | Default |
|---|---|---|
| `hardware` | GPUs the notebook runs on, matching its Hardware section: any of a product (`mi300x`, `mi308x`, `mi325x`, `mi350x`, `mi355x`, `rx7900xtx`, `w7900`, `rx9070xt`), an architecture (`gfx942`, `gfx950`, `gfx1100`, `gfx1201`) or a family (`instinct`, `radeon`, `cdna3`, `cdna4`, `rdna3`, `rdna4`). List every option the notebook supports, for example `["instinct", "w7900"]` | required |
| `gpus` | GPUs used at once, all on one node | required |
| `gpu_memory_gb` | Memory needed on each GPU | none |
| `memory_gb` | Host memory needed | 64 |
| `disk_gb` | Free disk needed for models and data | 100 |
| `capabilities` | Things the machine must provide that the notebook does not set up itself, for example `ollama` (a running Ollama server) or `rdma` | none |
| `env` | Environment variables to set, for example `{"WANDB_MODE": "disabled"}` | none |
| `requirements` | Python packages CI installs before running the notebook: a list of requirement strings, or the name of a requirements file next to the notebook (`"requirements.txt"`) | none |
| `system_packages` | Ubuntu (apt) packages the notebook's commands need that a minimal ROCm image may lack, for example `git`, `wget`, `unzip`, `cmake`, `ffmpeg`, `python3-venv` | none |
| `image` | Docker repository the notebook runs in, when it is not the one its imports suggest (`rocm/pytorch`, `vllm/vllm-openai-rocm`, `lmsysorg/sglang`); the runner picks the tag, unless the notebook pins one (`rocm/primus:v25.9_gfx942`) because it only works with that release | inferred |
| `sandbox` | `false` runs the notebook in the image's own Python instead of a fresh virtual environment, for notebooks whose tools start their own Python processes with the image's interpreter (for example Ray workers) | `true` |
| `parameters` | Values for the cell tagged `parameters` in CI (see below) | none |
| `depends_on` | Notebooks (paths under `docs/notebooks`) that must run first | none |
| `timeout_minutes` | How long one CI run of the notebook may take, model downloads included; a run that exceeds it is stopped | 60 |

Prefer making the notebook set things up itself over adding a capability:
a notebook that installs and starts the server it needs runs anywhere.

## Cells that cannot run unattended

Tag a code cell `ci-skip` when it cannot finish without a person: it opens a
browser UI, waits for typed input in a loop, or starts a foreground server
meant for a browser. CI skips the cell and grades the rest of the notebook.

```json
"metadata": {
  "tags": ["ci-skip"]
}
```

Don't tag cells that are merely slow or broken: CI is there to report those.
Gradio `launch()` calls, audio playback, and cells that only call `input()`
are skipped automatically.

## Dependencies

CI runs every notebook in a fresh virtual environment layered on the image's
Python, so the image's ROCm PyTorch stays available while everything the
notebook installs, including its own `!pip install` / `%pip install` cells,
goes into that environment. Before the first cell runs, CI installs the
`system_packages` and `requirements` the notebook declares.

Declare what the notebook's cells use but do not install themselves: a
`git clone` needs `git`, `wget` and `unzip` are not in every image, Whisper
needs `ffmpeg`, `python3 -m venv` needs `python3-venv`. If readers must install
Python packages before the first cell, install them from a code cell in the
notebook and give the same list in `ci.requirements`, so CI and readers install
the same thing. A tutorial is a single notebook: readers should not have to
fetch extra files to follow it, so list the packages in the notebook rather
than in a requirements file next to it.

## Servers

A notebook that needs a server (vLLM, SGLang, Ollama, ComfyUI) starts it from a
code cell instead of asking the reader to open a terminal, so it runs the same
way for readers and in CI:

1. A cell defines `ensure_server`, `stop_server` and `server_is_up` (copy them
   from any of the server notebooks, for example
   `inference/build_airbnb_agent_mcp.ipynb`).
2. `server = ensure_server(command, health_url, log_path)` starts the command in
   the background, unless the URL already answers, and waits until it does. If
   the server exits or is not ready in time, it raises with the end of the log.
3. `stop_server(server)` stops it when the notebook is done with it, for
   example before starting a different server on the same port.
   For a server started with `docker run`, the container stops shortly after
   its client does; give it a `--name` and run `docker stop <name>` when the
   next step needs it gone.

Keep the markdown that explains the command and its flags; only the "open a
terminal and paste this" instructions go.

## Make setup steps fail loudly

A `!command` line never fails its cell: the cell carries on after a failed
`!curl ... | sh` or `!pip install`, and the notebook breaks later with a
confusing error, or not at all. Write setup that later cells depend on as a
`%%bash` cell starting with `set -e`, which fails the cell on the first error.

## Check results, not just that cells ran

CI records a pass when every code cell runs without an error. A cell that
silently skips work (a missing checkpoint, an empty output) still passes, so
end each stage that produces something with a short assertion on it: the file
exists and is not empty, the mesh has faces, the video has frames, the server
returned an answer.

## Shorter runs in CI

When a full run takes hours (100 training epochs, a full benchmark), put the
knob in its own code cell tagged `parameters`, use the variable where it is
needed, and give the CI value in `ci.parameters`:

```python
NUM_EPOCHS = 100  # Total number of training epochs
```

```json
"ci": {"parameters": {"NUM_EPOCHS": 1}}
```

Readers still see and run the full value; CI injects the short one after the
`parameters` cell (this is how [papermill](https://papermill.readthedocs.io)
parameterizes notebooks).

## Check your declarations

```bash
python3 .github/scripts/check_ci_metadata.py docs/notebooks/pretrain/ddim_pretrain.ipynb
```

With no arguments, the script checks every notebook in the repository.
