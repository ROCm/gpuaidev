# CI requirements

Every tutorial notebook is run regularly, end to end, on AMD GPUs to catch
breakage before readers do. The runner reads what a notebook needs from the
notebook itself and picks a machine that has it, so a notebook written for a
Radeon card runs on a Radeon card and one that needs eight GPUs runs on an
eight-GPU node.

Most notebooks need nothing: with no declarations, a notebook runs on one AMD
Instinct GPU. Declare only what differs. The declarations are invisible on the
documentation site and in Jupyter.

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
    "hardware": ["cdna4"],
    "gpus": 8
  }
}
```

| Key | Meaning | Default |
|---|---|---|
| `hardware` | GPUs the notebook runs on: any of a product (`mi300x`, `mi355x`, `rx7900xtx`), an architecture (`gfx942`, `gfx950`, `gfx1100`) or a family (`instinct`, `radeon`, `cdna3`, `cdna4`, `rdna3`, `rdna4`) | `["instinct"]` |
| `gpus` | GPUs used at once, all on one node | `1` |
| `gpu_memory_gb` | Memory needed on each GPU | none |
| `memory_gb` | Host memory needed | 64 |
| `disk_gb` | Free disk needed for models and data | 100 |
| `capabilities` | Things the machine must provide that the notebook does not set up itself, for example `ollama` (a running Ollama server) or `rdma` | none |
| `env` | Environment variables to set, for example `{"WANDB_MODE": "disabled"}` | none |
| `image` | Docker repository the notebook runs in, when it is not the one its imports suggest (`rocm/pytorch`, `vllm/vllm-openai-rocm`, `lmsysorg/sglang`); the runner picks the tag | inferred |
| `parameters` | Values for the cell tagged `parameters` in CI (see below) | none |
| `depends_on` | Notebooks (paths under `docs/notebooks`) that must run first | none |

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
