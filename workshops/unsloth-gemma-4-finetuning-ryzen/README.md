# Finetune Gemma 4 with Unsloth on AMD Radeon

> **Train smarter, not harder — on your AMD GPU!**

Welcome to the **Unsloth x AMD Radeon Finetuning Workshop**! In this hands-on session you will use [Unsloth](https://unsloth.ai) to fine-tune **Google's Gemma 4** model with reinforcement learning — right on AMD hardware powered by ROCm. By the end you'll have a model that has learned to solve Sudoku puzzles. Let's go!

---

## What You'll Build

- **Model:** [Gemma 4](https://ai.google.dev/gemma) fine-tuned with GRPO reinforcement learning
- **Task:** Teach the model to solve Sudoku puzzles
- **Hardware:** AMD Radeon GPU (ROCm 7.2+)
- **Framework:** [Unsloth](https://unsloth.ai) — up to 2x faster training, 70% less VRAM

---

## Useful Resources

| Resource | Link |
|---|---|
| Unsloth Docs | [docs.unsloth.ai](https://docs.unsloth.ai) |
| Unsloth Studio (no-code finetuning!) | [unsloth.ai/studio](https://unsloth.ai/studio) |
| Unsloth GitHub | [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth) |
| Unsloth Discord | [discord.gg/unsloth](https://discord.gg/unsloth) |
| Unsloth Blog | [unsloth.ai/blog](https://unsloth.ai/blog) |
| ROCm Docs | [rocm.docs.amd.com](https://rocm.docs.amd.com) |
| Gemma Models on Hugging Face | [huggingface.co/google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) |

---

## Prerequisites

- Ubuntu 22.04 / 24.04 machine with an AMD GPU
- Internet access
- Workshop credentials:
  ```bash
  Username: amd-user
  Password: amd1234
  ```

---

## Step 1 — Check ROCm Installation

> Full reference: [Install Ryzen Software for Linux with ROCm](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/native_linux/install-ryzen.html)

ROCm is already installed on the **host machine**. Verify it's working:

```bash
rocminfo
```

You should see your AMD GPU listed. If the command is not found, follow the [ROCm installation guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/native_linux/install-ryzen.html).

---

## Step 2 — Start the Docker Container

If a container is already running, clean it up first:

```bash
docker kill workshop-env
docker rm  workshop-env
```

Then launch a fresh container with your GPU and workspace mounted:

```bash
docker run -it \
  --name workshop-env \
  --ipc=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt seccomp=unconfined \
  -v $HOME:/root/home \
  -v $PWD:/workspace \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -w /workspace \
  -p 8888:8888 \
  -e PYTHONUNBUFFERED=1 \
  -e HF_HOME=/root/.cache/huggingface \
  rocm/vllm-dev:rocm7.2.1_navi_ubuntu24.04_py3.12_pytorch_2.9_vllm_0.16.0 \
  bash
```

<details>
<summary><b>What do those flags do?</b></summary>

| Flag | Purpose |
|---|---|
| `--device=/dev/kfd` & `--device=/dev/dri` | Exposes your AMD GPU to the container |
| `--ipc=host` | Shared memory for faster GPU communication |
| `--security-opt seccomp=unconfined` | Required for ROCm GPU access |
| `-v $HOME:/root/home` | Mounts your home directory so your files persist |
| `-v $PWD:/workspace` | Mounts the current folder into `/workspace` |
| `-w /workspace` | Sets the working directory inside the container |
| `-p 8888:8888` | Exposes Jupyter Notebook on port 8888 |

</details>

You should now be inside the container shell at `/workspace`.

---

## Step 3 — Install Jupyter Notebook (inside the container)

```bash
pip install notebook ihighlight
```

---

## Step 4 — Download & Run the Workshop Notebook

Grab the notebook and fire it up:

```bash
cd /workspace

curl -L -o "Gemma4_(E2B)_Reinforcement_Learning_Sudoku_Game.ipynb" \
  "https://raw.githubusercontent.com/iswaryaalex/Unsloth-RL-Workshop-on-Radeon/main/Gemma4_(E2B)_Reinforcement_Learning_Sudoku_Game.ipynb"
```

Then start Jupyter:

```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Open the URL printed in the terminal (e.g. `http://<machine-ip>:8888/?token=...`) in your browser and open the downloaded notebook.

---

## Step 5 — Navigate the Notebook

Once the notebook is open in your browser, here's everything you need to know:

### Running Cells

| Action | Keyboard Shortcut |
|---|---|
| **Run the current cell and move to the next** | `Shift + Enter` |
| **Run the current cell and stay on it** | `Ctrl + Enter` |
| **Run the current cell and insert a new one below** | `Alt + Enter` |

### Cell Types

- **Code cells** — have `[ ]:` on the left. Click inside and press `Shift + Enter` to execute.
- **Markdown cells** — contain text/instructions. Run them too to render the formatting.
- A `[*]:` next to a cell means it is **currently running** — wait for it to finish before moving on.
- A `[1]:`, `[2]:` etc. means the cell has **finished running** (the number is the execution order).

### Recommended Workflow

1. Read each cell's description before running it
2. Run cells **top to bottom** — skipping cells can cause errors
3. If a cell errors, check the output message and re-run after fixing
4. For long-running training cells, you'll see a progress bar — sit back and watch the model learn!

### Useful Menu Options

- **Kernel > Restart & Run All** — reruns the entire notebook from scratch (useful if things go sideways)
- **Kernel > Interrupt** — stops a running cell (handy if training is stuck)
- **File > Download** — save the notebook with all outputs to your machine

---

## You're All Set — Happy Finetuning!

> **Tip:** After the workshop, check out [Unsloth Studio](https://unsloth.ai/studio) to fine-tune models with a no-code UI, or explore the [Unsloth blog](https://unsloth.ai/blog) for the latest tricks and benchmarks.
