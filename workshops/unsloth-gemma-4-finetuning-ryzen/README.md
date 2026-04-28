# Unsloth AMD Finetuning Workshop

A step-by-step guide to set up your environment and run the finetuning script.

---

## Prerequisites

- Ubuntu 22.04 / 24.04 machine with an AMD GPU
- Internet access
- Credentials
  ```bash
  Username: amd-user
  Password: amd1234
  ```

---

## Step 1 — Check ROCm Installation
> Full reference: [Install Ryzen Software for Linux with ROCm](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/native_linux/install-ryzen.html)

ROCm has been already installed on the **host machine**

Check if ROCm is already installed:

```bash
rocminfo
```

If the command is not found, install from [Install Ryzen Software for Linux with ROCm](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/native_linux/install-ryzen.html).


## Step 2 — Start the container
If a container is already running, recommend to kill/remove it 
```bash
docker kill workshop-env
docker rm  workshop-env
```

Run the container with your home directory mounted so files persist between sessions:

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

> **What the flags do:**
> - `--device=/dev/kfd` and `--device=/dev/dri` — exposes your AMD GPU to the container
> - `--group-add video --group-add render` — grants GPU access permissions
> - `-v $HOME:/root/home` — mounts your home directory so your files are accessible
> - `-v $PWD:/workspace` — mounts the current folder (where `finetune.py` lives) into `/workspace`
> - `-w /workspace` — sets the working directory inside the container

You should now be inside the container shell at `/workspace`.

---

## Step 3 — Install Notebook to kick start your workshop (inside the container)

Run these three commands in order:

```bash
pip install notebook ihighlight
```

---

## Step 4 — Configure your finetuning run

```bash
cd /workspace

curl -L -o "Gemma4_(E2B)_Reinforcement_Learning_Sudoku_Game.ipynb" "https://raw.githubusercontent.com/iswaryaalex/Unsloth-RL-Workshop-on-Radeon/main/Gemma4_(E2B)_Reinforcement_Learning_Sudoku_Game.ipynb"
```
### Start your Jupyter Notebook
```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### You are all set!
