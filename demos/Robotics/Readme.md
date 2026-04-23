# LeRobot Setup Guide

This guide provides step-by-step instructions for setting up robotic arms using [LeRobot](https://github.com/huggingface/lerobot) with AMD ROCm support.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Arm Configuration](#arm-configuration)
- [Camera Setup](#camera-setup)
- [Teleoperation](#teleoperation)
- [Dataset Recording](#dataset-recording)
- [Training a Policy](#training-a-policy)
- [Model Inference and Evaluation](#model-inference-and-evaluation)

## Prerequisites

### Conda Installation (Miniforge)
If you don't have conda installed, get it using [Miniforge](https://github.com/conda-forge/miniforge)

```bash
bash Miniforge3-Linux-x86_64.sh
```

Restart your terminal after installation

### ROCm Installation
Install ROCm following the [official AMD documentation](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/native_linux/install-ryzen.html).

## Environment Setup

### 1. Create Conda Environment
```bash
conda create -n lerobot python=3.12
conda activate lerobot
```

### 2. Install PyTorch with ROCm Support
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2
```

**Verify installation:**
```bash
pip list | grep rocm
```

**Expected output:**
```
torch             2.11.0+rocm7.2
torchaudio        2.11.0+rocm7.2
torchvision       0.26.0+rocm7.2
triton-rocm       3.6.0
```

**Test GPU detection:**
```bash
python3 -c "import torch; print(f'device name [0]:', torch.cuda.get_device_name(0))"
```

**Expected output:**
```
device name [0]: AMD Radeon 8060S
```

### 3. Install FFmpeg
```bash
conda install ffmpeg -c conda-forge
```

### 4. Install LeRobot
```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .
```

**Verify installation:**
```bash
pip list | grep lerobot
```

**Expected output:**
```
lerobot    0.5.2    /home/amd-user/robotics/lerobot
```

### 5. Install Additional Dependencies
```bash
pip install 'lerobot[feetech]'
pip install 'lerobot[viz]'
pip install 'lerobot[dataset]'
```

### Additional Resources
- [LeRobot GitHub Repository](https://github.com/huggingface/lerobot/blob/main/README.md)
- [SO-101 Robot Arm Documentation](https://huggingface.co/docs/lerobot/so101)

## Arm Configuration

For detailed arm setup instructions, refer to the [SO-101 Documentation](https://huggingface.co/docs/lerobot/so101).

> **Important:** Before proceeding, ensure you have connected the power adapters correctly:
> - **Follower arm**: 12V adapter
> - **Leader arm**: 5V adapter

### 1. Identify USB Ports

Run the port detection utility:
```bash
lerobot-find-port
```

The tool will prompt you to remove and reconnect each USB device. **Example output:**
```
Follower: /dev/ttyACM1
Leader: /dev/ttyACM0
```

### 2. Grant USB Port Permissions
```bash
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
```

### 3. Calibrate Arms

> **Video Guide:** Watch the [calibration video tutorial](https://huggingface.co/docs/lerobot/so101#calibration-video) for a visual walkthrough of the calibration process.

#### Follower Arm Calibration

```bash
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_awesome_follower_arm
```

#### Leader Arm Calibration

```bash
lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=my_awesome_leader_arm
```

## Camera Setup

### 1. Detect Available Cameras

```bash
lerobot-find-cameras opencv
```

**Example output:**
```
--- Detected Cameras ---
Camera #0:
  Name: OpenCV Camera @ /dev/video0
  Type: OpenCV
  Id: /dev/video0
  Backend api: V4L2
  Default stream profile:
    Format: 0.0
    Fourcc: YUYV
    Width: 640
    Height: 480
    Fps: 30.0
--------------------
Camera #1:
  Name: OpenCV Camera @ /dev/video2
  Type: OpenCV
  Id: /dev/video2
  Backend api: V4L2
  Default stream profile:
    Format: 0.0
    Fourcc: YUYV
    Width: 640
    Height: 480
    Fps: 30.0
--------------------
```

### 2. Preview Camera Output

Test camera feed using FFmpeg:
```bash
ffplay /dev/video0  # Replace with your camera device
```
	


## Teleoperation

### Basic Teleoperation (No Cameras)

Control the follower arm using the leader arm:

```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_awesome_follower_arm \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=my_awesome_leader_arm
```

### Teleoperation with Cameras

Enable visual feedback during teleoperation:

```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true
```

**Note:** Adjust camera indices (`0` and `2`) based on your camera detection results.
    
    
## Dataset Recording

For detailed instructions, see the [Record a Dataset Reference](https://huggingface.co/docs/lerobot/il_robots#record-a-dataset).

> **Tip:** Learn about [what makes a good dataset](https://huggingface.co/blog/lerobot-datasets#what-makes-a-good-dataset) to improve your training results.

### Hugging Face Setup

Before recording datasets, authenticate with Hugging Face:

#### 1. Login to Hugging Face
```bash
huggingface-cli login
```

#### 2. Get Your Username
```bash
export HF_USER=$(huggingface-cli whoami | grep '^  user:' | awk '{print $2}')
echo $HF_USER
```

### Record Dataset

Record robot demonstrations with cameras:

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/stack-cubes \
    --dataset.num_episodes=5 \
    --dataset.episode_time_s=20 \
    --dataset.reset_time_s=10 \
    --dataset.single_task="pickup the cube and place it on another cube" \
    --dataset.root=${HOME}/so101_dataset/
```

Your dataset is stored locally in the following directory:
```
~/.cache/huggingface/lerobot/{repo-id}
```

For example, if your `repo_id` is `${HF_USER}/stack-cubes`, the dataset will be at:
```
~/.cache/huggingface/lerobot/${HF_USER}/stack-cubes
```

At the end of data recording, your dataset will be automatically uploaded to your Hugging Face profile and will be accessible at:
```
https://huggingface.co/datasets/${HF_USER}/stack-cubes
```

**Resume Recording:**

To resume an interrupted recording session, add the `--resume=true` flag:
```bash
lerobot-record ... --resume=true
```

---

## Training a Policy

For detailed training instructions, see the [Train a Policy Reference](https://huggingface.co/docs/lerobot/il_robots#train-a-policy).

### AMD Development Cloud for Training

For high-performance training, we recommend using **AMD MI300X GPUs** on the [AMD Development Cloud](https://www.amd.com/en/registration/ai-dev-program-sign-up-form.html), which provides access to premium data center–grade hardware along with $100 in cloud credits when you join the AI Developer Program

### Training Command

After collecting your dataset, train a policy to control your robot using the `lerobot-train` script:

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/stack-cubes \
  --batch_size=64 \
  --steps=20000 \
  --save_freq=5000 \
  --policy.type=act \
  --output_dir=outputs/train/act_stack_cubes \
  --job_name=act_stack_cubes \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.push_to_hub=true \
  --policy.repo_id=${HF_USER}/act_stack_cubes_policy
```

### Checkpoints

Training checkpoints are saved in the output directory:
```
./outputs/train/act_stack_cubes/checkpoints/
```

The most recent checkpoint is always available at:
```
./outputs/train/act_stack_cubes/checkpoints/last/pretrained_model/
```

---

## Model Inference and Evaluation

After training your policy, you can test it by running inference on the robot. The `lerobot-record` script can be used with a trained policy checkpoint to evaluate performance and record evaluation episodes.

### Running Inference

Use the same `lerobot-record` command with the addition of the `--policy.path` argument pointing to your trained model:

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras="{top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
  --display_data=true \
  --dataset.repo_id=${HF_USER}/eval_stack_cubes \
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=20 \
  --dataset.single_task="pickup the cube and place it on another cube" \
  --dataset.root=${HOME}/eval_so101_dataset/ \
  --policy.path=outputs/train/act_stack_cubes/checkpoints/last/pretrained_model \
  --dataset.push_to_hub=false

```

## License

This project uses [LeRobot](https://github.com/huggingface/lerobot), which is licensed under Apache 2.0.
