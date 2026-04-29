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

#### Follow these steps if you're on Windows.
1. Get WSL with Ubuntu 24.04:
   ```bash
	wsl --install -d Ubuntu-24.04
   ```
3. Get [AMD Software: Adrenalin Edition™](https://www.amd.com/en/support/download/drivers.html)
4. Get [Windows SDK](https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/)
5. For all the following instructions, switch to WSL by running `WSL` in Powershell

### Conda Installation (Miniforge)
If you don't have conda installed, get it using [Miniforge](https://github.com/conda-forge/miniforge)

```bash
bash Miniforge3-Linux-x86_64.sh
```

Restart your terminal after installation

### ROCm Installation

#### Linux
Install ROCm following the [official AMD documentation](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/native_linux/install-ryzen.html).

#### Windows - WSL
1. Install ROCm following the [official AMD documentation](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/native_linux/install-ryzen.html).
2. Run these commands
```bash
sudo apt update -qq && sudo apt install -y cmake build-essential git && \
cd ~ && rm -rf librocdxg && git clone https://github.com/ROCm/librocdxg.git && \
cd librocdxg && git checkout develop && \
mkdir -p build && cd build && \
cmake .. -DWIN_SDK="/mnt/c/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0/shared" && \
make -j$(nproc) && sudo make install && \
grep -q "HSA_ENABLE_DXG_DETECTION" ~/.bashrc || echo 'export HSA_ENABLE_DXG_DETECTION=1' >> ~/.bashrc && \
export HSA_ENABLE_DXG_DETECTION=1

# Configure WSL memory allocation 
cat > /mnt/c/Users/$USER/.wslconfig << 'EOF'
[wsl2]
memory=<PORTION_OF_TOTAL_RAM>GB
EOF

# Restart your terminal.
# Verify setup - GPU should appear in rocminfo output
rocminfo
```

## Environment Setup

### 1. Create Conda Environment
```bash
conda create -n lerobot python=3.12
conda activate lerobot
```
**For Windows - WSL,** make sure to also install evdev:
```bash
conda install evdev -c conda-forge
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

### Follow these additional steps if you're on Windows.

**Install usbipd (Windows PowerShell as Administrator)**
```bash
winget install --id dorssel.usbipd-win -e
```

**Find your BUSIDs**
```powershell
usbipd list
```
 
Example output:
```
BUSID  VID:PID    DEVICE                                 STATE
8-1    1a86:55d3  USB-Enhanced-SERIAL CH343 (COM3)       Attached
8-2    1a86:55d3  USB-Enhanced-SERIAL CH343 (COM4)       Shared (forced)
```
> MotorBus = `USB-Enhanced-SERIAL` or similar USB Serial device  
 
**Bind each device**
```powershell
usbipd bind --force --busid <BUSID>
```

Example output:
```
# usbipd bind --force --busid 8-1
# usbipd bind --force --busid 8-2
```
 
**Attach to WSL**
```powershell
usbipd attach --wsl --busid <BUSID>
```

Example output:
```
# usbipd attach --wsl --busid 8-1
# usbipd attach --wsl --busid 8-2
```
 
**Verify in WSL:**
```bash
ls -l /dev/ttyACM*
```

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

### Follow these additional steps if you're on Windows.

**Find your BUSIDs**
```powershell
usbipd list
```
 
Example output:
```
BUSID  VID:PID    DEVICE                                 STATE
8-1    05a3:9230  USB2.0_CAM1                            Attached
8-2    0c45:2283  UGREEN Camera                          Attached
```
**Note:** You might see a different set of BusIDs depending on your motor connections

> Camera = `USB2.0_CAM`, `UGREEN Camera`, or similar device
 
**Bind each device**
```powershell
usbipd bind --force --busid <BUSID>
```

Example output:
```
# usbipd bind --force --busid 8-1
# usbipd bind --force --busid 8-2
```
 
**Attach to WSL**
```powershell
usbipd attach --wsl --busid <BUSID>
```

Example output:
```
# usbipd attach --wsl --busid 8-1
# usbipd attach --wsl --busid 8-2
```
 
**Verify in WSL:**
```bash
ls -l /dev/video*
```
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

#### Linux
```bash
ffplay /dev/video0  # Replace with your camera device
```

#### Windows - WSL
`ffplay` has no display in WSL. Use ffmpeg to save a snapshot instead:
```bash
/usr/bin/ffmpeg -f v4l2 -input_format mjpeg -i /dev/video0 -update 1 -frames:v 1 test.jpg && explorer.exe test.jpg
```

**IMPORTANT:** Cameras must use **MJPEG** in WSL. YUYV (the default) does not stream through usbipd. When configuring cameras in lerobot, set `fourcc="MJPG"`:
```python
OpenCVCameraConfig(0, 30, 640, 480, fourcc="MJPG")
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

#### Linux
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

#### Windows - WSL
```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
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
hf auth login
```

#### 2. Get Your Username
```bash
export HF_USER=$(huggingface-cli whoami | grep '^  user:' | awk '{print $2}')
echo $HF_USER
```

### Record Dataset

Record robot demonstrations with cameras:

#### Linux
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

#### Windows - WSL
```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
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

For high-performance training, we recommend using **AMD MI300X GPUs** on the [AMD Developer Cloud](https://www.amd.com/en/registration/ai-dev-program-sign-up-form.html), which provides access to premium data center–grade hardware along with $100 in cloud credits when you join the AI Developer Program

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

#### Linux
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

#### Windows
```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras="{top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
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
