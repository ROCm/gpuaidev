# QuickStart 

To set up the lerobot development environment for both edge and cloud with AMD ROCm Platform.

## Setup the Edge Development

The edge platform is based on AMD Ryzen AI PC for driving the SO101-ARM for create dataset and inference. The iGPU of the Ryzen AI Processor do accerate the model inference base on ROCm.

### Target Enviroment

```text
AMD Ryzen AI 9 HX370 PC
OS: ubuntu 24.04
ROCm v6.3+
PyTorch v2.7.x
LeRobot: v0.4.0
```

### Pre-requisites

#### Set the VRAM size to 16GB+ in BIOS

The BIOS UI may be different from vendors. You should refer to the User Guide about the BIOS setting of which PC you use. 

Here are two examples for reference,

```
Enter the BIOS Setup => Advanced => GFX Configuration => UMA Frame buffer Size => 16GB
```

```
Enter the BIOS Setup => Advanced => AMD CBS => NBIO Common Options => GFX Configuration => Dedicated Graphics Memory => 16GB
```

#### Install Ubuntu 24.04 LTS on the AMD Ryzen AI PC

```shell
alex@SER9:~$ lsb_release -a
No LSB modules are available.
Distributor ID: Ubuntu
Description:    Ubuntu 24.04.3 LTS
Release:        24.04
Codename:       noble
alex@SER9:~$ uname -r
6.14.0-27-generic
alex@SER9:~$
alex@SER9:~$ lsmod | grep amdgpu
amdgpu              19714048  1
amdxcp                 12288  1 amdgpu
drm_panel_backlight_quirks    12288  1 amdgpu
drm_buddy              24576  1 amdgpu
drm_ttm_helper         16384  1 amdgpu
ttm                   118784  2 amdgpu,drm_ttm_helper
drm_exec               12288  1 amdgpu
drm_suballoc_helper    20480  1 amdgpu
drm_display_helper    278528  1 amdgpu
cec                    94208  2 drm_display_helper,amdgpu
i2c_algo_bit           16384  1 amdgpu
gpu_sched              61440  2 amdxdna,amdgpu
video                  77824  1 amdgpu
```

### Setup the ROCm Development Environment for the LeRobot

At now [2025/11],  LeRobot depends on PyTorch version >=2.2.1, <2.8.0 (see `pyproject.toml` )

So that we recommend using ROCm 6.3 and PyTorch 2.7 combination to make compatible with LeRobot.

#### Install ROCm 6.3.x,

``` shell
sudo apt update
sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
sudo apt install python3-setuptools python3-wheel
sudo usermod -a -G render,video $LOGNAME # Add the current user to the render and video groups


wget https://repo.radeon.com/amdgpu-install/6.3.4/ubuntu/noble/amdgpu-install_6.3.60304-1_all.deb
sudo apt install ./amdgpu-install_6.3.60304-1_all.deb
amdgpu-install -y --usecase=rocm --no-dkms

sudo reboot
```

**NOTE**: --no-dkms , Do not install dkms and use built-in kernel driver

You could get more details from https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-6.3.4/docs/install/native_linux/install-radeon.html

#### Install PyTorch with ROCm

To repeat
At now [2025/10],  LeRobot depends on PyTorch version >=2.2.1, <2.8.0 (see `pyproject.toml` )

But PyTorch.org supports PyTorch-ROCm latest version is v2.8.0+. So that we need to install the previous-versions of PyTorch-ROCm.


LeRobot github repo using miniconda as example to create the venv for development. But it is use PyTorch-CUDA as default. Let’s follow it to create venv with minor changes in steps for PyTorch-ROCm.

Click Here about How to Install miniconda: 

```shell
conda create -n lerobot python=3.10
conda activate lerobot

pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/rocm6.3

# Check the installation
pip list | grep rocm
pytorch-triton-rocm    3.3.1
torch                  2.7.1+rocm6.3
torchaudio             2.7.1+rocm6.3
torchvision            0.22.1+rocm6.3
```

Then run the iGPU of Ryzen AI 300 series at gfx1100 compatilbe mode by, 

```
echo "export HSA_OVERRIDE_GFX_VERSION=11.0.0" >> ~/.bashrc
source ~/.bashrc
```

Now the iGPU of Ryzen AI APU will worked as a CUDA compatible device within PyTorch.

```shell
python3 -c 'import torch; print(torch.cuda.is_available())'
```
It will get `True`.

```shell
python3 -c "import torch; print(f'device name [0]:', torch.cuda.get_device_name(0))"
```
It will get ` device name [0]: AMD Radeon Graphics`.


#### Setup LeRobot development Environment

Follow the Installation in https://github.com/huggingface/lerobot/blob/main/README.md  

```shell
conda install ffmpeg=7.1.1 -c conda-forge
```

```shell
git clone https://github.com/huggingface/lerobot.git
cd lerobot

# let’s synchronize using this version
git checkout -b v0.4.0 v0.4.0
pip install -e .
```

Check the Installation
```shell
pip list | grep lerobot
lerobot                0.4.0          /home/alex/lerobot
```

Install the feetech-servo-sdk for SO-ARM101 in this Hackation.

```shell
pip install 'lerobot[feetech]'      # Feetech motor support
```

The edge development environment is ready. Please refer the [LeRobot Documentation] (https://huggingface.co/docs/lerobot/index)
for the calibarion, teleop, creating dataset and inference evalution with the SO-101 ARM.



## Setup the training environment


Get start your hackathon challenge with LeRobot. 

Good Luck~


## Reference

[LeRobot Documentation] (https://huggingface.co/docs/lerobot/index)
