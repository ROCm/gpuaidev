# Example Commands to Operate SO101 ARM

Suppose:
1. You have set up the Ubuntu + ROCm + PyTorch + LeRobot development environment by following [QuickStart.md](QuickStart.md).
2. You have the SO101 ARM assembled.

Here are some key steps with example commands to use the SO101 ARM.

These examples are based on [LeRobot Tutorial](https://huggingface.co/docs/lerobot/so101) and LeRobot v0.4.1 with some modifications and comments for our setup. YOU MAY NEED TO MAKE SOME MODIFICATIONS FOR YOUR JOBS AS REQUIRED.

## Connect the SO101 ARM
1. Connect the leader ARM with USB UART to PC first and it will get `/dev/ttyACM0` on Ubuntu
2. Connect the follower ARM with USB UART to PC and it will get `/dev/ttyACM1` on Ubuntu

The sequence of the connection of leader ARM and follower ARM will result in different device node names. The following steps with commands are based on:

```text
leader ARM => /dev/ttyACM0
follower ARM => /dev/ttyACM1
```

LeRobot provides the command `lerobot-find-port` to help find the UART device node of the SO101 ARM.

## Connect the Cameras
Suppose you have two cameras, one named `top` and another named `side`. The `top` camera may be set up to give a bird's eye view of the ARM's workspace. The `side` camera may be set up to give a side view.

1. Please connect the `top` camera first and it will get `/dev/video0` for it.
2. Then connect the `side` camera and it will get `/dev/video2` for it.

Use the lerobot-find-cameras CLI tool to detect available cameras

```shell
lerobot-find-cameras opencv      # Find OpenCV cameras  
```

## Calibrate the SO101 ARM

Refer to the [calibrate video](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/calibrate_so101_2.mp4) and steps at https://huggingface.co/docs/lerobot/so101

Follower:

```shell
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \ # <- The port of your robot
    --robot.id=my_awesome_follower_arm # <- Give the robot a unique name
```

Leader:

```shell
lerobot-calibrate \
    --robot.type=so101_leader \
    --robot.port=/dev/ttyACM0 \ # <- The port of your robot
    --robot.id=my_awesome_leader_arm # <- Give the robot a unique name
```

Then you can use the SO101 ARM 

## Teleoperate

```shell
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_awesome_follower_arm \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=my_awesome_leader_arm
```

## Teleoperate with cameras

```shell
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

`top` camera with index_or_path 0 (/dev/video0)
`side` camera with index_or_path 2 (/dev/video2)

## Record the dataset

We will use the leader ARM to teleoperate the follower ARM to perform the actions we want to record into the dataset.

The example command from the Hugging Face tutorial [here](https://huggingface.co/docs/lerobot/il_robots) will upload the dataset to Hugging Face when you log in to it. 

Here is the login procedure copied from the Hugging Face tutorial:

```text
Once you're familiar with teleoperation, you can record your first dataset.

We use the Hugging Face hub features for uploading your dataset. If you haven't previously used the Hub, make sure you can login via the cli using a write-access token. This token can be generated from the Hugging Face settings.

Add your token to the CLI by running this command:

$ huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential

Then store your Hugging Face repository name in a variable:

$ HF_USER=$(hf auth whoami | head -n 1)
$ echo $HF_USER
```

Dataset uploading can be disabled by `--dataset.push_to_hub=False`.

```shell
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=alexhegit/record-test \
    --dataset.num_episodes=60 \
    --dataset.episode_time_s=20 \
    --dataset.reset_time_s=10 \
    --dataset.single_task="pickup the cube and place it to the bin" \
    --dataset.root=${HOME}/so101_dataset/ \
    --dataset.push_to_hub=False # =True means to upload the dataset to HuggingFace in your space
```

`--dataset.num_episodes=60` means we will record 60 teleoperation sessions.
`--dataset.episode_time_s=20` means each episode has 20 seconds; this depends on whether it is enough time for your actions.
`--dataset.reset_time_s=10` means the reset time between episodes. You may use this time slot to reset your environment, like recovering the position of the cube to the source by hand and waiting to start the next episode recording.
`--dataset.root=${HOME}/stack2cube_dataset` means where your dataset will be saved.

The terminal has logs to notify you when new episodes start, reset, and when the dataset is recorded.

You can use `Ctrl-c` to stop the recording. Use `--resume=true` in the command to continue the dataset recording with the num_episodes added.

After the recording is done, you can use the dataset for training.

## Training

Refer to the [QuickStart.md](QuickStart.md) to do the training with MI300X on AMD Development Cloud

Suppose you train with `act` policy and save the model at local dir defined by `--output_dir=outputs/train/act_so101_test` with the commands,

```shell
lerobot-train \
  --dataset.repo_id=${HF_USER}/so101_test \
  --dataset.root=${HOME}/lerobot_dataset/ \
  --policy.type=act \
  --output_dir=outputs/train/act_so101_test \
  --job_name=act_so101_test \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.repo_id=${HF_USER}/my_policy \
  --policy.push_to_hub=false
```

To disable model uploading, use `--policy.push_to_hub=false`.

The checkpoints are generated in `./outputs/train/act_so101_test/checkpoints/` and the last one is `./outputs/train/act_so101_test/checkpoints/last/pretrained_model/`

## Inference Evaluation

Copy the `pretrained_model` under `./outputs/train/act_so101_test/` from cloud back to the Edge platform (PC) for inference evaluation.

```
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=my_awesome_leader_arm \
  --robot.cameras="{top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
  --dataset.single_task="Pick cube from source position and stack it on the fixed cube at target position" \
  --dataset.repo_id=alexhegit/eval_act_base \
  --dataset.root=${PWD}/eval_lerobot_dataset/ \
  --dataset.episode_time_s=20 \
  --dataset.num_episodes=1 \
  --policy.path=${PWD}/outputs/train/act_so101_test/checkpoints/last/pretrained_model/ \ # path to the pretrained_model
  --dataset.push_to_hub=false
```

Now you have finished the full round of work. Then you can continue the next round from `record dataset` => `training` => `inference evaluation`