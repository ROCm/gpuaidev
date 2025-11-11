# Example in command to operate with SO101 ARM

Suppose
1. You have setup the Ubuntu + ROCm + PyTorch + LeRobot development environment by following [QuickStart.md](QuickStart.md).
2. You have the O101 ARM assembled.

Here we go some key steps with example command to use SO101 ARM.

These examples are based on [LeRobot Tutorial](https://huggingface.co/docs/lerobot/so101) and LeRobot v0.4.0 with some modification and comments with our setup. YOU MAY MAKE SOME MODIFICATION FOR YOU JOBS AS REQUIRED.

## Connect the SO101 ARM
1. Connect the leader ARM with USB UART to PC fisrt and it will get `/dev/ttyACM0` on Ubuntu
2. Connect the follwer ARM with USB UART to PC fisrt and it will get `/dev/ttyACM1` on Ubuntu

The seuqence of the connection of leader ARM and follower ARM will get different devcie node name. The following steps with commands are based on,

```text
leader ARM => /dev/ttyACM0
follower ARM => /dev/ttyACM1
```

LeRobot provide the command `lerobot-find-port` to help find out the UART device node of the SO101 ARM.

## Connect the SO101 ARM
Suppose you have two cameras, one named `top` and another named `side`. The `top` camera may set up to give a bird view of the ARM's workspace. The `side` may set up to give a side view.

1. Please connect the `top` camera first and will get `/dev/video0` for it.
2. Then connect the `side` camera and will get `/dev/video2` for it.

Use the lerobot-find-cameras CLI tool to detect available cameras

```shell
lerobot-find-cameras opencv      # Find OpenCV cameras  
```

## Calibrate the SO101 ARM

Refer to the [calibrate video](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/calibrate_so101_2.mp4) and steps in https://huggingface.co/docs/lerobot/so101

Follower

```shell
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \ # <- The port of your robot
    --robot.id=my_awesome_follower_arm # <- Give the robot a unique name
```

```shell
lerobot-calibrate \
    --robot.type=so101_Leader \
    --robot.port=/dev/ttyACM0 \ # <- The port of your robot
    --robot.id=my_awesome_follower_arm # <- Give the robot a unique name
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

We will use Learder ARM teleoperate the Follower ARM do the action we want to be record into the dataset.

The exmaple command of Huggingface tutorial [here](https://huggingface.co/docs/lerobot/il_robots) will upload the dataset to Hugggingface with you logon it. 

Here is the how-to login copied from the Huggingface tutroial.

```text
Once you’re familiar with teleoperation, you can record your first dataset.

We use the Hugging Face hub features for uploading your dataset. If you haven’t previously used the Hub, make sure you can login via the cli using a write-access token, this token can be generated from the Hugging Face settings.

Add your token to the CLI by running this command:

$ huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential

Then store your Hugging Face repository name in a variable:

$ HF_USER=$(hf auth whoami | head -n 1)
$ echo $HF_USER

```

The dataset uploading could be disabled by `--dataset.push_to_hub=False`.

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
    --dataset.push_to_hub=False # =True means to uplaod the dataset to HuggingFace in your space
```

`--dataset.num_episodes=60` means we will record 60 times teleoperate.
`--dataset.episode_time_s=20` means each episodes have 20 seconde, it depends on if it is enought for you actions.
`--dataset.reset_time_s=10` means the reset time between the episodes, you may use this time slot to reset you environment like recover the position of the cube to the source by hand and waiting to start the next episodes record. 
`--dataset.root=${HOME}/stack2cube_dataset` means where you dataset be saved


The terminal has the log to notice you when the new episodes start, reset and dataset be record.

You can use `Ctrl-c` to stop the recording. Use `--resume=true` in the command to continue the dataset recording with the num_episodes added.

After the recording done. You could use the dataset for training.

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

To disable the model uploading by `--policy.push_to_hub=false`.


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

Now you finishe the full round work. Then you can continue the next round from `record dataset` => `trainng` => `inference evaluation`