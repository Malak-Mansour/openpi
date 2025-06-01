"""
uv run examples/libero/convert_do_rlds_to_lerobot.py --data_dir /l/users/malak.mansour/Datasets/do_manual/rlds --push_to_hub
"""

import os
import shutil
import tensorflow_datasets as tfds
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
import tyro
import numpy as np

REPO_NAME = "Malak-Mansour/DO_manual_lerobot"
LEROBOT_HOME = HF_LEROBOT_HOME

def main(data_dir: str, *, push_to_hub: bool = False):
    output_path = LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "image": {  # agentview_rgb
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "ee_states": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["ee_states"],
            },
            "gripper_states": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_states"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    raw_dataset = tfds.load("do_manual", data_dir=data_dir, split="train")

    for episode in raw_dataset:
        steps = list(episode["steps"].as_numpy_iterator())
        language_instruction = steps[0]["language_instruction"].decode()

        for i in range(len(steps)):
            step = steps[i]
            state = np.concatenate([
                step["observation"]["ee_states"],
                step["observation"]["gripper_states"],
            ]).astype(np.float32)

            # If not last step, compute delta to next state
            if i < len(steps) - 1:
                next_step = steps[i + 1]
                next_state = np.concatenate([
                    next_step["observation"]["ee_states"],
                    next_step["observation"]["gripper_states"],
                ]).astype(np.float32)
                delta_action = next_state - state
            else:
                delta_action = np.zeros(7, dtype=np.float32)  # No motion at last step

            dataset.add_frame(
                {
                    "image": step["observation"]["agentview_rgb"],
                    "wrist_image": step["observation"]["eye_in_hand_rgb"],
                    "ee_states": step["observation"]["ee_states"],
                    "gripper_states": step["observation"]["gripper_states"],
                    "state": state,
                    "actions": delta_action,
                    "task": language_instruction,
                }
            )

        dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub(
            tags=["do_manual", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )

if __name__ == "__main__":
    tyro.cli(main)
