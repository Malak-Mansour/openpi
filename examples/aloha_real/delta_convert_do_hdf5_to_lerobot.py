"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
"""
"""
Converts DO_manual HDF5 demonstrations into LeRobot dataset format.

Example usage:
uv run examples/aloha_real/convert_do_hdf5_to_lerobot.py --raw_dir /home/malak.mansour/recorded_data_hdf5 --repo_id Malak-Mansour/DO_manual_lerobot
"""
import h5py
import numpy as np
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
import tyro
import shutil
import os

def convert_hdf5_to_lerobot(raw_dir: Path, repo_id: str, push_to_hub: bool = False):
    dataset_path = HF_LEROBOT_HOME / repo_id
    if dataset_path.exists():
        shutil.rmtree(dataset_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=10,
        features={
            "image": {
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

    for h5_file in sorted(raw_dir.glob("*.h5")):
        with h5py.File(h5_file, "r") as f:
            for ep_key in f.keys():
                if not ep_key.startswith("episode_"):
                    continue

                group = f[ep_key]
                obs = group["obs"]
                agentview_rgb = obs["agentview_rgb"][:]
                eye_in_hand_rgb = obs["eye_in_hand_rgb"][:]
                ee_states = obs["ee_states"][:]
                gripper_states = obs["gripper_states"][:]

                states = np.concatenate([ee_states, gripper_states], axis=-1)
                task = os.path.splitext(h5_file.name)[0].replace("_", " ")

                for i in range(len(states)):
                    state = states[i].astype(np.float32)
                    if i < len(states) - 1:
                        next_state = states[i + 1].astype(np.float32)
                        delta_action = next_state - state
                    else:
                        delta_action = np.zeros(7, dtype=np.float32)

                    dataset.add_frame({
                        "image": agentview_rgb[i],
                        "wrist_image": eye_in_hand_rgb[i],
                        "ee_states": ee_states[i].astype(np.float32),
                        "gripper_states": gripper_states[i].astype(np.float32),
                        "state": state,
                        "actions": delta_action,
                        "task": task,
                    })

                dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub()

if __name__ == "__main__":
    tyro.cli(convert_hdf5_to_lerobot)
