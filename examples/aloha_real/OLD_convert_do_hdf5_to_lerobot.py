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
                "shape": (240, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (21,),
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

    for episode_file in sorted(raw_dir.glob("*.hdf5")):
        with h5py.File(episode_file, "r") as f:
            step_keys = sorted([k for k in f.keys() if k.startswith("step_")], key=lambda x: int(x.split("_")[-1]))


            task = f[step_keys[0]].attrs["language_instruction"]

            for step_key in step_keys:
                try:
                    img = f[f"{step_key}/observation/camera/image"][()]
                    joint_pose = f[f"{step_key}/observation/right/joint_pose"][()]
                    qpose_euler = f[f"{step_key}/observation/right/qpose_euler"][()]
                    qpose_quat = f[f"{step_key}/observation/right/qpose_quat"][()]
                    tip_state = f[f"{step_key}/observation/right/tip_state"][()]
                    state = np.concatenate([joint_pose, qpose_euler, qpose_quat, tip_state]).astype(np.float32)
                    action = f[f"{step_key}/action/right"][()].astype(np.float32)

                    dataset.add_frame({
                        "image": img,
                        "state": state,
                        "actions": action,
                        "task": task,  # Use the task from the first step
                    })
                except KeyError as e:
                    print(f"Skipping {step_key} due to missing key: {e}")

            dataset.save_episode()  

    # dataset.consolidate(run_compute_stats=False)  # Commented out per your request

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(convert_hdf5_to_lerobot)
