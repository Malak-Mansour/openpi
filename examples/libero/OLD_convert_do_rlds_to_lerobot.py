"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data
uv run examples/libero/convert_do_rlds_to_lerobot.py --data_dir /home/malak.mansour/tensorflow_datasets

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub
uv run examples/libero/convert_do_rlds_to_lerobot.py --data_dir /home/malak.mansour/tensorflow_datasets --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""


import shutil

# from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
LEROBOT_HOME=HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro
# import rlds

REPO_NAME = "Malak-Mansour/DO_manual_lerobot"  # Name of the output dataset

def main(data_dir: str, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (240, 320, 3),  # from features.json
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (21,), # from features.json: observation.state.shape = [21]: 
                                # "description": "Concatenated robot state: joint_pose (7), qpose_euler (6), qpose_quat (7), tip_state (1)."
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,), # from features.json: action.shape = [7]: right
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    raw_dataset = tfds.load("do_manual", data_dir=data_dir, split="train")


    for episode in raw_dataset:
        steps = list(episode["steps"].as_numpy_iterator())
        # language_instruction = (
        #     steps[0]["language_instruction"].decode() 
        #     if "language_instruction" in steps[0] 
        #     else "do_manual"
        # )
        language_instruction = steps[0]["language_instruction"].decode() 
        

        for step in steps:
            dataset.add_frame(
                {
                    "image": step["observation"]["image"],
                    "state": step["observation"]["state"],
                    "actions": step["action"],
                    "task": language_instruction,  # Use the first step's language instruction
                    # "task": step["language_instruction"].decode(),
                }
            )

        # dataset.save_episode(task=language_instruction)  
        # dataset.save_episode(task=step["language_instruction"].decode())
        dataset.save_episode()  


    # Consolidate the dataset, skip computing stats since we will do that later
    # dataset.consolidate(run_compute_stats=False)

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["do_manual", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )

if __name__ == "__main__":
    tyro.cli(main)
