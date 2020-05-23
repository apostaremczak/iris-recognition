"""
Reorganizes the original UBIRIS v1 dataset structure for easier model training.
"""
import os
import shutil
from glob import glob
from typing import List, Set

INPUT_DIR = "../UBIRIS_800_600_COLOR"
TARGET_DIR = "../data"


def extract_user_ids(input_dir: str) -> Set[int]:
    """
    :param input_dir: Path of the original UBIRIS dataset
    :return: Set of user IDs found in the dataset
    """
    return set(int(user_path.split("/")[-2]) for user_path in glob(input_dir))


def _safely_create_dir(dir_path: str):
    """
    Create a directory if it doesn't exist.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def organize_files(input_dir: str = INPUT_DIR,
                   target_dir: str = TARGET_DIR,
                   user_ids: List[int] = None) -> None:
    """
    :param input_dir:  Path of the original UBIRIS dataset
    :param target_dir: Output directory where the files should be stored
    :param user_ids:   List of user IDs to be organized; If None, the data
                       of all users present at both sessions will be used.

    :return:           Nothing. Files in the target directory will have names
                       in the format "<user-id>_<user-sample-number>.jpg"
    """
    if user_ids is None:
        first_session_ids = extract_user_ids(f"{input_dir}/Sessao_1/*/")
        second_session_ids = extract_user_ids(f"{input_dir}/Sessao_2/*/")
        user_ids = list(first_session_ids.intersection(second_session_ids))

    # Create the target directory
    _safely_create_dir(target_dir)

    for user_id in user_ids:
        pic_count = 0
        target_user_path = f"{target_dir}/{user_id}"

        for session in (1, 2):
            user_path = f"{input_dir}/Sessao_{session}/{user_id}/"

            assert os.path.exists(user_path), \
                f"Missing data for user {user_id} in session {session}"

            pics = filter(lambda file_name: file_name.endswith(".jpg"),
                          glob(user_path + "*"))

            for pic in pics:
                shutil.copyfile(pic, f"{target_user_path}_{pic_count}.jpg")
                pic_count += 1
