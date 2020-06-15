"""
Reorganizes the original UBIRIS v1 dataset structure for easier image
processing.
"""
import os
import shutil
from glob import glob
from typing import List, Set

INPUT_DIR = "../data/tmp/UBIRIS_800_600_COLOR"
TARGET_DIR = "../data/tmp/original_renamed"


def _extract_user_ids(input_dir: str) -> Set[int]:
    """
    :param input_dir: Path of the original UBIRIS dataset
    :return: Set of user IDs found in the dataset
    """
    return set(int(user_path.split("/")[-2]) for user_path in glob(input_dir))


def organize_files(input_dir: str = INPUT_DIR,
                   target_dir: str = TARGET_DIR,
                   user_ids: List[int] = None) -> None:
    """
    :param input_dir:  Path of the original UBIRIS dataset
    :param target_dir: Output directory where the files should be stored
    :param user_ids:   List of user IDs to be organized; If None, the data
                       of all users present at both sessions will be used.

    :return:           Nothing. Files in the target directory will have names
                       in the format "<user-id>_<user-sample-number>.jpg".

    Note: If the target directory already exists and has some data in it,
    the data will be deleted before copying the new dataset.
    """
    if user_ids is None:
        first_session_ids = _extract_user_ids(f"{input_dir}/Sessao_1/*/")
        second_session_ids = _extract_user_ids(f"{input_dir}/Sessao_2/*/")
        user_ids = list(first_session_ids.intersection(second_session_ids))

    # Create an empty target directory or remove data already present
    create_empty_dir(target_dir)

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


if __name__ == '__main__':
    organize_files()
