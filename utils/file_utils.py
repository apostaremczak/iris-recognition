import os
import re
from glob import glob
from shutil import copyfile
from typing import List, Tuple

FILENAME_REGEX = r"(\d+)_(\d+)\.jpg"


def get_file_name(file_path: str) -> str:
    return file_path.split("/")[-1]


def extract_user_sample_ids(image_path: str) -> Tuple[str, str]:
    """
    Extract user ID and sample number from filename
    """
    image_filename = get_file_name(image_path)
    user_id, sample_id = re.match(FILENAME_REGEX, image_filename).groups()
    return user_id, sample_id


def create_empty_dir(target_dir: str):
    """
    Create an empty directory, removing all files first if the target directory
    already exists.
    """
    if os.path.exists(target_dir):
        files = glob(target_dir + "/*")
        for file in files:
            os.remove(file)
    else:
        os.makedirs(target_dir)


def copy_dataset(file_paths: List[str], target_dir: str):
    for file_path in file_paths:
        file_name = get_file_name(file_path)
        copyfile(file_path, f"{target_dir}/{file_name}")
