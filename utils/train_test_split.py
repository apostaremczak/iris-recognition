import random

from glob import glob
from shutil import copyfile
from typing import List

from file_organizer import create_empty_dir
from preprocessing import extract_user_sample_ids


NORMALIZED_DATA_DIR = "../data/normalized"
OUTPUT_TEST_VAL_DIR = "../data/normalized_splitted"


def copy_dataset(file_paths: List[str], target_dir: str):
    create_empty_dir(target_dir)
    for file_path in file_paths:
        file_name = file_path.split("/")[-1]
        copyfile(file_path, f"{target_dir}/{file_name}")


def assign_train_test(file_paths: List[str],
                      target_dir: str,
                      test_size: float = 0.2,
                      random_state: int = 1):
    random.seed(random_state)

    assert 0 < test_size < 1.0
    samples = {}

    for path in file_paths:
        user_id, _ = extract_user_sample_ids(path)
        user_samples = samples.get(user_id, [])
        user_samples.append(path)
        samples[user_id] = user_samples

    train_dir = f"{target_dir}/train"
    create_empty_dir(train_dir)
    test_dir = f"{target_dir}/val"
    create_empty_dir(test_dir)

    for user_id, user_samples in samples.items():
        indices = list(range(len(user_samples)))
        random.shuffle(indices)

        split = int(len(user_samples) * (1.0 - test_size))
        train = [user_samples[i] for i in indices[:split]]
        test = [user_samples[i] for i in indices[split:]]

        copy_dataset(train, f"{train_dir}/{user_id}")
        copy_dataset(test, f"{test_dir}/{user_id}")


if __name__ == '__main__':
    image_paths = sorted(glob(f"{NORMALIZED_DATA_DIR}/*"))
    assign_train_test(image_paths, OUTPUT_TEST_VAL_DIR)
