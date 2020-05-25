import random

from glob import glob
from shutil import copyfile
from typing import List

from file_organizer import create_empty_dir
from preprocessing import extract_user_sample_ids


def copy_dataset(file_paths: List[str], target_dir: str):
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
    test_dir = f"{target_dir}/test"
    create_empty_dir(test_dir)

    for user_id, user_samples in samples.items():
        indices = list(range(len(user_samples)))
        random.shuffle(indices)

        split = int(len(user_samples) * (1.0 - test_size))
        train = [user_samples[i] for i in indices[:split]]
        test = [user_samples[i] for i in indices[split:]]

        copy_dataset(train, train_dir)
        copy_dataset(test, test_dir)


if __name__ == '__main__':
    image_paths = sorted(glob("../normalized_data/*"))
    assign_train_test(image_paths, "../normalized_data_splitted")
