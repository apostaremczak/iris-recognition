import random
from utils.file_utils import *

NORMALIZED_DATA_DIR = "../data/tmp/normalized_all"
OUTPUT_TEST_VAL_DIR = "../data/tmp/normalized_splitted"


def copy_to_train_val(normalized_images_dir: str = NORMALIZED_DATA_DIR,
                      target_dir: str = OUTPUT_TEST_VAL_DIR,
                      test_size: float = 0.2,
                      random_state: int = 1):
    random.seed(random_state)

    assert 0 < test_size < 1.0
    samples = {}

    normalized_images_paths = sorted(glob(f"{normalized_images_dir}/*"))

    for path in normalized_images_paths:
        user_id, _ = extract_user_sample_ids(path)
        user_samples = samples.get(user_id, [])
        user_samples.append(path)
        samples[user_id] = user_samples

    train_dir = f"{target_dir}/train"
    create_empty_dir(train_dir)
    val_dir = f"{target_dir}/val"
    create_empty_dir(val_dir)

    for user_id, user_samples in samples.items():
        indices = list(range(len(user_samples)))
        random.shuffle(indices)

        split = int(len(user_samples) * (1.0 - test_size))
        train = [user_samples[i] for i in indices[:split]]
        val = [user_samples[i] for i in indices[split:]]

        train_user_dir = f"{train_dir}/{user_id}"
        create_empty_dir(train_user_dir)
        copy_dataset(train, train_user_dir)

        val_user_dir = f"{val_dir}/{user_id}"
        create_empty_dir(val_user_dir)
        copy_dataset(val, val_user_dir)


if __name__ == '__main__':
    copy_to_train_val()
