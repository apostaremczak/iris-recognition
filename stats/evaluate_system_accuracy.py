import pickle
from glob import glob
from tqdm import tqdm

from utils.file_utils import extract_user_sample_ids, copy_dataset
from biometric_system import *

REGISTERED_USERS = "data/system_database/registered_users"
UNKNOWN_USERS = "data/system_database/unknown_users"
MODEL_PATH = "model/iris_recognition_trained_model.pt"


def run_classification_(image_path: str, mode: str, user_id: str = None):
    image = Image(image_path=image_path)

    try:
        image.find_iris_and_pupil()
    except ImageProcessingException:
        return RunResults.PROCESSING_FAILURE, 0

    iris = normalize_iris(image)
    iris.pupil = image.pupil
    iris.iris = image.iris

    # Save the normalized image to a temporary file for easier use with
    # the trained network
    create_empty_dir("tmp")
    iris_hash = hashlib.sha1(image_path.encode()).hexdigest()
    iris_path = f"tmp/{iris_hash}.jpg"
    iris.save(iris_path)

    # Get the classifier's prediction
    predicted_user, probability = classifier.classify_single_image(iris_path)

    if mode == Mode.IDENTIFY:
        if predicted_user == User.UNKNOWN:
            run_result = RunResults.IDENTIFICATION_FAILURE
        else:
            run_result = RunResults.IDENTIFICATION_SUCCESS
    else:
        if predicted_user == User.UNKNOWN:
            run_result = RunResults.VERIFICATION_FAILURE_USER_UNKNOWN
        else:
            if predicted_user == user_id:
                run_result = RunResults.VERIFICATION_SUCCESS
            else:
                run_result = RunResults.VERIFICATION_FAILURE_USER_MISMATCH

    # Remove temporary files
    remove(iris_path)

    return run_result, probability


# User IDs of identified and verified users
stats = {
    "registered": {
        "identified": [],
        "not_identified": [],
        "verified": [],
        "not_verified": [],
        "accepted_probabilities": [],
        "rejected_probabilities": []
    },
    "unknown": {
        "identified": [],
        "accepted_probabilities": [],
        "rejected_probabilities": []
    }
}

registered_user_paths = sorted(glob(REGISTERED_USERS + "/*"))
unknown_user_paths = sorted(glob(UNKNOWN_USERS + "/*"))

# Load trained classifier
classifier = IrisClassifier(load_from_checkpoint=True,
                            checkpoint_file=MODEL_PATH)

for user_path in tqdm(registered_user_paths, desc="Registered users"):
    user_id, _ = extract_user_sample_ids(user_path)

    # Run identification
    identification_result, prob = run_classification_(user_path, Mode.IDENTIFY)

    if identification_result == RunResults.IDENTIFICATION_SUCCESS:
        stats["registered"]["identified"].append(user_path)
        stats["registered"]["accepted_probabilities"].append(
            prob)
    else:
        stats["registered"]["not_identified"].append(user_path)
        stats["registered"]["rejected_probabilities"].append(
            prob)

    # Run verification
    verification_result, prob = run_classification_(user_path, Mode.VERIFY,
                                                    user_id)
    if verification_result == RunResults.VERIFICATION_SUCCESS:
        stats["registered"]["verified"].append(user_path)
    else:
        stats["registered"]["not_verified"].append(user_path)

for user_path in tqdm(unknown_user_paths, desc="Unknown users"):
    user_id, _ = extract_user_sample_ids(user_path)

    # Run identification
    identification_result, prob = run_classification_(user_path, Mode.IDENTIFY)

    if identification_result == RunResults.IDENTIFICATION_SUCCESS:
        stats["unknown"]["identified"].append(user_path)
        stats["unknown"]["accepted_probabilities"].append(prob)
    else:
        stats["unknown"]["rejected_probabilities"].append(prob)

# Save statistics for later usage
with open("stats/model_verification_stats.pickle", "wb") as f:
    pickle.dump(stats, f)

# Copy unrecognized images to a separate folder
not_identified = stats["registered"]["not_identified"]
copy_dataset(not_identified, "data/tmp/incorrectly_classified/registered")

frauds = stats["unknown"]["identified"]
copy_dataset(frauds, "data/tmp/incorrectly_classified/unknown")


def get_unique_ids(paths):
    return set(extract_user_sample_ids(path)[0] for path in paths)


print("REGISTERED USERS:")
identified = stats["registered"]["identified"]
identified_unique = get_unique_ids(identified)
identified_perc = len(identified) / len(registered_user_paths)
print(f"Identified: {len(identified_unique)} ({identified_perc:.2%})")

verified = stats["registered"]["verified"]
verified_unique = get_unique_ids(verified)
verified_perc = len(verified) / len(registered_user_paths)
print(f"Verified: {len(verified_unique)} ({verified_perc:.2%})")

print("UNKNOWN USERS:")
u_identified = stats["unknown"]["identified"]
u_identified_unique = get_unique_ids(u_identified)
u_identified_perc = len(u_identified) / len(unknown_user_paths)
print(f"Identified: {len(u_identified_unique)} ({u_identified_perc:.2%})")
