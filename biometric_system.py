import argparse
import hashlib
from dataclasses import dataclass
from enum import Enum

from model.iris_classifier_model import IrisClassifier
from utils.file_organizer import create_empty_dir
from utils.image import Image
from utils.preprocessing_exceptions import ImageProcessingException
from utils.preprocessing import normalize_iris


class Mode:
    IDENTIFY: str = "identify"
    VERIFY: str = "verify"


class User:
    UNKNOWN = "unknown"


@dataclass
class ProgramResult:
    message: str
    code: int


class RunResults(Enum):
    PROCESSING_FAILURE = ProgramResult("Failed to process the image", 1)
    IDENTIFICATION_SUCCESS = ProgramResult("Successfully identified a user", 0)
    IDENTIFICATION_FAILURE = ProgramResult("Could not identify a user", 1)
    VERIFICATION_SUCCESS = ProgramResult("Successfully verified a user", 0)
    VERIFICATION_FAILURE = ProgramResult("Failed to verify a user", 1)


def run(image_path: str, mode: str, user_id: str,
        model_checkpoint_file_path: str):
    """

    :param image_path:
    :param mode:
    :param user_id:
    :param model_checkpoint_file_path:
    :return:
    """
    image = Image(image_path=image_path)

    try:
        image.find_iris_and_pupil()
    except ImageProcessingException:
        return RunResults.PROCESSING_FAILURE

    iris = normalize_iris(image)
    iris.pupil = image.pupil
    iris.iris = image.iris

    # Save the normalized image to a temporary file for easier use with
    # the trained network
    create_empty_dir("tmp")
    iris_hash = hashlib.sha1(image_path.encode()).hexdigest()
    iris_path = f"tmp/{iris_hash}.jpg"
    iris.save(iris_path)

    # Load trained classifier
    classifier = IrisClassifier(load_from_checkpoint=True,
                                checkpoint_file=model_checkpoint_file_path)

    # Get the classifier's prediction
    predicted_user, probability = classifier.evaluate(iris_path)

    if mode == Mode.IDENTIFY:
        if predicted_user == User.UNKNOWN:
            return RunResults.IDENTIFICATION_FAILURE
        else:
            print(f"This image portraits user {predicted_user} "
                  f"(Prediction probability: {probability:.2%})")
            return RunResults.IDENTIFICATION_SUCCESS
    else:
        if predicted_user == User.UNKNOWN:
            return RunResults.VERIFICATION_FAILURE
        else:
            if predicted_user == user_id:
                print(f"Successfully verified user {user_id} "
                      f"(Prediction probability: {probability:.2%})")
                return RunResults.VERIFICATION_SUCCESS
            else:
                return RunResults.VERIFICATION_FAILURE


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Biometric system."
    )

    parser.add_argument("image", type=str, help="Path to the image.")

    parser.add_argument("mode",
                        type=str,
                        choices=[Mode.IDENTIFY, Mode.VERIFY],
                        help="Program mode."
                             "If you want to identify a user based on an "
                             "image, choose 'identify'; "
                             "If you want to verify whether an image "
                             "portraits a particular user, choose 'verify' "
                             "and provide the user's ID in the next argument.")

    parser.add_argument("-u", "--user",
                        type=str,
                        help="User's ID. Only used with mode 'verify'.")

    parser.add_argument("-m", "--model",
                        type=str,
                        help="Path to the trained classifier model",
                        default="model/iris_recognition_trained_model.pt")

    args = parser.parse_args()

    if args.mode == Mode.VERIFY:
        assert args.user is not None, "User ID required for mode 'verify'"

    result = run(
        image_path=args.image,
        mode=args.mode,
        user_id=args.user,
        model_checkpoint_file_path=args.model
    )

    print(f"Program exited with code: "
          f"{result.value.code} - {result.value.message}")
