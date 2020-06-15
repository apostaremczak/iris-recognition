import numpy as np
import random
from tqdm import tqdm

from utils.file_utils import *
from utils.image import Image
from utils.preprocessing_exceptions import ImageProcessingException

ORIGINAL_DATA_DIR = "../data/tmp/original_renamed"
CIRCLED_DATA_DIR = "../data/tmp/circled"
NORMALIZED_DATA_DIR = "../data/tmp/normalized_all"
SYSTEM_DATABASE_DIR = "../data/system_database"

NORMALIZED_IMAGE_WIDTH = 400
NORMALIZED_IMAGE_HEIGHT = 224


def normalize_iris(image: Image,
                   output_height: int = NORMALIZED_IMAGE_HEIGHT,
                   output_width: int = NORMALIZED_IMAGE_WIDTH) -> Image:
    """
    Normalize iris region by unwrapping the circular region into a rectangular
    block of constant dimensions.
    Source: https://github.com/thuyngch/Iris-Recognition/

    :param image: Input iris image
    :param output_height: Radial resolution (vertical dimension)
    :param output_width: Angular resolution (horizontal dimension)

    :return image: Normalized form of the iris region
    """
    radius_pixels = output_height + 2
    angle_divisions = output_width - 1

    theta = np.linspace(0, 2 * np.pi, angle_divisions + 1)

    # Calculate displacement of pupil center from the iris center
    ox = image.pupil.center_x - image.iris.center_x
    oy = image.pupil.center_y - image.iris.center_y

    sgn = -1 if ox <= 0 else 1

    if ox == 0 and oy > 0:
        sgn = 1

    a = np.ones(angle_divisions + 1) * (ox ** 2 + oy ** 2)

    # Need to do something for ox = 0
    if ox == 0:
        phi = np.pi / 2
    else:
        phi = np.arctan(oy / ox)

    b = sgn * np.cos(np.pi - phi - theta)

    # Calculate radius around the iris as a function of the angle
    r = np.sqrt(a) * b + np.sqrt(a * b ** 2 - (a - image.iris.radius ** 2))
    r = np.array([r - image.pupil.radius])

    r_mat = np.dot(np.ones([radius_pixels, 1]), r)

    r_mat = r_mat * np.dot(np.ones([angle_divisions + 1, 1]),
                           np.array([
                               np.linspace(0, 1, radius_pixels)
                           ])).transpose()
    r_mat = r_mat + image.pupil.radius

    # Exclude values at the boundary of the pupil iris border,
    # and the iris scelra border as these may not correspond to areas
    # in the iris region and will introduce noise.
    # ie don't take the outside rings as iris data.
    r_mat = r_mat[1: radius_pixels - 1, :]

    # Calculate cartesian location of each data point around
    # the circular iris region
    x_cos_mat = np.dot(np.ones([radius_pixels - 2, 1]),
                       np.array([np.cos(theta)]))
    x_sin_mat = np.dot(np.ones([radius_pixels - 2, 1]),
                       np.array([np.sin(theta)]))

    xo = r_mat * x_cos_mat
    yo = r_mat * x_sin_mat

    xo = image.pupil.center_x + xo
    xo = np.round(xo).astype(int)
    coords = np.where(xo >= image.shape[1])
    xo[coords] = image.shape[1] - 1
    coords = np.where(xo < 0)
    xo[coords] = 0

    yo = image.pupil.center_y - yo
    yo = np.round(yo).astype(int)
    coords = np.where(yo >= image.shape[0])
    yo[coords] = image.shape[0] - 1
    coords = np.where(yo < 0)
    yo[coords] = 0

    polar_array = image.img[yo, xo]
    polar_array = polar_array / 255

    # Get rid of outlying points in order to write out the circular pattern
    image.img[yo, xo] = 255

    # Get pixel coords for circle around iris
    x, y = image.iris.find_circle_coordinates(image.shape)
    image.img[y, x] = 255

    # Get pixel coords for circle around pupil
    xp, yp = image.pupil.find_circle_coordinates(image.shape)
    image.img[yp, xp] = 255

    # Replace NaNs before performing feature encoding
    coords = np.where((np.isnan(polar_array)))
    polar_array2 = polar_array
    polar_array2[coords] = 0.5
    avg = np.sum(polar_array2) / (polar_array.shape[0] * polar_array.shape[1])
    polar_array[coords] = avg

    return Image((polar_array * 255).astype(np.uint8))


def circle_available_images(image_paths: List[str], target_dir: str):
    create_empty_dir(target_dir)
    failed_count = 0

    for image_path in image_paths:
        eye_image = Image(image_path=image_path)
        user_id, sample_id = extract_user_sample_ids(image_path)
        try:
            eye_image.circle_iris_and_pupil().save(
                f"{target_dir}/{user_id}_{sample_id}.jpg"
            )
        except ImageProcessingException:
            failed_count += 1

    print(f"{failed_count} failed preprocessings")


def _find_users_with_most_photos(circled_images_paths: List[str],
                                 num_users_wanted: int = 100) -> List[str]:
    """
    :return: List of sorted user IDs
    """
    user_stats = {}

    for path in circled_images_paths:
        user_id, _ = extract_user_sample_ids(path)
        user_stats[user_id] = user_stats.get(user_id, 0) + 1

    sorted_stats = sorted(user_stats.items(), key=lambda x: x[1], reverse=True)

    assert len(sorted_stats) >= num_users_wanted, \
        "Could not find a sufficient number of users"

    user_ids = set(stats[0] for stats in sorted_stats[:num_users_wanted])

    return sorted(list(user_ids))


def split_system_users(circled_images_dir: str = CIRCLED_DATA_DIR,
                       original_images_dir: str = ORIGINAL_DATA_DIR,
                       system_database_dir: str = SYSTEM_DATABASE_DIR,
                       normalized_images_dir: str = NORMALIZED_DATA_DIR,
                       number_of_users_wanted: int = 100,
                       normalized_image_width: int = NORMALIZED_IMAGE_WIDTH,
                       normalized_image_height: int = NORMALIZED_IMAGE_HEIGHT,
                       random_state: int = 1):
    """
    Divide selected users into "registered", which will be known to the system,

    and "unknown", which should be rejected upon identification.
    Move their input images into their respective data/system_database folders.

    Additionally, split "registered" user photos into train and val subsets
    and save them into data/tmp/normalized.
    """
    circled_images_paths = sorted(glob(f"{circled_images_dir}/*"))
    top_users_ids = _find_users_with_most_photos(circled_images_paths,
                                                 number_of_users_wanted)
    # Filter out unnecessary images
    top_users_image_paths = [
        path for path in circled_images_paths
        if extract_user_sample_ids(path)[0] in top_users_ids
    ]

    # Shuffle IDs
    random.seed(random_state)
    random.shuffle(top_users_ids)

    # Split users into "registered" and "unknown"
    split_idx = number_of_users_wanted // 2
    registered_ids = top_users_ids[:split_idx]

    registered_dir = f"{system_database_dir}/registered_users"
    unknown_dir = f"{system_database_dir}/unknown_users"

    create_empty_dir(normalized_images_dir)
    create_empty_dir(registered_dir)
    create_empty_dir(unknown_dir)

    for image_path in tqdm(top_users_image_paths,
                           desc="Copying irides images"):
        user_id, _ = extract_user_sample_ids(image_path)

        is_registered = user_id in registered_ids
        target_dir = registered_dir if is_registered else unknown_dir

        file_name = get_file_name(image_path)
        original_image_path = f"{original_images_dir}/{file_name}"

        # Save original image to data/system_database
        copyfile(original_image_path, f"{target_dir}/{file_name}")

        if is_registered:
            eye_image = Image(image_path=original_image_path)
            eye_image.find_iris_and_pupil()
            normalized = normalize_iris(
                eye_image,
                output_height=normalized_image_height,
                output_width=normalized_image_width
            )

            normalized.save(f"{normalized_images_dir}/{file_name}")


if __name__ == '__main__':
    # Circle pupil and iris on all available photos from the database
    # circle_available_images(
    #     image_paths=sorted(glob(ORIGINAL_DATA_DIR + "/*")),
    #     target_dir=CIRCLED_DATA_DIR
    # )

    # After removing incorrectly circled images, extract pupils for
    # the remaining users and normalize them
    split_system_users()
