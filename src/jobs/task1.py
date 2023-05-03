from src.schema import MatrixImageRGB
from configs.task1.base_config import SIZE, VERBOSE, ROOT_OUTPUT_PATH, IMAGES
import os


def run():
    os.makedirs(ROOT_OUTPUT_PATH, exist_ok=True)
    matrix_image = MatrixImageRGB()
    for image in IMAGES:
        process_single_image(image, matrix_image)


def process_single_image(image, matrix_image):
    image_from_color_config(image, matrix_image, SIZE)
    matrix_image.save(image["path"])
    if VERBOSE:
        matrix_image.show()


def image_from_color_config(image, matrix_image, size):
    if image["color"] is None:
        matrix_image.from_gradient(size)
    else:
        matrix_image.from_rgb_color(size, image["color"])


if __name__ == '__main__':
    run()
