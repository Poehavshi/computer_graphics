from src.schema import MatrixImageRGB
from configs.task1.base_config import HEIGHT, WIDTH, ROOT_OUTPUT_PATH, IMAGES
import os


def run():
    os.makedirs(ROOT_OUTPUT_PATH, exist_ok=True)
    size = (HEIGHT, WIDTH)
    matrix_image = MatrixImageRGB()
    for image in IMAGES:
        image_from_color_config(image, matrix_image, size)
        matrix_image.save(image["path"])
        matrix_image.show()


def image_from_color_config(image, matrix_image, size):
    if image["color"] is None:
        matrix_image.from_gradient(size)
    else:
        matrix_image.from_rgb_color(size, image["color"])


if __name__ == '__main__':
    run()
