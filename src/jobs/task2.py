
import os
from configs.task2.base_config import SIZE, VERBOSE, ROOT_OUTPUT_PATH,  IMAGES
from src.schema import MatrixImageRGB
from src.drawing.figures import draw_star_at_center


def run():
    os.makedirs(ROOT_OUTPUT_PATH, exist_ok=True)
    matrix_image = MatrixImageRGB()
    for image in IMAGES:
        process_single_image(image, matrix_image)


def process_single_image(image, matrix_image):
    matrix_image.from_rgb_color(SIZE, image["color"])
    draw_star_at_center(matrix_image.matrix, image["line_color"], image["line_method"])
    matrix_image.save(image["path"])
    if VERBOSE:
        matrix_image.show()


if __name__ == '__main__':
    run()
