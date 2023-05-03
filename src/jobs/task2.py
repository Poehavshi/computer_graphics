
import os
from configs.task2.base_config import HEIGHT, WIDTH, ROOT_OUTPUT_PATH, IMAGES
from src.schema import MatrixImageRGB
from src.drawing.figures import draw_star_at_center


def run():
    os.makedirs(ROOT_OUTPUT_PATH, exist_ok=True)
    size = (HEIGHT, WIDTH)
    matrix_image = MatrixImageRGB()
    for image in IMAGES:
        matrix_image.from_rgb_color(size, image["color"])
        draw_star_at_center(matrix_image.matrix, image["line_color"], image["line_method"])
        matrix_image.save(image["path"])
        matrix_image.show()


if __name__ == '__main__':
    run()
