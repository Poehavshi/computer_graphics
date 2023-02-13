import numpy as np
from config import INPUT_PATH, OUTPUT_PATH
import os
import logging
from saving_utils import save_image
from task1 import create_matrix_full_of_value

log = logging.getLogger(__name__)
HEIGHT = 1000
WIDTH = 1000

class Model3d:
    def __init__(self):
        self.points = None
        self.edges = None
        self.faces = None

    def from_file(self, path):
        with open(path, 'r') as file:
            lines = file.readlines()
            self.points = []

            for line in lines:
                if line.startswith('v '):
                    x, y, z = line[2:].split(' ')
                    self.points.append((float(x), float(y), float(z)))
                elif line.startswith('f '):
                    pass
                elif line.startswith('l '):
                    pass
        log.info(f'Loaded {len(self.points)} points from {path} file')

    def render_on_image(self, matrix_with_image, coefficient=4000, shift=500):
        rendered_image = matrix_with_image.copy()
        for point in self.points:
            x, y, z = point
            x = int(x * coefficient + shift)
            y = int(y * coefficient + shift)
            rendered_image[y, x] = (255, 255, 255)
        return rendered_image


def save_rendered_image(image, path, coefficient=4000, shift=500):
    image = model.render_on_image(image, coefficient, shift)
    save_image(image, path)


if __name__ == '__main__':
    blank_image = create_matrix_full_of_value((HEIGHT, WIDTH, 3), (0, 0, 0))
    model = Model3d()
    model.from_file(os.path.join(INPUT_PATH, 'model_1.obj'))
    save_rendered_image(blank_image, os.path.join(OUTPUT_PATH, 'model_1_50.png'), 50)
    save_rendered_image(blank_image, os.path.join(OUTPUT_PATH, 'model_1_100.png'), 100)
    save_rendered_image(blank_image, os.path.join(OUTPUT_PATH, 'model_1_500.png'), 500)
    save_rendered_image(blank_image, os.path.join(OUTPUT_PATH, 'model_1_4000.png'), 4000)

