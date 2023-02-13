import numpy as np
from config import INPUT_PATH, OUTPUT_PATH
import os
import logging
from saving_utils import save_image
from task1 import create_matrix_full_of_value
from task2 import create_line_with_bresenham

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
            self.faces = []
            self.edges = []

            for line in lines:
                if line.startswith('v '):
                    x, y, z = line[2:].split(' ')
                    self.points.append((float(x), float(y), float(z)))
                elif line.startswith('f '):
                    v1, v2, v3 = line[2:].split(' ')
                    num_of_point1, num_of_point2, num_of_point3 = map(int, (v1.split('/')[0], v2.split('/')[0], v3.split('/')[0]))
                    self.faces.append((num_of_point1, num_of_point2, num_of_point3))
                    self.edges.append((num_of_point1, num_of_point2))
                    self.edges.append((num_of_point1, num_of_point3))
                    self.edges.append((num_of_point2, num_of_point3))
        log.info(f'Loaded {len(self.points)} points from {path} file')

    def render_on_image(self, matrix_with_image, coefficient=4000, shift=500):
        rendered_image = matrix_with_image.copy()
        self._render_points(rendered_image, coefficient, shift)
        self._render_edges(rendered_image, coefficient, shift)
        return rendered_image

    def _render_points(self, image, coefficient, shift):
        for point in self.points:
            x, y, z = point
            x = int(x * coefficient + shift)
            y = int(y * coefficient + shift)
            image[y, x] = (255, 255, 255)
        return image

    def _render_edges(self, image, coefficient=4000, shift=500):
        for edge in self.edges:
            point1 = self.points[edge[0] - 1]
            point2 = self.points[edge[1] - 1]
            x1, y1, z1 = point1
            x2, y2, z2 = point2
            x1 = int(x1 * coefficient + shift)
            y1 = int(y1 * coefficient + shift)
            x2 = int(x2 * coefficient + shift)
            y2 = int(y2 * coefficient + shift)
            create_line_with_bresenham(x1, y1, x2, y2, image, (255, 255, 255))
        return image

    def _render_faces(self, image, coefficient, shift):
        pass


def save_rendered_image(model3d, image, path, coefficient=4000, shift=500):
    image = model3d.render_on_image(image, coefficient, shift)
    save_image(image, path)


def do_experiment_with_obj(path):
    model = Model3d()
    model.from_file(path)
    blank_image = create_matrix_full_of_value((HEIGHT, WIDTH, 3), (0, 0, 0))
    save_rendered_image(model, blank_image, os.path.join(OUTPUT_PATH, 'task3', 'model_1_50.png'), 50)
    save_rendered_image(model, blank_image, os.path.join(OUTPUT_PATH, 'task3', 'model_1_100.png'), 100)
    save_rendered_image(model, blank_image, os.path.join(OUTPUT_PATH, 'task3', 'model_1_500.png'), 500)
    save_rendered_image(model, blank_image, os.path.join(task3_dirname, 'model_1_4000.png'), 4000)


if __name__ == '__main__':
    task3_dirname = os.path.join(OUTPUT_PATH, 'task3')
    os.makedirs(task3_dirname, exist_ok=True)
    do_experiment_with_obj(os.path.join(INPUT_PATH, 'model_1.obj'))
    # do_experiment_with_obj(os.path.join(INPUT_PATH, 'model_2.obj')) too large
