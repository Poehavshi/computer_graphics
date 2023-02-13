import numpy as np
from lab1.saving_utils import save_image
import os
from config import OUTPUT_PATH


def create_matrix_full_of_value(size: tuple, value: int | tuple) -> np.ndarray:
    return np.full(size, value, dtype=np.uint8)


def create_gradient_matrix(size: tuple) -> np.ndarray:
    matrix = np.zeros(size, dtype=np.uint8)
    for i in range(size[0]):
        for j in range(size[1]):
            matrix[i, j] = (i + j) % 256
    return matrix


def save_matrix_full_of_value_in_fs(size, path, value):
    matrix = create_matrix_full_of_value(size, value=value)
    save_image(matrix, path)


def save_gradient_image_in_fs(size, path):
    matrix = create_gradient_matrix(size)
    save_image(matrix, path)


if __name__ == '__main__':
    task1_output_path = os.path.join(OUTPUT_PATH, "task1")
    os.makedirs(task1_output_path, exist_ok=True)
    height = 100
    width = 100
    save_matrix_full_of_value_in_fs((height, width), os.path.join(task1_output_path, "black.png"), value=0)
    save_matrix_full_of_value_in_fs((height, width), os.path.join(task1_output_path, "white.png"), value=255)
    save_matrix_full_of_value_in_fs((height, width, 3), os.path.join(task1_output_path, "red.png"), value=(255, 0, 0))
    save_gradient_image_in_fs((height, width, 3), os.path.join(task1_output_path, "gradient.png"))
