from PIL import Image
import numpy as np


class MatrixImage:
    def __init__(self):
        self.matrix = None
        self.height = None
        self.width = None
        self.channels = None

    def create_matrix_full_of_value(self, size: tuple, value: int | tuple) -> np.ndarray:
        self.matrix = np.full(size, value, dtype=np.uint8)
        self.height, self.width, self.channels = self.matrix.shape
        return self.matrix

    def create_gradient_matrix(self, size: tuple) -> np.ndarray:
        self.matrix = np.zeros(size, dtype=np.uint8)
        for i in range(size[0]):
            for j in range(size[1]):
                self.matrix[i, j] = (i + j) % 256
        self.height, self.width, self.channels = self.matrix.shape
        return self.matrix

    def save(self, path):
        img = Image.fromarray(self.matrix)
        img.save(path)

    def load(self, path):
        img = Image.open(path)
        self.matrix = np.array(img)
        self.height, self.width, self.channels = self.matrix.shape
        return self.matrix

    def show(self):
        img = Image.fromarray(self.matrix)
        img.show()

    def __getitem__(self, item):
        return self.matrix[item]

    def __setitem__(self, key, value):
        self.matrix[key] = value

    def __str__(self):
        return f"Image with shape: {self.matrix.shape}"

    def __repr__(self):
        return f"Image with shape: {self.matrix.shape}"

    def __add__(self, other):
        return self.matrix + other.matrix

    def __sub__(self, other):
        return self.matrix - other.matrix

    def __mul__(self, other):
        return self.matrix * other.matrix

    def __truediv__(self, other):
        return self.matrix / other.matrix

    def __floordiv__(self, other):
        return self.matrix // other.matrix

    def __mod__(self, other):
        return self.matrix % other.matrix

    def __pow__(self, other):
        return self.matrix ** other.matrix

    def __and__(self, other):
        return self.matrix & other.matrix

    def __or__(self, other):
        return self.matrix | other.matrix

    def __xor__(self, other):
        return self.matrix ^ other.matrix
