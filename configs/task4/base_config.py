from configs.colors import WHITE, BLACK
from configs.config import OUTPUT_PATH
import os

# input
HEIGHT = 1000
WIDTH = 1000
SIZE = (HEIGHT, WIDTH)
VERBOSE = True


# output
ROOT_OUTPUT_PATH = os.path.join(OUTPUT_PATH, "task4")
IMAGES = [
    {
        "path": os.path.join(ROOT_OUTPUT_PATH, "triangle.png"),
        "color": WHITE,
        "line_color": BLACK,
        "coords": [100, 100, 1, 200, 200, 1, 300, 100, 1]
    },
    {
        "path": os.path.join(ROOT_OUTPUT_PATH, "triangle_out_of_bounds.png"),
        "color": WHITE,
        "line_color": BLACK,
        "coords": [1100, 100, 1, 200, 200, 1, 300, 300, 1]
    }
]
