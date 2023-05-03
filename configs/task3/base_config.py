from configs.colors import WHITE, BLACK
from configs.config import OUTPUT_PATH, INPUT_PATH
import os

# input
HEIGHT = 4000
WIDTH = 4000
SIZE = (HEIGHT, WIDTH)
MODEL_PATH = os.path.join(INPUT_PATH, "model_1.obj")
VERBOSE = True

# output
ROOT_OUTPUT_PATH = os.path.join(OUTPUT_PATH, "task3")
IMAGES = [
    {
        "path": os.path.join(ROOT_OUTPUT_PATH, "model_1.png"),
        "color": WHITE,
        "line_color": WHITE
    }
]
