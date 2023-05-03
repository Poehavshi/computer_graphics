from configs.colors import WHITE, BLACK, BLUE
from configs.config import OUTPUT_PATH, INPUT_PATH
import os

# input
HEIGHT = 1000
WIDTH = 1000
SIZE = (HEIGHT, WIDTH)
MODEL_PATH = os.path.join(INPUT_PATH, "model_1.obj")
Z_BUFFER_MIN = 10_000
Z_BUFFER_MAX = 20_000
VERBOSE = True
TEXTURE_PATH = os.path.join(INPUT_PATH, "texture.jpg")

# output
ROOT_OUTPUT_PATH = os.path.join(OUTPUT_PATH, "task3")
IMAGES = [
    {
        "path": os.path.join(ROOT_OUTPUT_PATH, "model_1.png"),
        "color": WHITE,
        "line_color": BLUE
    }
]
