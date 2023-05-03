from configs.config import OUTPUT_PATH
import os
from configs.colors import BLACK, WHITE, RED

# input
HEIGHT = 100
WIDTH = 100
SIZE = (HEIGHT, WIDTH)
VERBOSE = True

# output
ROOT_OUTPUT_PATH = os.path.join(OUTPUT_PATH, "task1")
IMAGES = [
    {
        "path": os.path.join(ROOT_OUTPUT_PATH, "black.png"),
        "color": BLACK
    },
    {
        "path": os.path.join(ROOT_OUTPUT_PATH, "white.png"),
        "color": WHITE
    },
    {
        "path": os.path.join(ROOT_OUTPUT_PATH, "red.png"),
        "color": RED
    },
    {
        "path": os.path.join(ROOT_OUTPUT_PATH, "gradient.png"),
        "color": None
    }
]
