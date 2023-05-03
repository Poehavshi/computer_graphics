from configs.colors import WHITE, BLACK
from configs.config import OUTPUT_PATH
import os
from src.drawing.lines import draw_simple_line, draw_advanced_line, draw_line_only_with_steep, \
    draw_line_with_bresenham

# input
HEIGHT = 200
WIDTH = 200
SIZE = (HEIGHT, WIDTH)
VERBOSE = True

# output
ROOT_OUTPUT_PATH = os.path.join(OUTPUT_PATH, "task2")
IMAGES = [
    {
        "path": os.path.join(ROOT_OUTPUT_PATH, "star1.png"),
        "color": BLACK,
        "line_color": WHITE,
        "line_method": draw_simple_line
    },
    {
        "path": os.path.join(ROOT_OUTPUT_PATH, "star2_advanced.png"),
        "color": BLACK,
        "line_color": WHITE,
        "line_method": draw_advanced_line
    },
    {
        "path": os.path.join(ROOT_OUTPUT_PATH, "star3_steep.png"),
        "color": BLACK,
        "line_color": WHITE,
        "line_method": draw_line_only_with_steep
    },
    {
        "path": os.path.join(ROOT_OUTPUT_PATH, "star4_bresenham.png"),
        "color": BLACK,
        "line_color": WHITE,
        "line_method": draw_line_with_bresenham
    }
]
