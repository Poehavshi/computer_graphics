from configs.task3.base_config import MODEL_PATH, IMAGES, VERBOSE, SIZE, ROOT_OUTPUT_PATH
from src.reading.model_reader import ObjModelReader
from src.render.model_renderer import ModelRenderer
from src.schema import MatrixImageRGB
import os


def run():
    os.makedirs(ROOT_OUTPUT_PATH, exist_ok=True)
    model_reader = ObjModelReader()
    model = model_reader.read(MODEL_PATH)
    model_renderer = ModelRenderer(model)
    for image_config in IMAGES:
        image = MatrixImageRGB().from_rgb_color(SIZE, image_config["color"])
        image = model_renderer.render(image)
        image.save(image_config["path"])
        if VERBOSE:
            image.show()


if __name__ == '__main__':
    run()
