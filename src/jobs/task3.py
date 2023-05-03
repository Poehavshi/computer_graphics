from PIL import Image

from configs.task3.base_config import MODEL_PATH, IMAGES, TEXTURE_PATH, VERBOSE, SIZE, ROOT_OUTPUT_PATH, Z_BUFFER_MIN, Z_BUFFER_MAX
from src.reading.model_reader import ObjModelReader
from src.render.model_renderer import ModelRenderer
from src.schema import MatrixImageRGB
import os


def run():
    os.makedirs(ROOT_OUTPUT_PATH, exist_ok=True)
    model_reader = ObjModelReader()
    texture = Image.open(TEXTURE_PATH)
    for image_config in IMAGES:
        model = model_reader.read(MODEL_PATH)
        model.rotate(0, 0, 0)
        model_renderer = ModelRenderer(model,
                                       z_min=Z_BUFFER_MIN,
                                       z_max=Z_BUFFER_MAX,
                                       texture_image=texture)
        image = MatrixImageRGB().from_rgb_color(SIZE, image_config["color"])
        image = model_renderer.render(image)
        image.save(image_config["path"])
        if VERBOSE:
            image.show()


if __name__ == '__main__':
    run()
