import coremltools as ct
import numpy as np
from PIL import (
    Image,
    ImageDraw
)

import os
from pathlib import Path

# Global variables
WD_PATH: str = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH: str = os.path.join(WD_PATH, "models")
APPLE_MLPACKAGE_ROOT_PATH: str = os.path.join(WD_PATH, "newmodel.mlpackage")
APPLE_MLPACKAGE_DATA_PATH: str = os.path.join(APPLE_MLPACKAGE_ROOT_PATH, "Data")
APPLE_MLPACKAGE_COREML_PATH: str = os.path.join(APPLE_MLPACKAGE_DATA_PATH, "com.apple.CoreML")
MLMODEL_PATH: str = os.path.join(APPLE_MLPACKAGE_COREML_PATH, "model.mlmodel")
# Apple MLModel format
mlmodel_format: str = "mlmodel"

print(f"Model path: {MLMODEL_PATH}")

model_name: str = "MNISTClassifier"

mlmodel = ct.models.MLModel(os.path.join(MODELS_PATH, model_name + "." + mlmodel_format))
print(mlmodel)
# Create a blank image with a white background
example_image = Image.new('L', (28, 28), color=255)
draw = ImageDraw.Draw(example_image)
# Draw the number 3 on the image
draw.text((10, 8), "3", fill=0)
# Save the image as a file (optional)
# example_image.show()

prediction = mlmodel.predict({
        "image": example_image

})
print(prediction)
