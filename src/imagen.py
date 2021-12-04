from keras_preprocessing.image import ImageDataGenerator
import numpy as np

def create() -> ImageDataGenerator:
  return ImageDataGenerator(
    rotation_range=180,
    shear_range=20,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1. / 255,
    brightness_range=[0.2, 1.2],
    validation_split=0.1
  )

def fitted(images: np.ndarray) -> ImageDataGenerator:
  imagen = create()
  imagen.fit(images)
  return imagen
