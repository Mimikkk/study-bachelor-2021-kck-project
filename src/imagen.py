from keras_preprocessing.image import ImageDataGenerator
import numpy as np

def create() -> ImageDataGenerator:
  return ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=90,
    shear_range=15.0,
    zoom_range=(0.9, 1.3),
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.2, 1.8],
    validation_split=0.1
  )

def fitted(images: np.ndarray) -> ImageDataGenerator:
  imagen = create()
  imagen.fit(images)
  return imagen
