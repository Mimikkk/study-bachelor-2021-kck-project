from __future__ import annotations
from dataclasses import dataclass
from os.path import join
from typing import Any

from keras import Sequential
from keras.activations import relu, softmax
from keras.backend import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.engine.input_layer import InputLayer
from keras.layers import Dense, Dropout, Flatten, MaxPool2D, Convolution2D, MaxPooling2D, Conv2D, AveragePooling2D
from keras.optimizer_v2.adam import Adam
from keras_preprocessing.image import ImageDataGenerator
from numpy.typing import NDArray

from constants import Paths, CardImageSize, CardImageChannels, CardImageShape

@dataclass
class Model(object):
  _handle: Sequential
  name: str

  @classmethod
  def uncompiled(cls, modelname: str) -> Model:
    return Model(Sequential([
      # InputLayer(CardImageShape),
      # Convolution2D(32, (5, 5), padding='same', activation=relu),
      # MaxPooling2D(2, 2),
      # Dropout(0.2),
      # Convolution2D(64, (3, 3), padding='same', activation=relu),
      # MaxPooling2D(2, 2),
      # Dropout(0.1),
      # Convolution2D(128, (3, 3), padding='same', activation=relu),
      # MaxPooling2D(2, 2),
      # Flatten(),
      # Dropout(0.3),
      # Dense(1024, activation=relu),
      # Dense(52, activation=softmax)

      # InputLayer(CardImageShape),
      # Conv2D(64, (3, 3), activation='relu'),
      # MaxPooling2D(2, 2),
      # Conv2D(64, (3, 3), activation='relu'),
      # MaxPooling2D(2, 2),
      # Conv2D(128, (3, 3), activation='relu'),
      # MaxPooling2D(2, 2),
      # Conv2D(128, (3, 3), activation='relu'),
      # MaxPooling2D(2, 2),
      # Flatten(),
      # Dropout(0.25),
      # Dense(512, activation='relu'),

      InputLayer(CardImageShape),
      Conv2D(64, (3, 3), activation='relu'),
      MaxPooling2D(2, 2),
      Conv2D(64, (3, 3), activation='relu'),
      MaxPooling2D(2, 2),
      Conv2D(128, (3, 3), activation='relu'),
      MaxPooling2D(2, 2),
      Conv2D(128, (3, 3), activation='relu'),
      MaxPooling2D(2, 2),
      Flatten(),
      Dropout(0.5),
      Dense(512, activation='relu'),
      Dense(52, activation='softmax')
    ]), modelname)

  @classmethod
  def complied(cls, modelname: str) -> Model:
    model = cls.uncompiled(modelname)
    model._handle.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])
    return model

  @classmethod
  def load(cls, modelname: str) -> Model:
    model = cls.complied(modelname)
    model._handle.load_weights(join(Paths['models'], f"{modelname}.h5"))
    return model

  def predict(self, item: Any) -> list[int]:
    return self._handle.predict(item)[0]

  def save(self):
    self._handle.save(join(Paths['models'], f"{self.name}.h5"))

  def listeners(self) -> list[Callback]:
    return [ModelCheckpoint(filepath=f"{Paths['models']}/best_{self.name}.h5", save_best_only=True)]

  def train(self, imagen: ImageDataGenerator, images: NDArray, labels: NDArray, epochs: int):
    return self._handle.fit(
      imagen.flow(images, labels, subset='training'),
      validation_data=imagen.flow(images, labels, subset='validation'),
      epochs=epochs,
      callbacks=self.listeners()
    )

  def summary(self):
    self._handle.summary()
