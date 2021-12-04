from dataclasses import dataclass, astuple

from PIL.Image import fromarray
import cv2
from numpy import array

from constants import CardCount, ImagesPerCard, ModelName, Epochs, CardImageSize
from dataset import create_dataset
from decorators import window, run, webcam
from decorators.window import Indefinite
from fp import exhaust
import imagen
from loader import shuffled_cards
from model import Model

@run(True)
def main():
  (images, labels) = create_dataset(shuffled_cards(CardCount), ImagesPerCard)
  imagenerator = imagen.fitted(images)

  @run(True)
  def present_cardgen():
    generator = exhaust(imagenerator.flow(images))

    @window(Indefinite)
    def show_next_card():
      cv2.imshow('a', next(generator))

  @run(False)
  def create_model():
    print("Creating model...")
    model = Model.complied(ModelName)
    print(f"Created {ModelName}.")

    model.summary()

    print("Training Model...")
    model.train(imagenerator, images, labels, Epochs)
    print(f"Trained {ModelName}.")

    print(f"Saving...")
    model.save()
    print(f"Saved {ModelName}...")

  @run(False)
  def present_model():
    model = Model.load(ModelName)
    model.summary()

    @webcam
    def predict(frame):
      image = array(fromarray(frame).convert('L').resize(CardImageSize))
      image.resize((1, *CardImageSize, 1))
      print(model.predict(image))

      (x, y) = CardImageSize
      image.resize(CardImageSize)
      frame[:x, :y, 0] = image
      frame[:x, :y, 1] = image
      frame[:x, :y, 2] = image
      return frame
