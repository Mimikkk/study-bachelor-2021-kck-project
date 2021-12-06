from dataclasses import dataclass, astuple

from PIL.Image import fromarray
import cv2
from numpy import array, argmax

from constants import CardCount, ImagesPerCard, ModelName, Epochs, CardImageSize, Labels, Classes, CardImageChannels, \
  CardImageShape
from dataset import create_dataset
from decorators import window, run, webcam
from decorators.window import Indefinite
from fp import exhaust, apply
import imagen
from loader import shuffled_cards
from model import Model

@run(True)
def create_model():
  (images, labels) = create_dataset(shuffled_cards(CardCount), ImagesPerCard)
  imagenerator = imagen.fitted(images)

  @run(False)
  def present_cardgen():
    generator = exhaust(map(apply(zip), imagenerator.flow(images, labels)))

    model = Model.load(ModelName)

    @window(Indefinite)
    def show_next_card():
      (image, label) = next(generator)
      real = f"Card is {Classes[int(argmax(labels))]}."
      prediction = f"Card is {Classes[int(argmax(model.predict(array(image))))]}."
      image = cv2.putText(image, real, (0, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
      image = cv2.putText(image, prediction, (0, CardImageSize[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
      cv2.imshow("Card", image)

  @run(True)
  def train_model():
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

@run(True)
def present_model():
  model = Model.load(ModelName)
  model.summary()

  @webcam
  def predict(frame):
    image = array(fromarray(frame).resize(CardImageSize))
    cv2.flip(image, 1, image)
    image.resize((1, *CardImageShape))

    label = int(argmax(model.predict(image)))
    classname = Classes[label]

    (x, y) = CardImageSize
    image.resize(CardImageShape)

    if CardImageChannels == 1:
      frame[:x, :y, 0] = image
      frame[:x, :y, 1] = image
      frame[:x, :y, 2] = image
    elif CardImageChannels == 3:
      frame[:x, :y, :] = image

    frame = cv2.putText(frame, f"Card is {label} which has", (0, x + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
    frame = cv2.putText(frame, f"{classname} class value.", (0, x + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
    return frame
