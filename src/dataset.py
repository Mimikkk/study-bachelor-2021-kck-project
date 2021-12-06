from random import shuffle, sample

from keras.utils.np_utils import to_categorical
from numpy import array
from numpy.typing import NDArray

from constants import CardCount, CardImageSize, CardImageChannels, CardImageShape
import imagen
from structures import CardRecord
def create_dataset(cards: list[CardRecord], batch_size: int) -> tuple[NDArray, NDArray]:
  labels = to_categorical(array([array([card.label] * batch_size) for card in cards]).flatten(), num_classes=CardCount)

  images = []
  for (i, card) in enumerate(map(lambda card: array([card.image]), cards), start=1):
    print(f'creating images for card {i}/{CardCount}')
    fit = imagen.fitted(card)
    images.append(array([image for _ in range(batch_size) for image in next(fit.flow(card))]))
  print(f"Created images for all {CardCount} cards.")

  print(f"Reshaping all images...")
  images = array(images).reshape((len(cards) * batch_size, *CardImageShape))
  (images, labels) = map(array, zip(*sample(list(zip(images, labels)), len(images))))
  print(f"Reshaped all images.")

  return (images, labels)
