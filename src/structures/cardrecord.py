from __future__ import annotations
from dataclasses import dataclass

from cv2 import imread, resize, IMREAD_GRAYSCALE, imshow, IMREAD_COLOR
from numpy.typing import NDArray

from constants import Paths, Labels, CardImageSize, CardImageChannels, CardImageShape

@dataclass
class CardRecord(object):
  image: NDArray
  label: int

  @classmethod
  def from_path(cls, path: str) -> CardRecord:
    image: NDArray
    if CardImageChannels == 1:
      image = imread(f"{Paths['cards']}/{path}", IMREAD_GRAYSCALE)
    else:
      image = imread(f"{Paths['cards']}/{path}", IMREAD_COLOR)
    image = resize(image, CardImageSize)
    image.resize(CardImageShape)

    label = Labels[path.split(".")[0]]
    return cls(image, label)

  def show(self):
    return imshow("Card", self.image)
