from __future__ import annotations
from dataclasses import dataclass

from cv2 import imread, resize, IMREAD_GRAYSCALE, imshow
from numpy.typing import NDArray

from constants import Paths, Classes, CardImageSize

@dataclass
class CardRecord(object):
  image: NDArray
  label: int

  @classmethod
  def from_path(cls, path: str) -> CardRecord:
    image = imread(f"{Paths['cards']}/{path}", IMREAD_GRAYSCALE)
    image = resize(image, CardImageSize)
    image.resize((*CardImageSize, 1))

    label = Classes[path.split(".")[0]]
    return cls(image, label)

  def show(self):
    return imshow("Card", self.image)
