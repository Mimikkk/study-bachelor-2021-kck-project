from __future__ import annotations

from dataclasses import dataclass, astuple


@dataclass
class BBox(object):
  label: int
  xmin: float
  xmax: float
  ymin: float
  ymax: float

  @classmethod
  def from_yolo(cls, yolo: 'Yolo') -> BBox:
    from src.structures import Yolo
    (label, x_center, y_center, width, height) = astuple(yolo)
    xmax = width / 2 + x_center
    xmin = 2 * x_center - xmax

    ymax = height / 2 + y_center
    ymin = 2 * y_center - ymax
    return cls(label, xmin, xmax, ymin, ymax)
