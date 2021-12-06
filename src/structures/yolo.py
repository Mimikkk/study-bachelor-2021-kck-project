from __future__ import annotations
from dataclasses import astuple, dataclass

from numpy import uint8

from constants import Labels


@dataclass
class Yolo(object):
  label: uint8
  x_center: float
  y_center: float
  width: float
  height: float

  @classmethod
  def from_bbox(cls, bbox: 'BBox') -> Yolo:
    from src.structures import BBox
    (label, xmin, xmax, ymin, ymax) = astuple(bbox)
    w = xmax - xmin
    h = ymax - ymin
    x = (xmax + xmin) / 2.0
    y = (ymax + ymin) / 2.0
    return cls(label, x, y, w, h)

  @classmethod
  def from_path(cls, path: str) -> list[Yolo]:
    labels: list[str] = ['2c', '3c', '4c', '5c',
                         '6c', '7c', '8c', '9c',
                         '10c', 'Ac', 'Jc', 'Kc', 'Qc',
                         '2d', '3d', '4d', '5d',
                         '6d', '7d', '8d', '9d',
                         '10d', 'Ad', 'Jd', 'Kd', 'Qd',
                         '2h', '3h', '4h', '5h',
                         '6h', '7h', '8h', '9h',
                         '10h', 'Ah', 'Jh', 'Kh', 'Qh',
                         'As', '2s', '3s', '4s',
                         '5s', '6s', '7s', '8s',
                         '9s', '10s', 'Js', 'Ks', 'Qs']

    yolos = []
    with open(path, 'r') as file:
      for yolo in file.readlines():
        (label, x, y, w, h) = map(float, yolo.split())
        yolos.append(cls(uint8(Labels[labels[int(label)]]), x, y, w, h))
    return yolos
