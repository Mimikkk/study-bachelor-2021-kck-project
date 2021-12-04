from os import getcwd
from os.path import join

Classes: dict[str, int] = {
  '2c': 0, '2s': 1, '2h': 2, '2d': 3,
  '3c': 4, '3s': 5, '3h': 6, '3d': 7,
  '4c': 8, '4s': 9, '4h': 10, '4d': 11,
  '5c': 12, '5s': 13, '5h': 14, '5d': 15,
  '6c': 16, '6s': 17, '6h': 18, '6d': 19,
  '7c': 20, '7s': 21, '7h': 22, '7d': 23,
  '8c': 24, '8s': 25, '8h': 26, '8d': 27,
  '9c': 28, '9s': 29, '9h': 30, '9d': 31,
  'Tc': 32, 'Ts': 33, 'Th': 34, 'Td': 35,
  'Jc': 36, 'Js': 37, 'Jh': 38, 'Jd': 39,
  'Qc': 40, 'Qs': 41, 'Qh': 42, 'Qd': 43,
  'Kc': 44, 'Ks': 45, 'Kh': 46, 'Kd': 47,
  'Ac': 48, 'As': 49, 'Ah': 50, 'Ad': 51,
}
Paths: dict[str, str] = {
  'cwd': getcwd(),
  'resources': join(getcwd(), 'resources'),
  'cards': join(getcwd(), 'resources', 'cards', 'handmade'),
  'yolo': join(getcwd(), 'resources', 'cards', 'yolo-labeled'),
  'models': join(getcwd(), 'resources', 'models')
}

CardImageSize: tuple[int, int] = (224, 224)
CardCount: int = 52
ImagesPerCard: int = 2
Epochs: int = 2
ModelName: str = 'Monika'
