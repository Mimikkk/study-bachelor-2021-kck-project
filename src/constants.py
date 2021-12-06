from os import getcwd
from os.path import join

Labels: dict[str, int] = {
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

Classes: dict[int, str] = {
  0: '2c', 1: '2s', 2: '2h', 3: '2d',
  4: '3c', 5: '3s', 6: '3h', 7: '3d',
  8: '4c', 9: '4s', 10: '4h', 11: '4d',
  12: '5c', 13: '5s', 14: '5h', 15: '5d',
  16: '6c', 17: '6s', 18: '6h', 19: '6d',
  20: '7c', 21: '7s', 22: '7h', 23: '7d',
  24: '8c', 25: '8s', 26: '8h', 27: '8d',
  28: '9c', 29: '9s', 30: '9h', 31: '9d',
  32: 'Tc', 33: 'Ts', 34: 'Th', 35: 'Td',
  36: 'Jc', 37: 'Js', 38: 'Jh', 39: 'Jd',
  40: 'Qc', 41: 'Qs', 42: 'Qh', 43: 'Qd',
  44: 'Kc', 45: 'Ks', 46: 'Kh', 47: 'Kd',
  48: 'Ac', 49: 'As', 50: 'Ah', 51: 'Ad',
}

Paths: dict[str, str] = {
  'cwd': getcwd(),
  'resources': join(getcwd(), 'resources'),
  'cards': join(getcwd(), 'resources', 'cards', 'handmade'),
  'yolo': join(getcwd(), 'resources', 'cards', 'yolo-labeled'),
  'models': join(getcwd(), 'resources', 'models')
}

CardImageShape: tuple[int, int, int] = (224, 224, 1)
CardImageSize: tuple[int, int] = CardImageShape[:2]
CardImageChannels: int = CardImageShape[2]
CardCount: int = 52
ImagesPerCard: int = 512
Epochs: int = 40
ModelName: str = 'KolorowaMonika'
