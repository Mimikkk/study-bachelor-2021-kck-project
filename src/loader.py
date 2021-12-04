from os import listdir
from random import shuffle, sample

from constants import Paths
from structures import CardRecord

def load_cards(count: int) -> list[CardRecord]:
  paths = listdir(Paths['cards'])
  shuffle(paths)
  cards = []

  print("Loading cards...")
  for (index, path) in enumerate(paths[:count], start=1):
    index % 4 == 0 and print(f"Loaded {index}/{count} cards.")
    cards.append(CardRecord.from_path(path))

  print(f"Loaded all {count} cards.")
  return cards

def shuffled_cards(count: int) -> list[CardRecord]:
  return sample(load_cards(count), count)
