from typing import TypeVar, Iterator

T = TypeVar('T')
def exhaust(iterator: Iterator[Iterator[T]]) -> T:
  while True: yield from next(iterator)
