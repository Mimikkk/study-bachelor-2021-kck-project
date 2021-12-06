from typing import TypeVar, Iterator, Callable

T = TypeVar('T')
def exhaust(iterator: Iterator[Iterator[T]]) -> T:
  while True: yield from next(iterator)


def apply(fn: Callable[[...], T]) -> Callable[..., T]:
  return lambda args: fn(*args)
