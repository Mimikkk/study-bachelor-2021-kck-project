from typing import Callable, Optional

import cv2

Indefinite: Optional[int] = 0
Infinitesimal: Optional[int] = 1
EscapeCode: int = 27

def window(wait_time: Optional[int] = Indefinite, exitkey: int = EscapeCode):
  def handle_loop(action: Callable[[], None]):
    def should_exit(keycode: int) -> bool:
      return keycode == exitkey
    while not should_exit(cv2.waitKey(wait_time)): action()
    cv2.destroyAllWindows()
  return handle_loop
