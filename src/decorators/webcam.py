from typing import Callable

import cv2
import numpy as np

from decorators.window import window, Infinitesimal

Frame = np.ndarray
def webcam(action: Callable[[Frame], Frame]):
  webcam_handle = cv2.VideoCapture(0)

  @window(wait_time=Infinitesimal)
  def handle_webcam():
    (_, frame) = webcam_handle.read()
    cv2.imshow("Capturing", action(frame))
  webcam_handle.release()
