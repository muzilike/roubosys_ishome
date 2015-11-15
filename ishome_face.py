#!/usr/bin/env python
# encoding: utf-8
import cv2

capture=cv2.VideoCapture(0)
print capture.isOpened()
while True:
  ret, img=capture.read()
  cv2.imshow('Video', img)
  key=cv2.waitKey(1)
  if key==ord('q'):
    break
capture.release()
cv2.destroyAllWindows()
