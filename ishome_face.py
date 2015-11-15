#!/usr/bin/env python
# encoding: utf-8

import cv2.cv as cv
import cv2
capture=cv.CaptureFromCAM(0)
#hc = cv.Load("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml")
cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt.xml")
while True:
  frame = cv.QueryFrame(capture)
  faces = cascade.detectMultiScale(frame)
  for ((x, y, w, h), stub) in faces:
    cv.Rectangle(frame, (int(x), int(y)), (int(x)+w, int(y)+h), (0, 255, 0), 2, 0)
  cv.ShowImage("Window", frame)
  c=cv.WaitKey(1)
  if c==27 or c == 1048603:
    break
