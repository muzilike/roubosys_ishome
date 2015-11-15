#!/usr/bin/env python
# encoding: utf-8
import cv2
import cv2.cv as cv

capture=cv2.VideoCapture(0)
cascade_fn = 'haarcascades/haarcascade_frontalface_alt.xml'
cascade = cv2.CascadeClassifier(cascade_fn)
while True:
  ret, img=capture.read()
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray = cv2.equalizeHist(gray)
  rects = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
  for x1, y1, x2, y2 in rects:
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
  cv2.imshow('Video', img)
  key=cv2.waitKey(1)
  if key==ord('q'):
    break
capture.release()
cv2.destroyAllWindows()
