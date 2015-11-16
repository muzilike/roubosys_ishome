#!usr/bin/env python
#coding=utf-8

import cv2

capture = cv2.VideoCapture(0)
width = int(capture.get(3))
height = int(capture.get(4))
firstFrame = None
while True:
  (grabbed, frame) = capture.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  gray = cv2.GaussianBlur(gray, (21, 21), 0)

  if firstFrame is None:
    firstFrame = gray
    continue

  frameDelta = cv2.absdiff(firstFrame, gray)
  thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
  thresh = cv2.dilate(thresh, None, iterations=2)
  (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
  for c in cnts:
     if cv2.contourArea(c) < 10000:
       continue
     (x, y, w, h) = cv2.boundingRect(c)
     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
  cv2.imshow("Security Feed", frame)
  firstFrame = gray.copy()
capture.release()
cv2.destroyAllWindows()
