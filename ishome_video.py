#!usr/bin/env python
#coding=utf-8

import cv2

capture = cv2.VideoCapture(0)
width = int(capture.get(3))
height = int(capture.get(4))
cascade_fn = 'haarcascades/haarcascade_frontalface_alt.xml'
cascade = cv2.CascadeClassifier(cascade_fn)
firstFrame = None
while True:
  (grabbed, frame) = capture.read()
  gray_step1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  gray_step2 = cv2.GaussianBlur(gray_step1, (21, 21), 0)

  if firstFrame is None:
    firstFrame = gray_step2
    continue

  firstFrame = gray_step2.copy()
  frameDelta = cv2.absdiff(firstFrame, gray_step2)
  thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
  thresh = cv2.dilate(thresh, None, iterations=2)
  (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
  if len(cnts) == 0:
    continue

  gray_step3 = cv2.equalizeHist(gray_step1)
  rects = cascade.detectMultiScale(gray_step3, scaleFactor=1.2, minNeighbors=5)
  if len(rects) == 0:
    continue

  for x1, y1, x2, y2 in rects:
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
  cv2.imshow("Security Feed", frame)
  key=cv2.waitKey(1)
  if key==ord('q'):
    break

capture.release()
cv2.destroyAllWindows()
