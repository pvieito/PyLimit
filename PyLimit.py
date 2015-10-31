#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''PyLimit.py - Pedro José Pereira Vieito © 2014

Usage:
  PyLimit.py [<video> -t TOLERANCE -m MATCHES -s SPEED -n NEIGHBORS]

Options:
  -t TOLERANCE  Tolerance [default: 10]
  -m MATCHES    Repeated matches [default: 3]
  -s SPEED      Droped frames per recognition [default: 0]
  -n NEIGHBORS  Neighbors per detection [default: 3]
  -h, --help    Show this help
'''

from __future__ import division
from docopt import docopt
import glob
import sys
import os
import cv2
import operator

__author__ = "Pedro José Pereira Vieito"
__email__ = "pvieito@gmail.com"

args = docopt(__doc__)


def all_same(items):
    return all(x == items[0] for x in items)


def read_signs():
    signs = []
    for item in glob.glob('signs/*.jpg'):
        signs.append((os.path.splitext(os.path.basename(item))[0], cv2.imread(item)))
    return signs


def to_blackwhite(im):
    img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    thresh, im_bw = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return im_bw


def check_sign(i1, i2, limit):
    diff = cv2.absdiff(to_blackwhite(i1), to_blackwhite(i2))
    diff = cv2.bitwise_and(mask, diff)
    diff_mean = diff.mean()

    preview = cv2.resize(diff, (100, 100))
    cv2.putText(preview, str(int(diff_mean)), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.imshow(limit, preview)

    # Show Comparisons
    if int(limit) > 90:
        cv2.moveWindow(limit, (int(limit) - 100) * 13 + 10, 150)
    else:
        cv2.moveWindow(limit, (int(limit) - 10) * 13 + 10, 10)
    return diff_mean


def analyze_rects(img, rects, color):
    printed_limit = None
    for (x, y, w, h) in rects:
        x1, y1, x2, y2 = x, y, x + w, y + h
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        sign = cv2.resize(img[y1:y2, x1:x2], (200, 200))

        official_signs = read_signs()

        limits = {}
        for limit, official_sign in official_signs:
            limits[limit] = check_sign(sign, official_sign, limit)

        detected_limit = min(limits.iteritems(), key=operator.itemgetter(1))[0]

        cv2.imshow('Signs', cv2.resize(sign, (100, 100)))
        cv2.moveWindow('Signs', 10, 270)

        cv2.imshow('BW Sign', cv2.resize(to_blackwhite(sign), (100, 100)))
        cv2.moveWindow('BW Sign', 150, 270)

        if limits[detected_limit] < int(args["-t"]):
            printed_limit = detected_limit
            print(detected_limit, limits[detected_limit])

    return printed_limit

cascade = cv2.CascadeClassifier("lbpcascade_signs.xml")
mask = to_blackwhite(cv2.imread('mask.jpg'))
detected_limits = []
last_detected_limit = '--'

if __name__ == '__main__':

    if args["<video>"]:
        try:
            video = cv2.VideoCapture(int(args["<video>"]))
        except:
            video = cv2.VideoCapture(args["<video>"])
    else:
        video = cv2.VideoCapture(0)

    while True:

        for i in range(int(args["-s"])):
            video.read()

        ret, img = video.read()

        rects = cascade.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), minNeighbors=int(args["-n"]))

        detected_limit = analyze_rects(img, rects, (0, 255, 0))

        if detected_limit:
            detected_limits.append(detected_limit)

        if len(detected_limits) >= int(args["-m"]):
            if all_same(detected_limits[-int(args["-m"]):]):
                if last_detected_limit != detected_limits[-1]:
                    last_detected_limit = detected_limits[-1]
                    print('===> ' + last_detected_limit)
        
        img = cv2.resize(img, (550, 300))
        cv2.putText(img, last_detected_limit, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('PyLimit', img)
        cv2.moveWindow('PyLimit', 10, 400)

        if 0xFF & cv2.waitKey(5) == 27:
            break
    
    cv2.destroyAllWindows()
