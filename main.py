#!/usr/bin/env python3

# -------------------------------------------------------------------------------
# Name:        Main
# Purpose:     Intelligent system to detect and track people's faces
# Authors:     Guilherme Mota | Miguel Cruz
# Created:     15/10/2022
# -------------------------------------------------------------------------------

# ------------------------
# Imports
# ------------------------
from copy import deepcopy
import cv2
import numpy as np


def main():

    # ------------------------
    # Initialization
    # ------------------------
    capture = cv2.VideoCapture(0)
    if capture.isOpened() == False:
        print("Error opening video stream or file!")

    # Resize window
    ret, frame = capture.read()
    window_name = 'Camera'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 500)

    # Load Pre-trained Classifiers
    face_detector = cv2.CascadeClassifier('/home/guilherme/workingcopy/opencv-4.5.4/data/haarcascades/haarcascade_frontalface_default.xml')


    # ------------------------
    # Execution
    # ------------------------
    while capture.isOpened():  # loop through all frames
        ret, image_original = capture.read()  # get a frame, ret will be true or false if getting succeeds
        image_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)  # convert color image to gray
        image_gui = deepcopy(image_original)

        if ret == False:
            break

        # ------------------------------------------
        # Detection of faces
        # ------------------------------------------
        bboxes = face_detector.detectMultiScale(image_gray, 1.1, 3, 0, (0, 0), (0, 0));
        print(bboxes)

        # Draw bbox around faces
        for bbox in bboxes:
            x1, y1, w, h = bbox
            cv2.rectangle(image_gui, (x1, y1), (x1+w, y1+h), (255, 0, 0), 3)

        # Display Image Capture
        cv2.imshow(window_name, image_gui)

        if cv2.waitKey(1) == ord('q'):
            break


    # ------------------------
    # Termination
    # ------------------------
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()