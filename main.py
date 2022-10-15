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


    # ------------------------
    # Execution
    # ------------------------
    while capture.isOpened():  # loop through all frames
        ret, image_original = capture.read()  # get a frame, ret will be true or false if getting succeeds

        if ret == False:
            break

        # Display Image Capture
        cv2.imshow(window_name, image_original)

        if cv2.waitKey(1) == ord('q'):
            break


    # ------------------------
    # Termination
    # ------------------------
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()