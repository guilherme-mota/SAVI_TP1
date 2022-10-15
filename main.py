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
import pyttsx3
from functions_lib import Detection
import threading 

# ------------------------
# Global Variables
# ------------------------
person_name = ""
input_read_control = False

# ------------------------
# Functions
# ------------------------
def getUserInput():

    global person_name, input_read_control

    input_read_control = True  # Set control variable
    person_name = input('What is your name?\n')  # Ask user input
    input_read_control = False  # Reset control variable

    engine = pyttsx3.init()
    engine.say("Hello " + person_name)
    engine.runAndWait()


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
    # Inittialize variables
    # ------------------------
    bbox_area_threshold = 100000  # normal value >= 80000
    frame_counter = 0
    detection_counter = 0

    # ------------------------
    # Execution
    # ------------------------
    while capture.isOpened():  # loop through all frames
        ret, image_original = capture.read()  # get a frame, ret will be true or false if getting succeeds
        image_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)  # convert color image to gray
        image_gui = deepcopy(image_original)  # image for graphical user interface

        if ret == False:
            break

        # ------------------------------------------
        # Detection of faces
        # ------------------------------------------
        bboxes = face_detector.detectMultiScale(image_gray, 1.1, 3, 0, (0, 0), (0, 0))
        # print(bboxes)

        # ------------------------------------------
        # Create detections per haard cascade bbox
        # ------------------------------------------
        detections = []
        for bbox in bboxes:
            x1, y1, w, h = bbox
            # print(w * h)
            if w * h > bbox_area_threshold:
                detection = Detection(x1, y1, w, h, image_gray, detection_counter)
                detection_counter += 1
                detections.append(detection)
                detection.draw(image_gui)  # draw bbox

                # Ask person detected name
                ask_name_thread = threading.Thread(target = getUserInput)
                if ask_name_thread.is_alive() == False and input_read_control == False:
                    ask_name_thread.start()
                

        # Display Image Capture
        cv2.imshow(window_name, image_gui)

        if cv2.waitKey(1) == ord('q'):
            break

        frame_counter += 1


    # ------------------------
    # Termination
    # ------------------------
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()