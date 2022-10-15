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
from functions_lib import Detection, Tracker
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

# Get Intersection over Union value
def computeIOU(bboxA, bboxB):
    # determine the (x, y)-coordinates of the intersection rectangle
	xA = max(bboxA[0], bboxB[0])
	yA = max(bboxA[1], bboxB[1])
	xB = min(bboxA[2], bboxB[2])
	yB = min(bboxA[3], bboxB[3])

	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (bboxA[2] - bboxA[0] + 1) * (bboxA[3] - bboxA[1] + 1)
	boxBArea = (bboxB[2] - bboxB[0] + 1) * (bboxB[3] - bboxB[1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou


def main():

    global person_name, input_read_control

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
    traker_counter = 0
    detection_counter = 0
    trackers = []

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

        # ----------------------------------------------
        # Create face detections per haard cascade bbox
        # ----------------------------------------------
        detections = []
        for bbox in bboxes:  # cycle all bounding boxes
            x1, y1, w, h = bbox
            
            if w * h > bbox_area_threshold:
                detection = Detection(x1, y1, w, h, image_gray, detection_counter)
                detection_counter += 1
                detections.append(detection)
                detection.draw(image_gui)  # draw bbox around face detected

        # ------------------------------------------------------------------------------------
        # For each detection, verify if there is alredy one tracker associated
        # Remove detection from list if that's the case
        # ------------------------------------------------------------------------------------
        for detection in detections: # cycle all detections
            for tracker in trackers: # cycle all trackers
                template = tracker.template
                h_template,w_template = template.shape

                # Apply template Matching
                res = cv2.matchTemplate(image_gray, template, cv2.TM_SQDIFF)

                _, _, min_loc, _ = cv2.minMaxLoc(res)

                x1_template, y1_template = min_loc
                x2_template = x1_template + w_template
                y2_template = y1_template + h_template

                bboxA = [x1_template, y1_template, x2_template, y2_template]
                bboxB = [detection.x1, detection.y1, detection.x2, detection.y2]

                iou_value = computeIOU(bboxA, bboxB)

                if iou_value > 0.6:
                    cv2.rectangle(image_gui, (x1_template, y1_template), (x2_template, y2_template), (0,255,0), 3)

                    cv2.putText(image_gui, tracker.id, (x1_template, y1_template), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

                    detections.remove(detection)

        # ------------------------------------------------
        # Create tracker for new detection
        # ------------------------------------------------
        for detection in detections:  # cycle all detections
            # Ask person detected his name
            ask_name_thread = threading.Thread(target = getUserInput)

            if ask_name_thread.is_alive() == False and input_read_control == False:
                ask_name_thread.start()

            if person_name != "":
                tracker = Tracker(detection, person_name)
                traker_counter += 1
                trackers.append(tracker)

                person_name = ""  # Reset person name variable
                

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