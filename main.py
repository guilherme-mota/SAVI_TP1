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
    window_name = 'Camera Face Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 500)

    # Load Pre-trained Classifiers
    # face_detector = cv2.CascadeClassifier('/home/guilherme/workingcopy/opencv-4.5.4/data/haarcascades/haarcascade_frontalface_default.xml')  # /home/miguel/Documents/SAVI_TP1/  
    face_detector = cv2.CascadeClassifier('/home/miguel/Documents/SAVI_TP1/haarcascade_frontalface_default.xml')
    # ------------------------
    # Inittialize variables
    # ------------------------
    bbox_area_threshold = 20000  # normal value >= 80000
    iou_threshold = 0.6  # normal value >= 0.7
    frame_counter = 0
    tracker_counter = 0
    detection_counter = 0
    trackers = []
    

    # ------------------------
    # Execution
    # ------------------------
    while capture.isOpened():  # loop through all frames
        ret, image_original = capture.read()  # get a frame, ret will be true or false if getting succeeds
        image_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)  # convert color image to gray
        image_gui = deepcopy(image_original)  # image for graphical user interface
        [H,W,NC] = image_gui.shape
        #darken_bbox = [50, 50, W-50, H-50]


        if ret == False:
            break

        stamp = float(capture.get(cv2.CAP_PROP_POS_MSEC))/1000


        # ------------------------------------------
        # Detection of faces
        # ------------------------------------------
        bboxes = face_detector.detectMultiScale(image_gray, 1.1, 3 , 0, (0, 0), (0, 0))

        # ----------------------------------------------
        # Create face detections per haard cascade bbox
        # ----------------------------------------------
        detections = []  # for each frame, there's a new lis of detections
        for bbox in bboxes:  # cycle all bounding boxes
            x1, y1, w, h = bbox

            detection = Detection(x1, y1, w, h, image_gray, detection_counter, stamp)
            detection_counter += 1
            detections.append(detection)  # add new detection to list of detections

        # --------------------------------------------------------------------
        # For each detection, verify if there already a tracker associated to
        # --------------------------------------------------------------------
        for detection in detections: # cycle all detections
            for tracker in trackers: # cycle all trackers
                tracker_bbox = tracker.detections[-1]  # get last detection from list
                iou = detection.computeIOU(tracker_bbox) # Get Intersection over Union value
                if iou > iou_threshold: # if condition is verified, associate detection with tracker 
                    tracker.addDetection(detection, image_gray)

        # ------------------------------------------
        # Track without using detection
        # ------------------------------------------
        for tracker in trackers: # cycle all trackers
            last_detection_id = tracker.detections[-1].id  # get last detection id
            detection_ids = [d.id for d in detections]  # get all detection id's in the list detections
            if not last_detection_id in detection_ids:  # id last detection id isn't found in the list, track using other method
                tracker.track(image_gray)

        # ------------------------------------------
        # Deactivate Tracker
        # ------------------------------------------
        for tracker in trackers:
            tracker.updateTime(stamp)


        # ------------------------------------------
        # Create Tracker for each Detection
        # ------------------------------------------
        for detection in detections:
            # verify if theres already a tracker associated and if the face is close to the camera
            if not detection.assigned_to_tracker and detection.area > bbox_area_threshold:
                tracker = Tracker(detection, id=tracker_counter, image=image_gray)
                tracker_counter += 1
                trackers.append(tracker)

                # Ask person detected his name
                ask_name_thread = threading.Thread(target = tracker.getUserInput)

                if ask_name_thread.is_alive() == False and tracker.input_read_control == False:
                    ask_name_thread.start()

        # ------------------------------------------
        # Draw stuff
        # ------------------------------------------
        # Draw detections with no tracker associated
        for detection in detections:
            if not detection.assigned_to_tracker:
                detection.draw(image_gui)  # draw red bbox around face detected

        # Draw trackers
        for tracker in trackers:
            tracker.draw(image_gui)  # draw green bbox around face tracked


        # Draw zone that disables trackers
        image_gui[:, 0:50] = (image_gui [:, 0:50]* 0.3).astype(np.uint8)
        image_gui[:, W-50:W] = (image_gui [:, W-50:W]* 0.3).astype(np.uint8)
        image_gui[0:50, 50:W-50] = (image_gui [0:50, 50:W-50]* 0.3).astype(np.uint8)
        image_gui[H-50:H, 50:W-50] = (image_gui [H-50:H, 50:W-50]* 0.3).astype(np.uint8)

        # Display Image Capture
        cv2.imshow(window_name, image_gui)

        if cv2.waitKey(1) == ord('q'):
            break

        frame_counter += 1  # increment frame counter


    # ------------------------
    # Termination
    # ------------------------
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()