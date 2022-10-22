import os
import cv2
import pyttsx3
import numpy as np
import face_recognition

class BoundingBox:
    
    def __init__(self, x1, y1, w, h):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.w = int(w)
        self.h = int(h)
        self.area = w * h

        self.x2 = self.x1 + self.w
        self.y2 = self.y1 + self.h

    # Get Intersection over Union value
    def computeIOU(self, bbox2):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(self.x1, bbox2.x1) 
        yA = max(self.y1, bbox2.y1) 
        xB = min(self.x2, bbox2.x2)
        yB = min(self.y2, bbox2.y2)

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	    # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (self.x2 - self.x1 + 1) * (self.y2- self.y1 + 1)
        boxBArea = (bbox2.x2 - bbox2.x1 + 1) * (bbox2.y2 - bbox2.y1 + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def extractSmallImage(self, image_full):
        return image_full[self.y1:self.y1+self.h, self.x1:self.x1+self.w]

class Detection(BoundingBox):

    def __init__(self, x1, y1, w, h, image_full, id,stamp):
        super().__init__(x1,y1,w,h) # call the super class constructor        
        self.id = id
        self.stamp = stamp
        self.image =self.extractSmallImage(image_full)
        self.assigned_to_tracker = False

    def draw(self, image_gui, color=(0, 0, 255)):
        cv2.rectangle(image_gui,(self.x1,self.y1),(self.x2, self.y2),color,3)

        image = cv2.putText(image_gui, '?FACE?', (self.x1, self.y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, color, 2, cv2.LINE_AA)

class Tracker():

    def __init__(self, detection, id, image):
        self.id = id
        self.template = None
        self.active = True
        self.bboxes = []
        self.detections = []
        self.input_read_control = False 
        self.time_since_last_detection = None

        # Create Tracker
        self.tracker = cv2.TrackerCSRT_create()

        self.time_since_last_detection = None
        self.addDetection(detection, image)

    def updateTime(self, stamp):
        self.time_since_last_detection = round(stamp-self.getLastDetectionStamp(),1)
        
        time_treshold = 10
        
        # =========================================
        # deactivate tracker with time treshold
        # ========================================
        
        bbox = self.bboxes[-1]

        # if the face tracker exceeds the image limits, the tracker's active 
        # time decreases (eventually predict a screen exit)

        if bbox.x2 > 590:       # Image 480 x 640 --> width darken frame = 50
            time_treshold = 2
        if bbox.x1 < 50:
            time_treshold = 2
        if bbox.y1 < 50:
            time_treshold = 2
        if bbox.y2 > 430:
            time_treshold = 2

      
        if self.time_since_last_detection > time_treshold:         
            self.active = False

        

    def getLastDetectionStamp(self):
        return self.detections[-1].stamp # get the last detection
        

    def drawLastDetection(self, image_gui, color=(0, 255, 0)):
        last_detection = self.detections[-1] # get the last detection

        cv2.rectangle(image_gui,(last_detection.x1,last_detection.y1),
                      (last_detection.x2, last_detection.y2),color,3)

        image = cv2.putText(image_gui, 'T' + str(self.id), 
                            (last_detection.x2-40, last_detection.y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, color, 2, cv2.LINE_AA)

    def draw(self, image_gui, color=(0, 255, 0)):

        #if not self.active:
        #    color = (100,100,100)
        if self.active:

            bbox = self.bboxes[-1] # get last bbox

            cv2.rectangle(image_gui,(bbox.x1,bbox.y1),(bbox.x2, bbox.y2),color,3)

            cv2.putText(image_gui, 'T' + str(self.id), 
                                (bbox.x1, bbox.y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, color, 2, cv2.LINE_AA)

            cv2.putText(image_gui, str(self.time_since_last_detection) + ' s', 
                                (bbox.x1, bbox.y1-25), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, color, 2, cv2.LINE_AA)


    def addDetection(self, detection, image):
        # Initiate Tracker
        self.tracker.init(image, (detection.x1, detection.y1, detection.w, detection.h))

        self.detections.append(detection)
        detection.assigned_to_tracker = True
        self.template = detection.image
        bbox = BoundingBox(detection.x1, detection.y1, detection.w, detection.h)
        self.bboxes.append(bbox)

    def track(self, image):

        ret, bbox = self.tracker.update(image)
        x1,y1,w,h = bbox

        bbox = BoundingBox(x1, y1, w, h)
        self.bboxes.append(bbox)

        # Update template using new bbox coordinates
        self.template = bbox.extractSmallImage(image)

    def getUserInput(self):
        self.input_read_control = True  # Set control variable
        self.id = input('What is your first name T' + str(self.id) + '?\n')  # Ask user input
        self.input_read_control = False  # Reset control variable

        engine = pyttsx3.init()
        engine.say("Hello " + str(self.id))
        engine.runAndWait()

        # Save image of new face detected in database
        img_rgb = cv2.cvtColor(self.template, cv2.COLOR_GRAY2BGR)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        cv2.imwrite('Image_Database/' + str(self.id) + '.jpg', img_rgb)
        
    def __str__(self):
        text =  'T' + str(self.id) + ' Detections = ['
        for detection in self.detections:
            text += str(detection.id) + ', '

        return text

class FaceRecognition():

    def __init__(self, path):
        self.path =  path
        self.images = []
        self.images_names = []
        self.list_of_files = []
        self.encode_list = []

    def readFilesInPath(self):
        # Read files in path
        self.list_of_files = os.listdir(self.path)
        print('List of Images in Database: ' + str(self.list_of_files))

        # Cycle through all files in the directory
        for file in self.list_of_files:
            current_image = cv2.imread(f'{self.path}/{file}')
            self.images.append(current_image)  # add image to the list
            self.images_names.append(os.path.splitext(file)[0])  # get image name

    def findEncodings(self):
        # Cycle through all images in list
        for img in self.images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            self.encode_list.append(encode)