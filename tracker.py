import os
import cv2
import pyttsx3
from detection import BoundingBox
from colorama import Fore, Back, Style

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
        if self.active:
            bbox = self.bboxes[-1] # get last bbox

            cv2.rectangle(image_gui,(bbox.x1,bbox.y1),(bbox.x2, bbox.y2),color,3)

            cv2.rectangle(image_gui,(bbox.x1,bbox.y1), (bbox.x2, bbox.y1-35),color, -1)

            cv2.putText(image_gui, 'T ' + str(self.id), 
                                (bbox.x1, bbox.y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, (255,255,255), 2, cv2.LINE_AA)

            cv2.putText(image_gui, str(self.time_since_last_detection) + ' s', 
                                (bbox.x2-60, bbox.y2-10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, color, 2, cv2.LINE_AA)


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
        self.id = input(Fore.GREEN + 'What is your first name T' + str(self.id) + '?\n' + Style.RESET_ALL)  # Ask user input
        # self.input_read_control = False  # Reset control variable

        # Save image of new face detected in database
        img_rgb = cv2.cvtColor(self.template, cv2.COLOR_GRAY2BGR)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        cv2.imwrite('Image_Database/' + str(self.id) + '.jpg', img_rgb)

        # Greet Person
        self.greetPerson()

        # Print database data
        list_of_files = os.listdir('Image_Database')
        n_people =  len(list_of_files)

        print(Fore.RED  + '\nNumber of People in Database: ' + str(n_people) + Style.RESET_ALL)
        print(Fore.RED  + 'List of Images in Database: ' + str(list_of_files) + '\n' + Style.RESET_ALL)

        self.input_read_control = False  # Reset control variable

    def greetPerson(self):
        engine = pyttsx3.init()
        engine.say("Hello " + str(self.id))
        engine.runAndWait()
        
    def __str__(self):
        text =  'T' + str(self.id) + ' Detections = ['
        for detection in self.detections:
            text += str(detection.id) + ', '

        return text