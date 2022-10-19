import cv2
import pyttsx3 

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

    def __init__(self, x1, y1, w, h, image_full, id):
        super().__init__(x1,y1,w,h) # call the super class constructor        
        self.id = id
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

        # Create Tracker
        self.tracker = cv2.TrackerCSRT_create()

        self.time_since_last_detection = None
        self.addDetection(detection, image)

    def updateTime(self, stamp):
        self.time_since_last_detection = round(stamp-self.getLastDetectionStamp(),1)

        if self.time_since_last_detection > 2: # deactivate tracker        
            self.active = False

    def drawLastDetection(self, image_gui, color=(0, 255, 0)):
        last_detection = self.detections[-1] # get the last detection

        cv2.rectangle(image_gui,(last_detection.x1,last_detection.y1),
                      (last_detection.x2, last_detection.y2),color,3)

        image = cv2.putText(image_gui, 'T' + str(self.id), 
                            (last_detection.x2-40, last_detection.y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, color, 2, cv2.LINE_AA)

    def draw(self, image_gui, color=(0, 255, 0)):

        if not self.active:
            color = (100,100,100)

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

    # Teste
    def getUserInput(self):
        self.input_read_control = True  # Set control variable
        self.id = input('What is your name T' + str(self.id) + '?\n')  # Ask user input
        self.input_read_control = False  # Reset control variable

        engine = pyttsx3.init()
        engine.say("Hello " + self.id )
        engine.runAndWait()
    # Teste
        
    def __str__(self):
        text =  'T' + str(self.id) + ' Detections = ['
        for detection in self.detections:
            text += str(detection.id) + ', '

        return text