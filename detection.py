import cv2

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