import os
import cv2
import face_recognition

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