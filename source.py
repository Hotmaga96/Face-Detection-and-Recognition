import numpy as np
import cv2
from os.path import isfile, join, split
from os import listdir

recognizer = cv2.face.LBPHFaceRecognizer_create()
image_names = []
image_labels = []

def train_recognizer(recognizer):
    folder_path = "Faces\\"
    image_file_path_list = [f for f in listdir(folder_path) if isfile(join(folder_path, f)) and f.endswith(".png")]
    images = []
    labels = []
    for image_file_path in image_file_path_list:
        image = cv2.imread(join(folder_path, image_file_path), 0)
        images.append(image)
        label = int(image_file_path.split(".")[0][-1])
        labels.append(label)
    recognizer.train(images, np.array(labels))
    return recognizer

while(True):
    choice = int(input("1-Facial Registration or 2-Facial Detection and Recognition or 3-Retrain Recognizer or 4-Exit"))
    if choice == 1:
        image_name = input("Enter Name of the Person: ")
        image_names.append(image_name)
        image_label = input("Enter Label to assign to the image: ")
        image_labels.append(image_label)
        lbp_classifier = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
        capture = cv2.VideoCapture(0)
        images = []
        while(True):
            ret, frame = capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = lbp_classifier.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xff == ord('y'):
                for (x, y, w, h) in faces:
                    images.append(gray[y: y + h, x: x + w])
                    print("Face has been registered")
                    break
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        capture.release()
        cv2.destroyAllWindows()
        f1 = open("ImageNames.txt", "a")
        f2 = open("ImageLabels.txt", "a")
        f1.write(image_name)
        f2.write(image_label)
        f1.close()
        f2.close()
        
    
        folder_path = 'Faces\\'
        qualifier = 0
        for image in images:
            folder_path_temp = folder_path + str(image_name) + str(qualifier) + str(image_label) + ".png"
            cv2.imwrite(folder_path_temp, image)
            qualifier = qualifier + 1
        recognizer = train_recognizer(recognizer)
        
    elif choice == 2:
        lbp_classifier = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
        capture = cv2.VideoCapture(0)
        while(True):
            ret, frame = capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = lbp_classifier.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                predicted_label, confidence_value = recognizer.predict(gray[y: y + h, x: x + w])
                text = image_names[predicted_label] + "," + str(confidence_value)
                cv2.putText(frame, text, (x - 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        capture.release()
        cv2.destroyAllWindows()
    elif choice == 3:
        f1 = open("ImageNames.txt", "r")
        f2 = open("ImageLabels.txt", "r")
        if len(image_names) == 0:
            for line in f1:
                image_names.append(line)
        if len(image_labels) == 0:
            for line in f2:
                image_labels.append(int(line))
        f1.close()
        f2.close()
        recognizer = train_recognizer(recognizer)
    elif choice == 4:
        break