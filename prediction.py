import argparse
import sys
from PIL import Image, ImageDraw
import face_recognition
import cv2
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import datetime
import os

# Put the media you want to perform the prediction on in this directory
mediapath = "media/"
# Initialize global variables
allfaces = []
images = []
videos = []

def detect_faces(input):
    num = 0
    faces = []
    # Load the test image with unknown faces into a numpy array
    # Find all the faces in the test image using the default HOG-based model
    face_locations = face_recognition.face_locations(input)  # default HOG, otherwise add , model='cnn')
    no_of_faces = len(face_locations)
    # print("Number of faces detected: ", no_of_faces)

    # Fill array with face encodings
    for i in range(no_of_faces):
        face_enc = face_recognition.face_encodings(input)[i]
        faces.append(face_enc)

    # Scale the data to the range between 0 and 1 before using PCA
    facesX = np.array(faces)
    scaler = MinMaxScaler()
    faces = scaler.fit_transform(facesX)
    # Load the model
    pca_model = pickle.load(open('../models/pca_model.pickle', 'rb'))
    svm_model = pickle.load(open('../models/svm_model.pickle', 'rb'))
    faces_found = []
    # Predict all the faces in the test image using the trained classifier
    # print("Found:")
    for i in range(no_of_faces):
        test_image = faces[i]  # faces = (number of samples, number of features)
        test_image = test_image.reshape(1, -1)
        test_image_pca = pca_model.transform(test_image)
        name = svm_model.predict(test_image_pca)
        # print(*name)
        faces_found.append(name)
    allfaces.append(faces_found)
    return faces_found, face_locations


def writeCSV():
    file = (str("Attendance.csv"))
    flat_list = [item for sublist in allfaces for item in sublist]
    faces_found = np.array(flat_list)
    df = pd.DataFrame(faces_found, columns=['Name'])
    df['Name'] = df['Name'].astype(str).str.replace("\']|\['", "")
    df[['Name', 'ID']] = df.Name.str.split("-", expand=True, )
    df.to_csv(file, mode='w')  # , index=False)
    faces_csv = pd.read_csv(file)
    faces_csv.drop_duplicates(subset="Name", inplace=True)
    faces_csv.drop(faces_csv.filter(regex="Unnamed"), axis=1, inplace=True)
    faces_csv.to_csv(file)
    return file


def imageR():
    count = 0
    for image in images:
        # Read image
        #image = np.array(image)
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get detected faces
        faces_found, face_locations = detect_faces(image)

        # Get names only without ID for drawing
        faces = [str(i).split('-') for i in faces_found]
        faces = np.array(faces)
        faces = np.delete(faces, 1,1)

        # Draw detected faces and names on group image
        i = 0
        #groupimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        groupimage = Image.fromarray(image)
        for face_location in face_locations:
            # Print the location of each face in this image
            top, right, bottom, left = face_location
            # Draw a box around the face
            draw = ImageDraw.Draw(groupimage)
            # Draw a box around the face using the Pillow module
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 0))
            # Draw a label with a name below the face
            text_width, text_height = draw.textsize(str(faces[i]))
            draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 0), outline=(0, 0, 0))
            draw.text((left - 20, bottom - text_height - 5), str(faces[i]), fill=(255, 255, 255, 255),align=left)
            i = i + 1
        del draw
        filename = " Attendance photo " + str(count) + " .jpg"
        groupimage.save(filename)
        ImageDraw.Draw(groupimage)
        groupimage.show()
        count += 1
    return 1

def videoR():
    i = 0
    for video in videos:
        names = []
        count = 0
        videopath = video
        vidcap = cv2.VideoCapture(videopath)
        success,image = vidcap.read()
        success = True
        while success:
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line
            success,image = vidcap.read()
            print ('Read a new frame: ', success)
            if success:
                faces_found, x = detect_faces(image)
                count = count + 1
                allfaces.extend(faces_found)
    return 1

###############################################################################

# For debugging, needs to be changed

media = os.listdir(mediapath)
if not os.path.exists(mediapath):
    sys.exit("No images/videos chosen, please choose some files and try again")
if os.path.exists(mediapath):
    ic = 0
    vc = 0
    ir = 0
    iv = 0
    for file in media:
        if file.lower().endswith(('.png','.jpg','.jpeg','.bmp')):
            images.append(mediapath + file)
            ic = 1
        elif file.lower().endswith(('.mp4','.wmv','.3gp','.')):
            videos.append(mediapath + file)
            vc = 1
        else:
            print("No files found or incorrect file type")
    if ic == 1:
        ir = imageR()
    if vc == 1:
        iv = videoR()
    if ir or iv == 1:
        filepath = writeCSV()




