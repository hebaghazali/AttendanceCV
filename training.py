import os

import face_recognition
import cv2
import numpy as np
import pywt
import pickle

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

# Dataset is labeled by "Name -StudentID"
# For example:
# -Dataset
# --John Doe -20125
# ---Pic1.jpg
# ---Pic2.jpg
# ---...
# --Jane Doe -24145
# ---....


# Global Variables
face_data = []
labels = []
n_samples = 0  # Total number of Images
target_names = []  # Array to store the labels of the persons
label_count = 0
n_classes = 0

#############################################################################
## Load Training Data

# Training/dataset directory
path = "dataset/"
train_dir = os.listdir(path)

# Loop through each training image for the current person
for person in train_dir:
    pix = os.listdir(path + person + "/")
    # Loop through each training image for the current person
    for person_img in pix:
        imagex = cv2.imread(path + person + "/" + person_img)
        imagex = cv2.resize(imagex, (512, 512))
        image = cv2.cvtColor(imagex, cv2.COLOR_BGR2RGB)
        #############################################################################
        # Face detection
        face_locations = face_recognition.face_locations(image)  # default HOG, otherwise add , model='cnn')
        if len(face_locations) == 1:
            face_enc = face_recognition.face_encodings(image)[0]
            # Add face encoding for current image with corresponding label (name) to the training data
            face_data.append(face_enc)
            labels.append(person)
            n_samples = n_samples + 1
    label_count = label_count + 1
    target_names.append(person)

n_classes = len(target_names)
face_data = np.array(face_data)
n_features = face_data.shape[1]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# Scale the data to the range between 0 and 1 before using PCA
scaler = MinMaxScaler()
data_rescaled = scaler.fit_transform(face_data)
#############################################################################
### Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
### dataset): unsupervised feature extraction / dimensionality reduction
pca = PCA(n_components=0.95) # Variance set to 95%
pca.fit(data_rescaled)
face_data_train_pca = pca.transform(data_rescaled)
###############################################################################
### Create and train the SVC classifier
clf = SVC(kernel='rbf', class_weight='balanced', C=1e3, gamma=0.0005)
clf.fit(face_data_train_pca,labels)
### Or use GridSearch for tuning, better for larger datasets
# param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
#               'gamma': [0.0005, 0.001, 0.005, 0.01, 0.1], }
# clf = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
# clf.fit(face_data_train_pca,labels_train)
# print("Best estimator found by grid search:")
# print(clf.best_estimator_)
##############################################################################
# Save the model to use it later for recognition
if not os.path.exists('models/'):
    os.mkdir('models/')
pca_file = 'models/pca_model.pickle'
clf_file = 'models/svm_model.pickle'
pickle.dump(pca, open(pca_file, 'wb'))
pickle.dump(clf, open(clf_file, 'wb'))
###############################################################################


