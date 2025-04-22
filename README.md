# AttendanceCV: Facial Recognition Attendance System

## Project Overview

AttendanceCV is an advanced facial recognition system designed to automate attendance tracking in educational or professional environments. This project was developed as a graduation project for a Computer Science degree and demonstrates expertise in computer vision, machine learning, and practical software development.

## Key Features

- **Facial Recognition**: Utilizes state-of-the-art facial detection and recognition algorithms
- **Multi-format Support**: Works with both images and video inputs
- **Automated Attendance Tracking**: Generates CSV reports of attendance records
- **Visual Feedback**: Marks recognized faces with bounding boxes and labels in output images
- **Scalable Architecture**: Can be trained on new individuals with minimal effort

## Technical Implementation

### Technology Stack
- Python
- OpenCV (cv2)
- face_recognition library
- scikit-learn (SVM classifier, PCA)
- Pandas (data management)
- NumPy (numerical operations)
- PIL (image processing)

### Machine Learning Approach
The system employs a sophisticated facial recognition pipeline:
1. **Face Detection**: HOG-based face detection (with CNN option available)
2. **Feature Extraction**: Extracts facial encodings from detected faces
3. **Dimensionality Reduction**: PCA (Principal Component Analysis) to reduce feature dimensions while preserving 95% variance
4. **Classification**: Support Vector Machine (SVM) with RBF kernel for robust recognition
5. **Data Normalization**: MinMaxScaler ensures optimal model performance

## System Architecture

### Training Module (`training.py`)
- Processes labeled training images from a dataset directory
- Extracts facial features from each training image
- Trains SVM model on PCA-transformed facial features
- Saves trained models for later use in prediction

### Recognition Module (`prediction.py`)
- Processes input media (images or videos) from the media directory
- Detects and recognizes faces using the pre-trained models
- Generates visual output with labeled faces
- Creates attendance records in CSV format

## Usage

### Dataset Structure
The system expects training data organized as follows:
```
dataset/
├── Person Name -StudentID/
│   ├── Pic1.jpg
│   ├── Pic2.jpg
│   └── ...
├── Another Person -StudentID/
│   └── ...
```

### Training Process
1. Place labeled training data in the dataset directory
2. Run `training.py` to build and save the recognition models

### Recognition Process
1. Place test media in the media directory
2. Run `prediction.py` to process the media and generate attendance records
3. Review generated attendance CSV file and annotated output images

## Applications

This system can be deployed in various settings:
- **Educational Institutions**: Automate classroom attendance tracking
- **Corporate Environments**: Track employee presence in meetings or workplaces
- **Events and Conferences**: Monitor participant attendance
- **Security Systems**: Identify authorized personnel

## Future Enhancements

Potential areas for expansion include:
- Real-time recognition via webcam feed
- Web or mobile application interface
- Cloud-based storage for attendance records
- Integration with learning management systems or HR software

---

*This project demonstrates proficiency in machine learning, computer vision, and software development with practical real-world applications.*