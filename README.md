# AttendanceCV: Facial Recognition Attendance System

## Project Overview

AttendanceCV is an advanced facial recognition system designed to automate attendance tracking in educational or professional environments. This project was developed as a graduation project for a Computer Science degree and demonstrates expertise in computer vision, machine learning, and practical software development.

## Key Features

- **Facial Recognition**: Utilizes state-of-the-art facial detection and recognition algorithms
- **Multi-format Support**: Works with both images and video inputs
- **Automated Attendance Tracking**: Generates attendance records in multiple formats (CSV, Excel, JSON)
- **Visual Feedback**: Marks recognized faces with bounding boxes and labels in output media
- **Modular Architecture**: Well-structured, maintainable codebase with proper separation of concerns
- **Configurable**: Customizable settings via configuration files
- **Robust Logging**: Comprehensive logging for better troubleshooting

## Technical Implementation

### Technology Stack
- Python
- OpenCV (cv2)
- face_recognition library
- scikit-learn (SVM classifier, PCA)
- Pandas (data management)
- NumPy (numerical operations)
- PIL (image processing)
- PyYAML (configuration management)

### Machine Learning Approach
The system employs a sophisticated facial recognition pipeline:
1. **Face Detection**: HOG-based face detection (with CNN option available)
2. **Feature Extraction**: Extracts facial encodings from detected faces
3. **Dimensionality Reduction**: PCA (Principal Component Analysis) to reduce feature dimensions while preserving 95% variance
4. **Classification**: Support Vector Machine (SVM) with RBF kernel for robust recognition
5. **Data Normalization**: MinMaxScaler ensures optimal model performance

## System Architecture

The project follows a modular architecture with clear separation of concerns:

```
AttendanceCV/
├── attendance/            # Core package
│   ├── face_recognition/  # Face detection and recognition
│   ├── data/             # Data loading and processing
│   ├── models/           # Model training and processing
│   └── utils/            # Utilities (config, logging)
├── config/               # Configuration files
├── scripts/              # Entry point scripts
│   ├── train.py          # Training script
│   └── predict.py        # Prediction script
├── trained_models/       # Saved model files
├── dataset/              # Training data
├── media/                # Media for processing
├── logs/                 # Log files
└── tests/                # Test suite
```

### Core Components

1. **Configuration Management**: Load and manage settings from YAML config files
2. **Face Detection**: Process images/videos to detect face locations
3. **Face Recognition**: Recognize detected faces using trained models
4. **Data Loading**: Load and preprocess training and prediction data
5. **Model Training**: Train PCA and SVM models with hyperparameter tuning options
6. **Attendance Processing**: Generate and export attendance records

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/AttendanceCV.git
   cd AttendanceCV
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

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

### Configuration
Edit the configuration in `config/config.yaml` to customize settings:
- Data paths
- Model parameters
- Face detection options
- Processing options

### Training Process
Train the recognition models:
```
python scripts/train.py
```

Additional options:
```
python scripts/train.py --config custom_config.yaml  # Use custom config
python scripts/train.py --grid-search               # Enable hyperparameter tuning
```

### Recognition Process
Process media and generate attendance records:
```
python scripts/predict.py
```

Additional options:
```
python scripts/predict.py --media-dir /path/to/media   # Custom media directory
python scripts/predict.py --save-annotated            # Save annotated media
python scripts/predict.py --export-format excel       # Export format (csv, excel, json)
```

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
- API for integration with other systems

---

*This project demonstrates proficiency in machine learning, computer vision, and software development with practical real-world applications.*