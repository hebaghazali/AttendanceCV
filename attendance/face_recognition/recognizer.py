"""
Face recognition module for the AttendanceCV system
"""
import pickle
import os
import numpy as np
from typing import List, Optional, Tuple
from sklearn.preprocessing import MinMaxScaler

from attendance.utils.config import config
from attendance.utils.logger import logger
from attendance.face_recognition.detector import FaceDetector

class FaceRecognizer:
    """
    Class for recognizing faces using pre-trained models
    """
    
    def __init__(self, pca_model_path: str = None, svm_model_path: str = None):
        """
        Initialize the face recognizer with pre-trained models
        
        Args:
            pca_model_path (str, optional): Path to PCA model file
            svm_model_path (str, optional): Path to SVM model file
        """
        cfg = config.get_config()
        
        self.pca_model_path = pca_model_path or cfg['paths']['pca_model']
        self.svm_model_path = svm_model_path or cfg['paths']['svm_model']
        self.detector = FaceDetector()
        self.scaler = MinMaxScaler()
        
        # Load models
        self._load_models()
        
    def _load_models(self) -> None:
        """
        Load the PCA and SVM models from disk
        """
        try:
            # Ensure the models directory exists
            os.makedirs(os.path.dirname(self.pca_model_path), exist_ok=True)
            
            # Load the PCA model
            with open(self.pca_model_path, 'rb') as f:
                self.pca_model = pickle.load(f)
            logger.info(f"Loaded PCA model from {self.pca_model_path}")
            
            # Load the SVM model
            with open(self.svm_model_path, 'rb') as f:
                self.svm_model = pickle.load(f)
            logger.info(f"Loaded SVM model from {self.svm_model_path}")
            
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def recognize_faces(self, face_encodings: List) -> List[str]:
        """
        Recognize faces using the pre-trained models
        
        Args:
            face_encodings (List): List of face encodings
            
        Returns:
            List[str]: List of recognized face names
        """
        if not face_encodings:
            return []
            
        try:
            # Scale the face encodings
            faces_array = np.array(face_encodings)
            faces_scaled = self.scaler.fit_transform(faces_array)
            
            # Transform scaled faces using PCA
            faces_pca = self.pca_model.transform(faces_scaled)
            
            # Predict using SVM model
            predictions = self.svm_model.predict(faces_pca)
            
            logger.info(f"Recognized {len(predictions)} faces")
            return predictions
            
        except Exception as e:
            logger.error(f"Error in face recognition: {e}")
            return []
            
    def process_image(self, image: np.ndarray) -> Tuple[List[str], List]:
        """
        Process an image to detect and recognize faces
        
        Args:
            image (np.ndarray): Image to process
            
        Returns:
            Tuple[List[str], List]: Recognized names and face locations
        """
        # Detect faces
        face_locations, face_encodings = self.detector.detect_faces(image)
        
        if not face_encodings:
            return [], []
        
        # Recognize faces
        recognized_names = self.recognize_faces(face_encodings)
        
        return recognized_names, face_locations