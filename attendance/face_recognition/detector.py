"""
Face detection module for the AttendanceCV system
"""
import face_recognition
import cv2
import numpy as np
from typing import Tuple, List, Optional

from attendance.utils.config import config
from attendance.utils.logger import logger

class FaceDetector:
    """
    Class for detecting faces in images and videos
    """
    
    def __init__(self, detection_model: str = None):
        """
        Initialize the face detector
        
        Args:
            detection_model (str, optional): Face detection model ('hog' or 'cnn')
        """
        cfg = config.get_config()
        self.detection_model = detection_model or cfg['face_detection']['model']
        self.scale_factor = cfg['face_detection']['scale_factor']
        logger.info(f"Initialized face detector with model: {self.detection_model}")
    
    def detect_faces(self, image: np.ndarray) -> Tuple[List, List]:
        """
        Detect faces in an image
        
        Args:
            image (np.ndarray): Image to detect faces in
            
        Returns:
            Tuple[List, List]: Face locations and face encodings
        """
        try:
            # Find face locations
            face_locations = face_recognition.face_locations(
                image, 
                model=self.detection_model
            )
            
            # If no faces found, return empty lists
            if not face_locations:
                logger.debug("No faces detected in the image")
                return [], []
                
            # Get face encodings
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            logger.info(f"Detected {len(face_locations)} faces in the image")
            return face_locations, face_encodings
            
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return [], []
    
    def preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load and preprocess an image for face detection
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            np.ndarray: Preprocessed image
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
                
            # Convert BGR to RGB (face_recognition uses RGB)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            return None