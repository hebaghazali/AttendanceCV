"""
Data loading module for the AttendanceCV system
"""
import os
import cv2
import numpy as np
from typing import Tuple, List, Dict
from pathlib import Path

from attendance.utils.config import config
from attendance.utils.logger import logger
from attendance.face_recognition.detector import FaceDetector

class DataLoader:
    """
    Class for loading data for training and prediction
    """
    
    def __init__(self, dataset_path: str = None, media_path: str = None):
        """
        Initialize the data loader
        
        Args:
            dataset_path (str, optional): Path to the training dataset
            media_path (str, optional): Path to the media files for prediction
        """
        cfg = config.get_config()
        
        self.dataset_path = dataset_path or cfg['paths']['dataset']
        self.media_path = media_path or cfg['paths']['media']
        self.detector = FaceDetector()
        
        # Ensure directories exist
        os.makedirs(self.dataset_path, exist_ok=True)
        os.makedirs(self.media_path, exist_ok=True)
        
    def load_training_data(self) -> Tuple[np.ndarray, List, List, int]:
        """
        Load training data from dataset directory
        
        Returns:
            Tuple[np.ndarray, List, List, int]: Face data, labels, target names, and number of samples
        """
        logger.info(f"Loading training data from {self.dataset_path}")
        
        face_data = []
        labels = []
        target_names = []
        n_samples = 0
        
        try:
            # Check if dataset directory exists and has subdirectories
            if not os.path.exists(self.dataset_path):
                raise FileNotFoundError(f"Dataset directory not found: {self.dataset_path}")
                
            train_dir = os.listdir(self.dataset_path)
            if not train_dir:
                logger.warning("No subdirectories found in dataset directory")
                return np.array([]), [], [], 0
                
            # Loop through each person's directory
            for person in train_dir:
                person_dir = os.path.join(self.dataset_path, person)
                
                # Skip if not a directory
                if not os.path.isdir(person_dir):
                    continue
                    
                logger.info(f"Processing images for {person}")
                person_images = os.listdir(person_dir)
                
                # Loop through each image for the current person
                for person_img in person_images:
                    img_path = os.path.join(person_dir, person_img)
                    
                    # Skip if not an image file
                    if not person_img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        continue
                    
                    # Load and preprocess image
                    image = self.detector.preprocess_image(img_path)
                    if image is None:
                        continue
                        
                    # Resize image to standard dimensions
                    cfg = config.get_config()
                    width = cfg['image_processing']['resize_width']
                    height = cfg['image_processing']['resize_height']
                    image = cv2.resize(image, (width, height))
                    
                    # Detect faces
                    face_locations, face_encodings = self.detector.detect_faces(image)
                    
                    # Skip if no face or multiple faces detected
                    if len(face_encodings) != 1:
                        logger.warning(f"Skipping {img_path}: Expected 1 face, found {len(face_encodings)}")
                        continue
                        
                    # Add face encoding and label
                    face_data.append(face_encodings[0])
                    labels.append(person)
                    n_samples += 1
                    
                # Add to target names
                target_names.append(person)
                    
            logger.info(f"Loaded {n_samples} face samples for {len(target_names)} people")
            return np.array(face_data), labels, target_names, n_samples
            
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return np.array([]), [], [], 0
            
    def load_media_files(self) -> Dict[str, List[str]]:
        """
        Load media files for prediction
        
        Returns:
            Dict[str, List[str]]: Dictionary with image and video file paths
        """
        logger.info(f"Loading media files from {self.media_path}")
        
        images = []
        videos = []
        
        try:
            # Check if media directory exists
            if not os.path.exists(self.media_path):
                raise FileNotFoundError(f"Media directory not found: {self.media_path}")
                
            media_files = os.listdir(self.media_path)
            
            # Categorize files by type
            for file in media_files:
                file_path = os.path.join(self.media_path, file)
                
                # Skip if not a file
                if not os.path.isfile(file_path):
                    continue
                    
                # Categorize by extension
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    images.append(file_path)
                elif file.lower().endswith(('.mp4', '.avi', '.mov', '.wmv', '.3gp')):
                    videos.append(file_path)
                    
            logger.info(f"Loaded {len(images)} images and {len(videos)} videos")
            return {'images': images, 'videos': videos}
            
        except Exception as e:
            logger.error(f"Error loading media files: {e}")
            return {'images': [], 'videos': []}