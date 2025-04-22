"""
Model training module for the AttendanceCV system
"""
import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Tuple

from attendance.utils.config import config
from attendance.utils.logger import logger
from attendance.data.loader import DataLoader

class ModelTrainer:
    """
    Class for training face recognition models
    """
    
    def __init__(self, use_grid_search: bool = False):
        """
        Initialize the model trainer
        
        Args:
            use_grid_search (bool): Whether to use grid search for hyperparameter tuning
        """
        self.cfg = config.get_config()
        self.use_grid_search = use_grid_search
        self.models_dir = self.cfg['paths']['models_dir']
        
        # Ensure models directory exists
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize the scaler
        self.scaler = MinMaxScaler()
        
    def train_models(self, face_data: np.ndarray, labels: list) -> Dict[str, object]:
        """
        Train PCA and SVM models on the face data
        
        Args:
            face_data (np.ndarray): Face encoding data
            labels (list): Labels for the face data
            
        Returns:
            Dict[str, object]: Trained models
        """
        if len(face_data) == 0 or len(labels) == 0:
            logger.error("Cannot train on empty data")
            return {}
            
        logger.info(f"Training models on {len(face_data)} samples")
        
        try:
            # Scale the face data
            data_scaled = self.scaler.fit_transform(face_data)
            
            # Train PCA model
            pca = self._train_pca(data_scaled)
            
            # Transform data using PCA
            face_data_pca = pca.transform(data_scaled)
            
            # Train SVM model
            svm = self._train_svm(face_data_pca, labels)
            
            # Save models
            self._save_models(pca, svm)
            
            return {'pca': pca, 'svm': svm}
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return {}
    
    def _train_pca(self, data_scaled: np.ndarray) -> PCA:
        """
        Train a PCA model for dimensionality reduction
        
        Args:
            data_scaled (np.ndarray): Scaled face data
            
        Returns:
            PCA: Trained PCA model
        """
        n_components = self.cfg['model']['pca']['n_components']
        
        logger.info(f"Training PCA model with n_components={n_components}")
        pca = PCA(n_components=n_components)
        pca.fit(data_scaled)
        
        # Log explained variance
        cumulative_variance = np.sum(pca.explained_variance_ratio_)
        n_components_used = len(pca.explained_variance_ratio_)
        logger.info(f"PCA using {n_components_used} components explaining {cumulative_variance:.2f} of variance")
        
        return pca
        
    def _train_svm(self, face_data_pca: np.ndarray, labels: list) -> SVC:
        """
        Train an SVM model for face recognition
        
        Args:
            face_data_pca (np.ndarray): PCA-transformed face data
            labels (list): Labels for the face data
            
        Returns:
            SVC: Trained SVM model
        """
        if self.use_grid_search:
            logger.info("Training SVM model with grid search")
            param_grid = {
                'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                'gamma': [0.0005, 0.001, 0.005, 0.01, 0.1],
            }
            svm = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid, cv=5)
            svm.fit(face_data_pca, labels)
            
            # Log best parameters
            logger.info(f"Best parameters found: {svm.best_params_}")
            return svm.best_estimator_
        else:
            # Use the parameters from the config file
            kernel = self.cfg['model']['svm']['kernel']
            class_weight = self.cfg['model']['svm']['class_weight']
            c = self.cfg['model']['svm']['C']
            gamma = self.cfg['model']['svm']['gamma']
            
            logger.info(f"Training SVM model with kernel={kernel}, C={c}, gamma={gamma}")
            svm = SVC(kernel=kernel, class_weight=class_weight, C=c, gamma=gamma)
            svm.fit(face_data_pca, labels)
            
            return svm
            
    def _save_models(self, pca: PCA, svm: SVC) -> None:
        """
        Save the trained models to disk
        
        Args:
            pca (PCA): Trained PCA model
            svm (SVC): Trained SVM model
        """
        pca_path = self.cfg['paths']['pca_model']
        svm_path = self.cfg['paths']['svm_model']
        
        logger.info(f"Saving models to {self.models_dir}")
        
        try:
            # Save PCA model
            with open(pca_path, 'wb') as f:
                pickle.dump(pca, f)
            logger.info(f"Saved PCA model to {pca_path}")
            
            # Save SVM model
            with open(svm_path, 'wb') as f:
                pickle.dump(svm, f)
            logger.info(f"Saved SVM model to {svm_path}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise
            
    def run_training_pipeline(self) -> Tuple[Dict[str, object], int]:
        """
        Run the complete training pipeline
        
        Returns:
            Tuple[Dict[str, object], int]: Trained models and number of samples
        """
        # Load training data
        data_loader = DataLoader()
        face_data, labels, target_names, n_samples = data_loader.load_training_data()
        
        if n_samples == 0:
            logger.error("No training samples found")
            return {}, 0
        
        # Train models
        models = self.train_models(face_data, labels)
        
        return models, n_samples