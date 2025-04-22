"""
Configuration handler for the AttendanceCV system
"""
import os
import yaml
from pathlib import Path

class Config:
    """
    Configuration class for loading and accessing project settings
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._config = None
        self._initialized = True
        
    def load_config(self, config_path=None):
        """
        Load configuration from YAML file
        
        Args:
            config_path (str, optional): Path to config file. Defaults to None.
        
        Returns:
            dict: Configuration dictionary
        """
        if config_path is None:
            # Find the config relative to the project root
            project_root = Path(__file__).parent.parent.parent
            config_path = os.path.join(project_root, 'config', 'config.yaml')
        
        try:
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
                return self._config
        except Exception as e:
            raise Exception(f"Failed to load configuration: {e}")
    
    def get_config(self):
        """
        Get the loaded configuration
        
        Returns:
            dict: Configuration dictionary
        """
        if self._config is None:
            self.load_config()
        return self._config

# Create a singleton instance
config = Config()