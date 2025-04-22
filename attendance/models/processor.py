"""
Attendance processing module for the AttendanceCV system
"""
import os
import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime

from attendance.utils.config import config
from attendance.utils.logger import logger

class AttendanceProcessor:
    """
    Class for processing attendance records
    """
    
    def __init__(self):
        """
        Initialize the attendance processor
        """
        self.cfg = config.get_config()
        self.attendance_file = self.cfg['paths']['attendance_file']
        
    def process_recognized_faces(self, all_recognized_faces: List[List[str]]) -> str:
        """
        Process recognized faces and generate attendance records
        
        Args:
            all_recognized_faces (List[List[str]]): List of recognized face names from all images/frames
            
        Returns:
            str: Path to the generated attendance file
        """
        logger.info("Processing recognized faces for attendance")
        
        try:
            # Flatten the list of recognized faces
            flat_list = [item for sublist in all_recognized_faces for item in sublist]
            
            if not flat_list:
                logger.warning("No faces recognized for attendance")
                return ""
            
            # Create a DataFrame from the face names
            df = pd.DataFrame(flat_list, columns=['Name'])
            
            # Clean up the names
            df['Name'] = df['Name'].astype(str).str.replace(r"\']|\['", "", regex=True)
            
            # Split name and ID
            df[['Name', 'ID']] = df.Name.str.split("-", expand=True)
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df['Timestamp'] = timestamp
            
            # Save to CSV
            df.to_csv(self.attendance_file, mode='w', index=False)
            
            # Remove duplicates (keeping first occurrence)
            attendance_df = pd.read_csv(self.attendance_file)
            attendance_df.drop_duplicates(subset="Name", inplace=True)
            
            # Remove unnamed columns
            if any(col.startswith('Unnamed:') for col in attendance_df.columns):
                attendance_df.drop(attendance_df.filter(regex="Unnamed"), axis=1, inplace=True)
            
            # Save cleaned data
            attendance_df.to_csv(self.attendance_file, index=False)
            
            logger.info(f"Attendance saved to {self.attendance_file}")
            return self.attendance_file
            
        except Exception as e:
            logger.error(f"Error processing attendance: {e}")
            return ""
    
    def get_attendance_report(self) -> pd.DataFrame:
        """
        Get the attendance report as a DataFrame
        
        Returns:
            pd.DataFrame: Attendance records
        """
        try:
            if not os.path.exists(self.attendance_file):
                logger.warning(f"Attendance file not found: {self.attendance_file}")
                return pd.DataFrame()
                
            return pd.read_csv(self.attendance_file)
            
        except Exception as e:
            logger.error(f"Error reading attendance file: {e}")
            return pd.DataFrame()
            
    def export_attendance_report(self, format: str = 'csv', output_path: str = None) -> str:
        """
        Export the attendance report in the specified format
        
        Args:
            format (str): Export format ('csv', 'excel', 'json')
            output_path (str, optional): Path to save the exported file
            
        Returns:
            str: Path to the exported file
        """
        try:
            df = self.get_attendance_report()
            
            if df.empty:
                return ""
                
            # Generate output path if not provided
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"attendance_report_{timestamp}"
            
            # Export according to format
            if format.lower() == 'csv':
                output_file = f"{output_path}.csv"
                df.to_csv(output_file, index=False)
            elif format.lower() == 'excel':
                output_file = f"{output_path}.xlsx"
                df.to_excel(output_file, index=False)
            elif format.lower() == 'json':
                output_file = f"{output_path}.json"
                df.to_json(output_file, orient='records')
            else:
                logger.error(f"Unsupported export format: {format}")
                return ""
                
            logger.info(f"Attendance report exported to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error exporting attendance report: {e}")
            return ""