"""
Prediction script for the AttendanceCV system
"""
import sys
import os
import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from attendance.face_recognition.recognizer import FaceRecognizer
from attendance.data.loader import DataLoader
from attendance.models.processor import AttendanceProcessor
from attendance.utils.config import config
from attendance.utils.logger import logger

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process media files for attendance')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--media-dir', type=str, help='Path to media directory')
    parser.add_argument('--save-annotated', action='store_true', help='Save annotated images/videos')
    parser.add_argument('--export-format', type=str, choices=['csv', 'excel', 'json'], default='csv', 
                        help='Format to export attendance report')
    return parser.parse_args()

def process_image(image_path, recognizer, save_annotated=False):
    """
    Process a single image for face recognition
    
    Args:
        image_path (str): Path to the image
        recognizer (FaceRecognizer): Face recognizer instance
        save_annotated (bool): Whether to save the annotated image
        
    Returns:
        list: List of recognized face names
    """
    logger.info(f"Processing image: {image_path}")
    
    # Load and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return []
        
    # Convert to RGB (face_recognition uses RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image
    recognized_names, face_locations = recognizer.process_image(image)
    
    if not recognized_names:
        logger.warning(f"No faces recognized in {image_path}")
        return []
    
    # If requested, save the annotated image
    if save_annotated:
        # Convert to PIL Image for drawing
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        
        # Draw bounding box and name for each detected face
        for (name, (top, right, bottom, left)) in zip(recognized_names, face_locations):
            # Parse the name to remove ID if present
            display_name = name.split("-")[0] if "-" in name else name
            
            # Draw a box around the face
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0), width=2)
            
            # Draw a label with the name below the face
            text_width, text_height = draw.textsize(display_name)
            draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), 
                          fill=(0, 255, 0), outline=(0, 255, 0))
            draw.text((left + 6, bottom - text_height - 5), display_name, fill=(0, 0, 0))
        
        # Save the annotated image
        output_path = f"annotated_{os.path.basename(image_path)}"
        pil_image.save(output_path)
        logger.info(f"Saved annotated image to {output_path}")
    
    logger.info(f"Recognized {len(recognized_names)} faces in {image_path}")
    return recognized_names

def process_video(video_path, recognizer, save_annotated=False):
    """
    Process a video for face recognition by sampling frames
    
    Args:
        video_path (str): Path to the video
        recognizer (FaceRecognizer): Face recognizer instance
        save_annotated (bool): Whether to save the annotated video
        
    Returns:
        list: List of recognized face names
    """
    logger.info(f"Processing video: {video_path}")
    
    all_recognized_names = []
    cfg = config.get_config()
    frame_interval = cfg['video_processing']['frame_interval']  # Seconds between frames
    
    try:
        # Open the video file
        video_capture = cv2.VideoCapture(video_path)
        
        if not video_capture.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return []
            
        # Get video properties
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        logger.info(f"Video properties: {fps} FPS, {frame_count} frames, {duration:.2f} seconds")
        
        # Process frames at intervals
        frames_to_process = int(fps * frame_interval)
        
        # Output video writer for annotated video
        out = None
        if save_annotated:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output_path = f"annotated_{os.path.basename(video_path)}"
            width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_number = 0
        while True:
            ret, frame = video_capture.read()
            
            if not ret:
                break
                
            # Process only every nth frame
            if frame_number % frames_to_process == 0:
                # Convert BGR to RGB for face_recognition
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the frame
                recognized_names, face_locations = recognizer.process_image(rgb_frame)
                
                if recognized_names:
                    all_recognized_names.extend(recognized_names)
                    
                    # If saving annotated video, draw on the frame
                    if save_annotated and out:
                        for (name, (top, right, bottom, left)) in zip(recognized_names, face_locations):
                            # Parse the name to remove ID if present
                            display_name = name.split("-")[0] if "-" in name else name
                            
                            # Draw a rectangle around the face
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                            
                            # Draw a label with the name below the face
                            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                            cv2.putText(frame, display_name, (left + 6, bottom - 6), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
                
                # Write the frame to output video if annotating
                if save_annotated and out:
                    out.write(frame)
                    
            frame_number += 1
            
        # Clean up
        video_capture.release()
        if out:
            out.release()
            logger.info(f"Saved annotated video to {output_path}")
            
        # Remove duplicates
        all_recognized_names = list(set(all_recognized_names))
        logger.info(f"Recognized {len(all_recognized_names)} unique faces in {video_path}")
        
        return all_recognized_names
        
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {e}")
        return []

def main():
    """Main prediction function"""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    if args.config:
        config.load_config(args.config)
    else:
        config.load_config()
    
    # Override media directory if provided
    if args.media_dir:
        config.get_config()['paths']['media'] = args.media_dir
    
    logger.info("Starting attendance prediction...")
    
    # Initialize components
    data_loader = DataLoader()
    recognizer = FaceRecognizer()
    attendance_processor = AttendanceProcessor()
    
    # Load media files
    media_files = data_loader.load_media_files()
    
    if not media_files['images'] and not media_files['videos']:
        logger.error("No media files found for processing")
        return 1
    
    # Store all recognized faces
    all_recognized_faces = []
    
    # Process images
    for image_path in media_files['images']:
        recognized_names = process_image(image_path, recognizer, args.save_annotated)
        if recognized_names:
            all_recognized_faces.append(recognized_names)
    
    # Process videos
    for video_path in media_files['videos']:
        recognized_names = process_video(video_path, recognizer, args.save_annotated)
        if recognized_names:
            all_recognized_faces.append(recognized_names)
    
    # Generate attendance records
    if all_recognized_faces:
        attendance_file = attendance_processor.process_recognized_faces(all_recognized_faces)
        
        if attendance_file:
            # Export in requested format
            export_file = attendance_processor.export_attendance_report(
                format=args.export_format
            )
            if export_file:
                logger.info(f"Attendance exported to {export_file}")
            
            logger.info("Attendance prediction completed successfully")
            return 0
    
    logger.error("Attendance prediction failed: no faces recognized")
    return 1
    
if __name__ == "__main__":
    sys.exit(main())