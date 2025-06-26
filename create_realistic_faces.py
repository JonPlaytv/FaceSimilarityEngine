#!/usr/bin/env python3
"""
Create realistic face images that can be detected by InsightFace
Using simple geometric patterns that represent face-like structures
"""

import os
import numpy as np
from PIL import Image, ImageDraw
import logging

def create_face_like_image(output_path: str, face_id: int):
    """Create a simple face-like image that InsightFace can detect"""
    
    # Create a 200x200 image with skin-like color
    width, height = 200, 200
    img = Image.new('RGB', (width, height), color=(230, 200, 180))
    draw = ImageDraw.Draw(img)
    
    # Face outline (oval)
    face_margin = 30
    draw.ellipse([face_margin, face_margin, width-face_margin, height-face_margin], 
                fill=(220, 190, 170), outline=(200, 170, 150), width=2)
    
    # Eyes
    eye_y = height // 3
    eye_size = 15
    left_eye_x = width // 3
    right_eye_x = 2 * width // 3
    
    # Left eye
    draw.ellipse([left_eye_x-eye_size, eye_y-eye_size//2, 
                 left_eye_x+eye_size, eye_y+eye_size//2], 
                fill=(50, 50, 50))
    draw.ellipse([left_eye_x-eye_size//2, eye_y-eye_size//4, 
                 left_eye_x+eye_size//2, eye_y+eye_size//4], 
                fill=(255, 255, 255))
    
    # Right eye
    draw.ellipse([right_eye_x-eye_size, eye_y-eye_size//2, 
                 right_eye_x+eye_size, eye_y+eye_size//2], 
                fill=(50, 50, 50))
    draw.ellipse([right_eye_x-eye_size//2, eye_y-eye_size//4, 
                 right_eye_x+eye_size//2, eye_y+eye_size//4], 
                fill=(255, 255, 255))
    
    # Nose
    nose_x = width // 2
    nose_y = height // 2
    nose_points = [
        (nose_x-5, nose_y-10),
        (nose_x+5, nose_y-10),
        (nose_x+8, nose_y+5),
        (nose_x-8, nose_y+5)
    ]
    draw.polygon(nose_points, fill=(210, 180, 160))
    
    # Mouth
    mouth_y = 2 * height // 3
    mouth_width = 30
    draw.arc([nose_x-mouth_width, mouth_y-10, 
             nose_x+mouth_width, mouth_y+10], 
            start=0, end=180, fill=(150, 100, 100), width=3)
    
    # Add some variation based on face_id
    if face_id % 2 == 0:
        # Add eyebrows for even IDs
        draw.arc([left_eye_x-eye_size-5, eye_y-eye_size-5, 
                 left_eye_x+eye_size+5, eye_y-eye_size+5], 
                start=0, end=180, fill=(100, 70, 50), width=3)
        draw.arc([right_eye_x-eye_size-5, eye_y-eye_size-5, 
                 right_eye_x+eye_size+5, eye_y-eye_size+5], 
                start=0, end=180, fill=(100, 70, 50), width=3)
    
    # Save the image
    img.save(output_path, 'PNG')
    logging.info(f"Created face-like image: {output_path}")

def create_realistic_dataset():
    """Create realistic face-like images for the dataset"""
    dataset_dir = 'static/dataset'
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Remove old sample files
    for i in range(1, 6):
        old_file = os.path.join(dataset_dir, f'sample_face_{i}.png')
        if os.path.exists(old_file):
            os.remove(old_file)
    
    # Create new realistic face images
    for i in range(1, 6):
        output_path = os.path.join(dataset_dir, f'realistic_face_{i}.png')
        create_face_like_image(output_path, i)
    
    logging.info("Created 5 realistic face-like images for dataset")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_realistic_dataset()