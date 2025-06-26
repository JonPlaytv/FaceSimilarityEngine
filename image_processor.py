import cv2
import os
import logging
from PIL import Image, ImageOps
import numpy as np
from typing import Tuple, Optional

class ImageProcessor:
    """Image processing utilities for thumbnails and preprocessing"""
    
    def __init__(self):
        """Initialize image processor"""
        self.thumbnail_size = (150, 150)
        self.face_thumbnail_size = (100, 100)
        self.thumbnails_folder = 'static/thumbnails'
        
        # Ensure thumbnails directory exists
        os.makedirs(self.thumbnails_folder, exist_ok=True)
        
        logging.info("ImageProcessor initialized")
    
    def create_thumbnail(self, image_path: str, filename: str) -> str:
        """
        Create a thumbnail from an image file
        
        Args:
            image_path: Path to source image
            filename: Original filename
            
        Returns:
            Path to created thumbnail
        """
        try:
            # Generate thumbnail filename
            name, ext = os.path.splitext(filename)
            thumbnail_filename = f"{name}_thumb{ext}"
            thumbnail_path = os.path.join(self.thumbnails_folder, thumbnail_filename)
            
            # Create thumbnail using PIL for better quality
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Create thumbnail maintaining aspect ratio
                img.thumbnail(self.thumbnail_size, Image.Resampling.LANCZOS)
                
                # Create a square thumbnail with padding
                square_img = Image.new('RGB', self.thumbnail_size, (40, 40, 40))
                
                # Center the image
                offset = ((self.thumbnail_size[0] - img.size[0]) // 2,
                         (self.thumbnail_size[1] - img.size[1]) // 2)
                square_img.paste(img, offset)
                
                # Save thumbnail
                square_img.save(thumbnail_path, 'JPEG', quality=85)
            
            logging.debug(f"Created thumbnail: {thumbnail_path}")
            return thumbnail_path
            
        except Exception as e:
            logging.error(f"Error creating thumbnail for {image_path}: {str(e)}")
            return image_path  # Return original path as fallback
    
    def create_face_thumbnail(self, image: np.ndarray, face_box: Tuple[int, int, int, int], filename: str) -> str:
        """
        Create a thumbnail of a detected face
        
        Args:
            image: Source image as numpy array
            face_box: Face bounding box as (x, y, w, h)
            filename: Filename for the thumbnail
            
        Returns:
            Path to created face thumbnail
        """
        try:
            x, y, w, h = face_box
            
            # Extract face region with some padding
            padding = max(w, h) // 10  # 10% padding
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            
            face_region = image[y1:y2, x1:x2]
            
            if face_region.size == 0:
                logging.warning(f"Empty face region for {filename}")
                return ""
            
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            
            # Create PIL image
            pil_image = Image.fromarray(face_rgb)
            
            # Resize to standard face thumbnail size
            pil_image = pil_image.resize(self.face_thumbnail_size, Image.Resampling.LANCZOS)
            
            # Save thumbnail
            thumbnail_path = os.path.join(self.thumbnails_folder, filename)
            pil_image.save(thumbnail_path, 'JPEG', quality=90)
            
            logging.debug(f"Created face thumbnail: {thumbnail_path}")
            return thumbnail_path
            
        except Exception as e:
            logging.error(f"Error creating face thumbnail {filename}: {str(e)}")
            return ""
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better face detection
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        try:
            # Resize if image is too large
            height, width = image.shape[:2]
            max_dimension = 800
            
            if max(height, width) > max_dimension:
                if height > width:
                    new_height = max_dimension
                    new_width = int(width * (max_dimension / height))
                else:
                    new_width = max_dimension
                    new_height = int(height * (max_dimension / width))
                
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Enhance contrast
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            enhanced = cv2.merge([l, a, b])
            image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Denoise
            image = cv2.bilateralFilter(image, 9, 75, 75)
            
            return image
            
        except Exception as e:
            logging.error(f"Error preprocessing image: {str(e)}")
            return image  # Return original image as fallback
    
    def validate_image(self, image_path: str) -> bool:
        """
        Validate if image file is readable and valid
        
        Args:
            image_path: Path to image file
            
        Returns:
            True if image is valid, False otherwise
        """
        try:
            # Try to open with OpenCV
            image = cv2.imread(image_path)
            if image is None:
                return False
            
            # Try to open with PIL as well
            with Image.open(image_path) as img:
                img.verify()
            
            return True
            
        except Exception as e:
            logging.warning(f"Invalid image {image_path}: {str(e)}")
            return False
    
    def get_image_info(self, image_path: str) -> dict:
        """
        Get information about an image file
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with image information
        """
        try:
            with Image.open(image_path) as img:
                return {
                    'width': img.width,
                    'height': img.height,
                    'mode': img.mode,
                    'format': img.format,
                    'size_bytes': os.path.getsize(image_path)
                }
        except Exception as e:
            logging.error(f"Error getting image info for {image_path}: {str(e)}")
            return {}
