import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional

class FaceDetector:
    """Fallback face detection using OpenCV Haar Cascades"""
    
    def __init__(self):
        """Initialize face detector with OpenCV's Haar Cascade"""
        # Load the face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Parameters for face detection
        self.scale_factor = 1.1
        self.min_neighbors = 5
        self.min_size = (30, 30)
        
        logging.info("FaceDetector (OpenCV fallback) initialized successfully")
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of face bounding boxes as (x, y, w, h) tuples
        """
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size
            )
            
            # Convert to list of tuples
            face_list = [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
            
            logging.debug(f"Detected {len(face_list)} faces")
            return face_list
            
        except Exception as e:
            logging.error(f"Error detecting faces: {str(e)}")
            return []
    
    def generate_embedding(self, image: np.ndarray, face_box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Generate a face embedding using basic computer vision features
        
        Args:
            image: Input image as numpy array
            face_box: Face bounding box as (x, y, w, h)
            
        Returns:
            Face embedding as numpy array or None if failed
        """
        try:
            x, y, w, h = face_box
            
            # Extract face region
            face_roi = image[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                logging.warning("Empty face region extracted")
                return None
            
            # Convert to grayscale
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Resize to standard size for consistent features
            standard_size = (100, 100)
            resized_face = cv2.resize(gray_face, standard_size)
            
            # Generate basic features combining multiple techniques
            features = []
            
            # 1. LBP (Local Binary Pattern) features
            lbp_features = self._extract_lbp_features(resized_face)
            features.extend(lbp_features)
            
            # 2. Histogram features
            hist_features = self._extract_histogram_features(resized_face)
            features.extend(hist_features)
            
            # 3. Edge features
            edge_features = self._extract_edge_features(resized_face)
            features.extend(edge_features)
            
            # 4. Geometric features
            geometric_features = self._extract_geometric_features(resized_face)
            features.extend(geometric_features)
            
            # Convert to numpy array and normalize
            embedding = np.array(features, dtype=np.float32)
            
            # L2 normalization
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            logging.debug(f"Generated embedding with {len(embedding)} features")
            return embedding
            
        except Exception as e:
            logging.error(f"Error generating embedding: {str(e)}")
            return None
    
    def _extract_lbp_features(self, face: np.ndarray) -> List[float]:
        """Extract Local Binary Pattern features"""
        try:
            # Simple LBP implementation
            height, width = face.shape
            lbp = np.zeros((height-2, width-2), dtype=np.uint8)
            
            for i in range(1, height-1):
                for j in range(1, width-1):
                    center = face[i, j]
                    binary_string = ''
                    
                    # 8-neighbor pattern
                    neighbors = [
                        face[i-1, j-1], face[i-1, j], face[i-1, j+1],
                        face[i, j+1], face[i+1, j+1], face[i+1, j],
                        face[i+1, j-1], face[i, j-1]
                    ]
                    
                    for neighbor in neighbors:
                        binary_string += '1' if neighbor >= center else '0'
                    
                    lbp[i-1, j-1] = int(binary_string, 2)
            
            # Create histogram of LBP values
            hist, _ = np.histogram(lbp.flatten(), bins=256, range=(0, 256))
            return hist.astype(float).tolist()
            
        except Exception as e:
            logging.warning(f"Error extracting LBP features: {str(e)}")
            return [0.0] * 256
    
    def _extract_histogram_features(self, face: np.ndarray) -> List[float]:
        """Extract intensity histogram features"""
        try:
            # Global histogram
            hist = cv2.calcHist([face], [0], None, [64], [0, 256])
            
            # Regional histograms (divide face into 4 quadrants)
            h, w = face.shape
            regions = [
                face[0:h//2, 0:w//2],      # Top-left
                face[0:h//2, w//2:w],      # Top-right
                face[h//2:h, 0:w//2],      # Bottom-left
                face[h//2:h, w//2:w]       # Bottom-right
            ]
            
            regional_hists = []
            for region in regions:
                if region.size > 0:
                    reg_hist = cv2.calcHist([region], [0], None, [16], [0, 256])
                    regional_hists.extend(reg_hist.flatten())
            
            # Combine global and regional features
            features = hist.flatten().tolist() + regional_hists
            return features
            
        except Exception as e:
            logging.warning(f"Error extracting histogram features: {str(e)}")
            return [0.0] * 128
    
    def _extract_edge_features(self, face: np.ndarray) -> List[float]:
        """Extract edge-based features"""
        try:
            # Apply Canny edge detection
            edges = cv2.Canny(face, 50, 150)
            
            # Count edges in different regions
            h, w = edges.shape
            regions = [
                edges[0:h//3, 0:w//3],          # Top-left
                edges[0:h//3, w//3:2*w//3],     # Top-center
                edges[0:h//3, 2*w//3:w],        # Top-right
                edges[h//3:2*h//3, 0:w//3],     # Middle-left
                edges[h//3:2*h//3, w//3:2*w//3], # Center
                edges[h//3:2*h//3, 2*w//3:w],   # Middle-right
                edges[2*h//3:h, 0:w//3],        # Bottom-left
                edges[2*h//3:h, w//3:2*w//3],   # Bottom-center
                edges[2*h//3:h, 2*w//3:w]       # Bottom-right
            ]
            
            edge_features = []
            for region in regions:
                if region.size > 0:
                    edge_density = np.sum(region > 0) / region.size
                    edge_features.append(edge_density)
                else:
                    edge_features.append(0.0)
            
            return edge_features
            
        except Exception as e:
            logging.warning(f"Error extracting edge features: {str(e)}")
            return [0.0] * 9
    
    def _extract_geometric_features(self, face: np.ndarray) -> List[float]:
        """Extract basic geometric features"""
        try:
            h, w = face.shape
            
            # Basic geometric measurements
            aspect_ratio = w / h if h > 0 else 1.0
            
            # Intensity-based features
            mean_intensity = np.mean(face)
            std_intensity = np.std(face)
            
            # Symmetry measure (simple left-right comparison)
            left_half = face[:, :w//2]
            right_half = np.fliplr(face[:, w//2:])
            min_width = min(left_half.shape[1], right_half.shape[1])
            
            if min_width > 0:
                left_resized = left_half[:, :min_width]
                right_resized = right_half[:, :min_width]
                symmetry = np.corrcoef(left_resized.flatten(), right_resized.flatten())[0, 1]
                if np.isnan(symmetry):
                    symmetry = 0.0
            else:
                symmetry = 0.0
            
            return [aspect_ratio, mean_intensity, std_intensity, symmetry]
            
        except Exception as e:
            logging.warning(f"Error extracting geometric features: {str(e)}")
            return [1.0, 128.0, 64.0, 0.0]
