import numpy as np
import cv2
import logging
from typing import List, Tuple, Optional, Dict
import insightface
from insightface.app import FaceAnalysis
import os
import time

class InsightFaceEngine:
    """Advanced face detection and embedding generation using InsightFace"""
    
    def __init__(self):
        """Initialize InsightFace engine with ONNX models"""
        self.app = None
        self.model_loaded = False
        self.face_cascade = None
        
        # Detection parameters - set before initialization
        self.detection_size = (640, 640)
        self.confidence_threshold = 0.1
        self.embedding_dim = 512
        
        self.initialize_models()
        
        logging.info("InsightFaceEngine initialized")
    
    def initialize_models(self):
        """Initialize InsightFace models with Windows compatibility"""
        try:
            import os
            # Set ONNX runtime optimization for Windows compatibility
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['ONNXRUNTIME_EXECUTION_PROVIDERS'] = 'CPUExecutionProvider'
            
            # Create InsightFace app with minimal configuration
            self.app = FaceAnalysis(
                providers=['CPUExecutionProvider'],
                allowed_modules=['detection', 'recognition']
            )
            
            # Prepare model with smaller detection size for Windows stability
            self.app.prepare(ctx_id=0, det_size=(320, 320))
            self.detection_size = (320, 320)
            self.model_loaded = True
            
            logging.info("InsightFace models loaded successfully with Windows optimizations")
            
        except Exception as e:
            logging.error(f"InsightFace initialization failed: {e}")
            logging.info("Using OpenCV fallback for face detection")
            self.model_loaded = False
            
            # Initialize OpenCV fallback
            try:
                import cv2
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                if self.face_cascade.empty():
                    raise Exception("Could not load Haar cascade classifier")
                logging.info("OpenCV fallback detector initialized successfully")
            except Exception as fallback_error:
                logging.error(f"OpenCV fallback initialization failed: {fallback_error}")
                self.face_cascade = None
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in an image using InsightFace
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of face detection results with bounding boxes and landmarks
        """
        if not self.model_loaded:
            return []
        
        try:
            # Ensure image is in correct format
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Convert BGR to RGB for InsightFace
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            # Detect faces
            faces = self.app.get(rgb_image)
            
            face_results = []
            for i, face in enumerate(faces):
                # Extract bounding box
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                
                # Face confidence
                confidence = float(face.det_score)
                
                if confidence >= self.confidence_threshold:
                    face_info = {
                        'bbox': (x1, y1, x2 - x1, y2 - y1),  # (x, y, w, h) format
                        'bbox_xyxy': (x1, y1, x2, y2),  # (x1, y1, x2, y2) format
                        'confidence': confidence,
                        'landmarks': face.kps.astype(int) if hasattr(face, 'kps') else None,
                        'embedding': face.normed_embedding if hasattr(face, 'normed_embedding') else None,
                        'age': int(face.age) if hasattr(face, 'age') and face.age is not None else None,
                        'gender': face.sex if hasattr(face, 'sex') else None,
                        'face_id': i
                    }
                    face_results.append(face_info)
            
            logging.debug(f"Detected {len(face_results)} faces with confidence >= {self.confidence_threshold}")
            return face_results
            
        except Exception as e:
            logging.error(f"Error in face detection: {e}")
            return []
    
    def generate_embedding(self, image: np.ndarray, face_info: Dict = None) -> Optional[np.ndarray]:
        """
        Generate face embedding using InsightFace
        
        Args:
            image: Input image as numpy array
            face_info: Face detection information (optional)
            
        Returns:
            Face embedding as numpy array or None if failed
        """
        if not self.model_loaded:
            return None
        
        try:
            # If face_info is provided and has embedding, return it
            if face_info and face_info.get('embedding') is not None:
                return face_info['embedding']
            
            # Otherwise, detect faces and get embedding from the first face
            faces = self.detect_faces(image)
            if faces:
                return faces[0].get('embedding')
            else:
                return None
                
        except Exception as e:
            logging.error(f"Error generating embedding: {e}")
            return None
    
    def extract_face_embeddings(self, image: np.ndarray) -> List[Dict]:
        """
        Extract all face embeddings from an image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of dictionaries containing face info and embeddings
        """
        faces = self.detect_faces(image)
        
        embeddings_data = []
        for face in faces:
            if face.get('embedding') is not None:
                embeddings_data.append({
                    'embedding': face['embedding'],
                    'bbox': face['bbox'],
                    'confidence': face['confidence'],
                    'landmarks': face.get('landmarks'),
                    'age': face.get('age'),
                    'gender': face.get('gender'),
                    'face_id': face['face_id']
                })
        
        return embeddings_data
    
    def compare_faces(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compare two face embeddings using cosine similarity
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Normalize embeddings
            embedding1_norm = embedding1 / np.linalg.norm(embedding1)
            embedding2_norm = embedding2 / np.linalg.norm(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1_norm, embedding2_norm)
            
            # Convert to 0-1 scale (cosine similarity ranges from -1 to 1)
            similarity = (similarity + 1) / 2
            
            return float(similarity)
            
        except Exception as e:
            logging.error(f"Error comparing faces: {e}")
            return 0.0
    
    def is_same_person(self, embedding1: np.ndarray, embedding2: np.ndarray, threshold: float = 0.6) -> bool:
        """
        Determine if two embeddings represent the same person
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            threshold: Similarity threshold for same person
            
        Returns:
            True if same person, False otherwise
        """
        similarity = self.compare_faces(embedding1, embedding2)
        return similarity >= threshold
    
    def align_face(self, image: np.ndarray, landmarks: np.ndarray, output_size: Tuple[int, int] = (112, 112)) -> Optional[np.ndarray]:
        """
        Align face using facial landmarks
        
        Args:
            image: Input image
            landmarks: Facial landmarks (5 points)
            output_size: Output aligned face size
            
        Returns:
            Aligned face image or None if failed
        """
        try:
            from skimage import transform as trans
            
            # Reference points for alignment
            reference_points = np.array([
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041]
            ], dtype=np.float32)
            
            if output_size != (112, 112):
                # Scale reference points
                reference_points *= (output_size[0] / 112.0)
            
            # Estimate transformation matrix
            tform = trans.SimilarityTransform()
            tform.estimate(landmarks, reference_points)
            
            # Apply transformation
            aligned_face = trans.warp(image, tform.inverse, output_shape=output_size)
            aligned_face = (aligned_face * 255).astype(np.uint8)
            
            return aligned_face
            
        except Exception as e:
            logging.warning(f"Face alignment failed: {e}")
            return None
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        return {
            'model_loaded': self.model_loaded,
            'detection_size': self.detection_size,
            'confidence_threshold': self.confidence_threshold,
            'providers': ['CPUExecutionProvider'] if self.model_loaded else [],
            'embedding_size': 512 if self.model_loaded else 0
        }
    
    def preprocess_image(self, image: np.ndarray, max_size: int = 1280) -> np.ndarray:
        """
        Preprocess image for better face detection
        
        Args:
            image: Input image
            max_size: Maximum dimension size
            
        Returns:
            Preprocessed image
        """
        try:
            # Resize if too large
            height, width = image.shape[:2]
            if max(height, width) > max_size:
                if height > width:
                    new_height = max_size
                    new_width = int(width * (max_size / height))
                else:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            return image
            
        except Exception as e:
            logging.error(f"Error preprocessing image: {e}")
            return image
    
    def batch_process_images(self, image_paths: List[str]) -> List[Dict]:
        """
        Process multiple images in batch for efficiency
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of processing results
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    logging.warning(f"Could not load image: {image_path}")
                    continue
                
                # Preprocess
                image = self.preprocess_image(image)
                
                # Extract face embeddings
                face_data = self.extract_face_embeddings(image)
                
                result = {
                    'image_path': image_path,
                    'faces': face_data,
                    'face_count': len(face_data),
                    'processing_time': time.time()
                }
                
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    logging.info(f"Processed {i + 1}/{len(image_paths)} images")
                
            except Exception as e:
                logging.error(f"Error processing {image_path}: {e}")
                continue
        
        return results