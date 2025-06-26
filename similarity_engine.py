import numpy as np
import json
import os
import cv2
import logging
from typing import List, Dict, Tuple, Optional
from face_detector import FaceDetector
from image_processor import ImageProcessor

class SimilarityEngine:
    """Face similarity search engine using basic feature matching"""
    
    def __init__(self):
        """Initialize similarity engine"""
        self.face_detector = FaceDetector()
        self.image_processor = ImageProcessor()
        self.embeddings_file = 'data/face_embeddings.json'
        self.face_database = []
        self.load_database()
        
        logging.info("SimilarityEngine initialized")
    
    def load_database(self):
        """Load face embeddings database from file"""
        try:
            if os.path.exists(self.embeddings_file):
                with open(self.embeddings_file, 'r') as f:
                    data = json.load(f)
                    
                # Convert embeddings back to numpy arrays
                for entry in data:
                    entry['embedding'] = np.array(entry['embedding'], dtype=np.float32)
                
                self.face_database = data
                logging.info(f"Loaded {len(self.face_database)} face embeddings from database")
            else:
                self.face_database = []
                logging.info("No existing database found, starting with empty database")
                
        except Exception as e:
            logging.error(f"Error loading database: {str(e)}")
            self.face_database = []
    
    def save_database(self):
        """Save face embeddings database to file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.embeddings_file), exist_ok=True)
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_data = []
            for entry in self.face_database:
                serializable_entry = entry.copy()
                serializable_entry['embedding'] = entry['embedding'].tolist()
                serializable_data.append(serializable_entry)
            
            with open(self.embeddings_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)
                
            logging.info(f"Saved {len(self.face_database)} face embeddings to database")
            
        except Exception as e:
            logging.error(f"Error saving database: {str(e)}")
    
    def build_dataset(self, image_paths: List[str]) -> int:
        """
        Build face dataset from list of image paths
        
        Args:
            image_paths: List of paths to images
            
        Returns:
            Number of faces successfully processed
        """
        processed_count = 0
        self.face_database = []  # Reset database
        
        for image_path in image_paths:
            try:
                logging.info(f"Processing image: {image_path}")
                
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    logging.warning(f"Could not load image: {image_path}")
                    continue
                
                # Detect faces
                faces = self.face_detector.detect_faces(image)
                
                if not faces:
                    logging.warning(f"No faces detected in: {image_path}")
                    continue
                
                # Process each face (usually just one per image)
                for i, face_box in enumerate(faces):
                    # Generate embedding
                    embedding = self.face_detector.generate_embedding(image, face_box)
                    
                    if embedding is None:
                        logging.warning(f"Could not generate embedding for face {i} in {image_path}")
                        continue
                    
                    # Create thumbnail
                    filename = os.path.basename(image_path)
                    thumbnail_name = f"{os.path.splitext(filename)[0]}_face_{i}.jpg"
                    thumbnail_path = self.image_processor.create_face_thumbnail(
                        image, face_box, thumbnail_name
                    )
                    
                    # Add to database
                    face_entry = {
                        'id': len(self.face_database),
                        'source_image': image_path,
                        'face_index': i,
                        'thumbnail': thumbnail_path,
                        'embedding': embedding,
                        'face_box': face_box
                    }
                    
                    self.face_database.append(face_entry)
                    processed_count += 1
                    
                    logging.debug(f"Added face {i} from {filename} to database")
                
            except Exception as e:
                logging.error(f"Error processing {image_path}: {str(e)}")
                continue
        
        # Save the database
        self.save_database()
        
        logging.info(f"Dataset built with {processed_count} faces from {len(image_paths)} images")
        return processed_count
    
    def find_similar_faces(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict]:
        """
        Find similar faces to the query embedding
        
        Args:
            query_embedding: Query face embedding
            top_k: Number of similar faces to return
            
        Returns:
            List of similar faces with similarity scores
        """
        if len(self.face_database) == 0:
            logging.warning("Face database is empty")
            return []
        
        similarities = []
        
        for face_entry in self.face_database:
            try:
                # Calculate cosine similarity
                db_embedding = face_entry['embedding']
                
                # Cosine similarity
                dot_product = np.dot(query_embedding, db_embedding)
                norm_query = np.linalg.norm(query_embedding)
                norm_db = np.linalg.norm(db_embedding)
                
                if norm_query > 0 and norm_db > 0:
                    cosine_sim = dot_product / (norm_query * norm_db)
                else:
                    cosine_sim = 0.0
                
                # Euclidean distance (converted to similarity)
                euclidean_dist = np.linalg.norm(query_embedding - db_embedding)
                euclidean_sim = 1.0 / (1.0 + euclidean_dist)
                
                # Combined similarity score
                combined_similarity = 0.7 * cosine_sim + 0.3 * euclidean_sim
                
                similarities.append({
                    'face_entry': face_entry,
                    'similarity': combined_similarity,
                    'cosine_similarity': cosine_sim,
                    'euclidean_similarity': euclidean_sim
                })
                
            except Exception as e:
                logging.error(f"Error calculating similarity for face {face_entry.get('id', 'unknown')}: {str(e)}")
                continue
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return top-k results
        results = []
        for sim_data in similarities[:top_k]:
            face_entry = sim_data['face_entry']
            
            result = {
                'id': face_entry['id'],
                'thumbnail': face_entry['thumbnail'],
                'source_image': os.path.basename(face_entry['source_image']),
                'similarity_score': round(sim_data['similarity'] * 100, 1),
                'cosine_similarity': round(sim_data['cosine_similarity'] * 100, 1),
                'euclidean_similarity': round(sim_data['euclidean_similarity'] * 100, 1)
            }
            results.append(result)
        
        logging.info(f"Found {len(results)} similar faces")
        return results
    
    def get_dataset_size(self) -> int:
        """Get the number of faces in the dataset"""
        return len(self.face_database)
    
    def get_dataset_info(self) -> Dict:
        """Get information about the current dataset"""
        if not self.face_database:
            return {
                'size': 0,
                'source_images': 0,
                'avg_faces_per_image': 0
            }
        
        source_images = set()
        for entry in self.face_database:
            source_images.add(entry['source_image'])
        
        return {
            'size': len(self.face_database),
            'source_images': len(source_images),
            'avg_faces_per_image': round(len(self.face_database) / len(source_images), 1)
        }
