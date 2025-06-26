import numpy as np
import faiss
import logging
import os
import pickle
from typing import List, Dict, Tuple, Optional
import time

class FAISSSearchEngine:
    """High-performance vector similarity search using FAISS"""
    
    def __init__(self, embedding_dim: int = 512):
        """
        Initialize FAISS search engine
        
        Args:
            embedding_dim: Dimension of face embeddings (default 512 for InsightFace)
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.embeddings = []
        self.metadata = []
        self.index_file = 'data/faiss_index.bin'
        self.metadata_file = 'data/faiss_metadata.pkl'
        
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        
        # Initialize FAISS index
        self.initialize_index()
        
        logging.info(f"FAISS search engine initialized with {embedding_dim}D embeddings")
    
    def initialize_index(self):
        """Initialize FAISS index for similarity search"""
        try:
            # Create FAISS index
            # Using IndexFlatIP for inner product (cosine similarity with normalized vectors)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            
            # Load existing index if available
            self.load_index()
            
            logging.info(f"FAISS index initialized with {self.index.ntotal} vectors")
            
        except Exception as e:
            logging.error(f"Error initializing FAISS index: {e}")
            # Fallback to basic index
            self.index = faiss.IndexFlatL2(self.embedding_dim)
    
    def add_embeddings(self, embeddings: List[np.ndarray], metadata: List[Dict]) -> bool:
        """
        Add face embeddings to the search index
        
        Args:
            embeddings: List of face embeddings
            metadata: List of metadata dictionaries for each embedding
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if len(embeddings) != len(metadata):
                logging.error("Number of embeddings and metadata entries must match")
                return False
            
            # Normalize embeddings for cosine similarity
            normalized_embeddings = []
            for emb in embeddings:
                if isinstance(emb, np.ndarray) and emb.size > 0:
                    # Normalize to unit vector
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        normalized_emb = emb / norm
                    else:
                        normalized_emb = emb
                    normalized_embeddings.append(normalized_emb.astype(np.float32))
                else:
                    logging.warning("Invalid embedding encountered, skipping")
                    continue
            
            if not normalized_embeddings:
                logging.warning("No valid embeddings to add")
                return False
            
            # Convert to numpy array
            embeddings_array = np.vstack(normalized_embeddings)
            
            # Add to FAISS index
            self.index.add(embeddings_array)
            
            # Store embeddings and metadata
            self.embeddings.extend(normalized_embeddings)
            self.metadata.extend(metadata)
            
            logging.info(f"Added {len(normalized_embeddings)} embeddings to index")
            return True
            
        except Exception as e:
            logging.error(f"Error adding embeddings to index: {e}")
            return False
    
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 10, threshold: float = 0.5) -> List[Dict]:
        """
        Search for similar face embeddings
        
        Args:
            query_embedding: Query face embedding
            top_k: Number of similar faces to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of similar faces with similarity scores and metadata
        """
        try:
            if self.index.ntotal == 0:
                logging.warning("Search index is empty")
                return []
            
            # Normalize query embedding
            query_norm = np.linalg.norm(query_embedding)
            if query_norm > 0:
                query_normalized = (query_embedding / query_norm).astype(np.float32)
            else:
                query_normalized = query_embedding.astype(np.float32)
            
            # Reshape for FAISS
            query_vector = query_normalized.reshape(1, -1)
            
            # Search
            start_time = time.time()
            scores, indices = self.index.search(query_vector, min(top_k * 2, self.index.ntotal))
            search_time = time.time() - start_time
            
            # Process results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # Invalid index
                    continue
                
                # Convert score to similarity (0-1 range)
                similarity = float(score)  # Inner product of normalized vectors = cosine similarity
                
                if similarity >= threshold:
                    result = {
                        'similarity': similarity,
                        'index': int(idx),
                        'metadata': self.metadata[idx].copy() if idx < len(self.metadata) else {},
                        'rank': i + 1
                    }
                    results.append(result)
            
            # Sort by similarity (highest first)
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
            logging.debug(f"Search completed in {search_time:.3f}s, found {len(results)} results above threshold")
            return results[:top_k]
            
        except Exception as e:
            logging.error(f"Error in similarity search: {e}")
            return []
    
    def batch_search(self, query_embeddings: List[np.ndarray], top_k: int = 10) -> List[List[Dict]]:
        """
        Perform batch similarity search for multiple queries
        
        Args:
            query_embeddings: List of query embeddings
            top_k: Number of results per query
            
        Returns:
            List of result lists, one for each query
        """
        try:
            if not query_embeddings:
                return []
            
            # Normalize query embeddings
            normalized_queries = []
            for emb in query_embeddings:
                norm = np.linalg.norm(emb)
                if norm > 0:
                    normalized_queries.append((emb / norm).astype(np.float32))
                else:
                    normalized_queries.append(emb.astype(np.float32))
            
            # Stack queries
            query_matrix = np.vstack(normalized_queries)
            
            # Batch search
            scores, indices = self.index.search(query_matrix, top_k)
            
            # Process results for each query
            all_results = []
            for query_idx in range(len(query_embeddings)):
                query_results = []
                for i, (score, idx) in enumerate(zip(scores[query_idx], indices[query_idx])):
                    if idx != -1 and idx < len(self.metadata):
                        result = {
                            'similarity': float(score),
                            'index': int(idx),
                            'metadata': self.metadata[idx].copy(),
                            'rank': i + 1
                        }
                        query_results.append(result)
                
                all_results.append(query_results)
            
            return all_results
            
        except Exception as e:
            logging.error(f"Error in batch search: {e}")
            return []
    
    def save_index(self) -> bool:
        """Save FAISS index and metadata to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, self.index_file)
            
            # Save metadata
            with open(self.metadata_file, 'wb') as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'embeddings': self.embeddings,
                    'embedding_dim': self.embedding_dim
                }, f)
            
            logging.info(f"Saved FAISS index with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            logging.error(f"Error saving FAISS index: {e}")
            return False
    
    def load_index(self) -> bool:
        """Load FAISS index and metadata from disk"""
        try:
            # Load FAISS index
            if os.path.exists(self.index_file):
                self.index = faiss.read_index(self.index_file)
                
                # Load metadata
                if os.path.exists(self.metadata_file):
                    with open(self.metadata_file, 'rb') as f:
                        data = pickle.load(f)
                        self.metadata = data.get('metadata', [])
                        self.embeddings = data.get('embeddings', [])
                        
                        # Verify embedding dimension
                        saved_dim = data.get('embedding_dim', self.embedding_dim)
                        if saved_dim != self.embedding_dim:
                            logging.warning(f"Embedding dimension mismatch: {saved_dim} vs {self.embedding_dim}")
                
                logging.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
                return True
            else:
                logging.info("No existing FAISS index found, starting fresh")
                return False
                
        except Exception as e:
            logging.error(f"Error loading FAISS index: {e}")
            return False
    
    def rebuild_index(self) -> bool:
        """Rebuild the FAISS index from stored embeddings"""
        try:
            if not self.embeddings:
                logging.warning("No embeddings to rebuild index from")
                return False
            
            # Create new index
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            
            # Add all embeddings
            embeddings_array = np.vstack(self.embeddings)
            self.index.add(embeddings_array)
            
            logging.info(f"Rebuilt FAISS index with {len(self.embeddings)} vectors")
            return True
            
        except Exception as e:
            logging.error(f"Error rebuilding index: {e}")
            return False
    
    def remove_embeddings(self, indices: List[int]) -> bool:
        """
        Remove embeddings from index (requires rebuild)
        
        Args:
            indices: List of indices to remove
            
        Returns:
            True if successful
        """
        try:
            # Sort indices in descending order to avoid index shifting issues
            indices = sorted(set(indices), reverse=True)
            
            # Remove from metadata and embeddings
            for idx in indices:
                if 0 <= idx < len(self.metadata):
                    del self.metadata[idx]
                    del self.embeddings[idx]
            
            # Rebuild index
            return self.rebuild_index()
            
        except Exception as e:
            logging.error(f"Error removing embeddings: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get search engine statistics"""
        return {
            'total_vectors': self.index.ntotal if self.index else 0,
            'embedding_dimension': self.embedding_dim,
            'index_type': type(self.index).__name__ if self.index else None,
            'metadata_count': len(self.metadata),
            'embeddings_count': len(self.embeddings),
            'index_file_exists': os.path.exists(self.index_file),
            'metadata_file_exists': os.path.exists(self.metadata_file)
        }
    
    def clear_index(self):
        """Clear all data from the index"""
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.embeddings = []
        self.metadata = []
        
        # Remove saved files
        for file_path in [self.index_file, self.metadata_file]:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        logging.info("FAISS index cleared")
    
    def optimize_index(self):
        """Optimize the index for better performance (if using advanced index types)"""
        try:
            # For basic IndexFlatIP, no optimization needed
            # This method is for future use with more advanced index types
            if hasattr(self.index, 'train') and not self.index.is_trained:
                if self.embeddings:
                    embeddings_array = np.vstack(self.embeddings)
                    self.index.train(embeddings_array)
                    logging.info("FAISS index trained/optimized")
            
        except Exception as e:
            logging.warning(f"Index optimization failed: {e}")