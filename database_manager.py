import duckdb
import logging
import os
import json
from typing import List, Dict, Optional, Tuple
import time
import numpy as np

class DatabaseManager:
    """DuckDB database manager for face search metadata and analytics"""
    
    def __init__(self, db_path: str = 'data/face_search.db'):
        """
        Initialize database manager
        
        Args:
            db_path: Path to DuckDB database file
        """
        self.db_path = db_path
        self.conn = None
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self.initialize_database()
        
        logging.info(f"Database manager initialized with {db_path}")
    
    def initialize_database(self):
        """Initialize DuckDB connection and create tables"""
        try:
            # Create data directory if it doesn't exist
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Connect to DuckDB database with retry logic for file locks
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.conn = duckdb.connect(self.db_path)
                    break
                except Exception as e:
                    if attempt < max_retries - 1 and ("another process" in str(e).lower() or "verwendet wird" in str(e)):
                        logging.warning(f"Database locked, retrying in 2 seconds... (attempt {attempt + 1})")
                        import time
                        time.sleep(2)
                        continue
                    else:
                        # If still locked after retries, use in-memory database
                        logging.warning(f"Cannot access database file, using in-memory database: {e}")
                        self.conn = duckdb.connect(':memory:')
                        break
            
            self.create_tables()
            logging.info("Database connection established and tables created")
            
        except Exception as e:
            logging.error(f"Error initializing database: {e}")
            raise
    
    def create_tables(self):
        """Create necessary database tables"""
        try:
            # Images table - stores information about crawled/uploaded images
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY,
                    file_path TEXT NOT NULL UNIQUE,
                    filename TEXT NOT NULL,
                    source TEXT NOT NULL,
                    url TEXT,
                    file_size INTEGER,
                    width INTEGER,
                    height INTEGER,
                    format TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed_at TIMESTAMP,
                    face_count INTEGER DEFAULT 0,
                    processing_status TEXT DEFAULT 'pending'
                )
            ''')
            
            # Faces table - stores face detection and embedding data
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS faces (
                    id INTEGER PRIMARY KEY,
                    image_id INTEGER,
                    face_index INTEGER,
                    bbox_x INTEGER,
                    bbox_y INTEGER,
                    bbox_width INTEGER,
                    bbox_height INTEGER,
                    confidence REAL,
                    embedding_vector BLOB,
                    age INTEGER,
                    gender TEXT,
                    landmarks TEXT,
                    thumbnail_path TEXT,
                    faiss_index INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (image_id) REFERENCES images (id)
                )
            ''')
            
            # Search queries table - stores user search history
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS search_queries (
                    id INTEGER PRIMARY KEY,
                    query_image_path TEXT,
                    search_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    results_count INTEGER,
                    processing_time_ms INTEGER,
                    top_similarity REAL,
                    search_parameters TEXT
                )
            ''')
            
            # Crawling sessions table - tracks web crawling activities
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS crawl_sessions (
                    id INTEGER PRIMARY KEY,
                    session_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_end TIMESTAMP,
                    target_websites TEXT,
                    images_crawled INTEGER DEFAULT 0,
                    faces_detected INTEGER DEFAULT 0,
                    errors_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'running'
                )
            ''')
            
            # Create indices for better performance
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_images_source ON images (source)')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_images_created_at ON images (created_at)')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_faces_image_id ON faces (image_id)')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_faces_confidence ON faces (confidence)')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_search_timestamp ON search_queries (search_timestamp)')
            
            logging.info("Database tables and indices created successfully")
            
        except Exception as e:
            logging.error(f"Error creating database tables: {e}")
            raise
    
    def add_image(self, file_path: str, filename: str, source: str, url: str = None, 
                  file_size: int = None, width: int = None, height: int = None, 
                  format: str = None) -> int:
        """
        Add a new image record to the database
        
        Args:
            file_path: Path to the image file
            filename: Original filename
            source: Source of the image (e.g., 'unsplash', 'upload')
            url: Original URL if crawled
            file_size: File size in bytes
            width: Image width in pixels
            height: Image height in pixels
            format: Image format (JPEG, PNG, etc.)
            
        Returns:
            Image ID of the inserted record
        """
        try:
            result = self.conn.execute('''
                INSERT INTO images (file_path, filename, source, url, file_size, width, height, format)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
            ''', [file_path, filename, source, url, file_size, width, height, format])
            
            image_id = result.fetchone()[0]
            logging.debug(f"Added image record with ID {image_id}")
            return image_id
            
        except Exception as e:
            logging.error(f"Error adding image record: {e}")
            return -1
    
    def add_face(self, image_id: int, face_data: Dict, embedding: np.ndarray = None, 
                 faiss_index: int = None) -> int:
        """
        Add a face record to the database
        
        Args:
            image_id: ID of the parent image
            face_data: Dictionary containing face detection data
            embedding: Face embedding vector
            faiss_index: Index in FAISS vector database
            
        Returns:
            Face ID of the inserted record
        """
        try:
            # Extract face data
            bbox = face_data.get('bbox', (0, 0, 0, 0))
            confidence = face_data.get('confidence', 0.0)
            age = face_data.get('age')
            gender = face_data.get('gender')
            landmarks = json.dumps(face_data.get('landmarks', []).tolist()) if face_data.get('landmarks') is not None else None
            thumbnail_path = face_data.get('thumbnail_path')
            face_index = face_data.get('face_id', 0)
            
            # Serialize embedding
            embedding_blob = embedding.tobytes() if embedding is not None else None
            
            result = self.conn.execute('''
                INSERT INTO faces (image_id, face_index, bbox_x, bbox_y, bbox_width, bbox_height,
                                 confidence, embedding_vector, age, gender, landmarks, thumbnail_path, faiss_index)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
            ''', [image_id, face_index, bbox[0], bbox[1], bbox[2], bbox[3], 
                  confidence, embedding_blob, age, gender, landmarks, thumbnail_path, faiss_index])
            
            face_id = result.fetchone()[0]
            logging.debug(f"Added face record with ID {face_id}")
            return face_id
            
        except Exception as e:
            logging.error(f"Error adding face record: {e}")
            return -1
    
    def update_image_processing_status(self, image_id: int, status: str, face_count: int = None):
        """Update image processing status"""
        try:
            if face_count is not None:
                self.conn.execute('''
                    UPDATE images 
                    SET processing_status = ?, face_count = ?, processed_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', [status, face_count, image_id])
            else:
                self.conn.execute('''
                    UPDATE images 
                    SET processing_status = ?, processed_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', [status, image_id])
            
            logging.debug(f"Updated processing status for image {image_id}")
            
        except Exception as e:
            logging.error(f"Error updating image status: {e}")
    
    def log_search_query(self, query_image_path: str, results_count: int, 
                        processing_time_ms: int, top_similarity: float = None, 
                        search_parameters: Dict = None) -> int:
        """
        Log a search query for analytics
        
        Args:
            query_image_path: Path to the query image
            results_count: Number of results returned
            processing_time_ms: Processing time in milliseconds
            top_similarity: Highest similarity score
            search_parameters: Search parameters used
            
        Returns:
            Query ID
        """
        try:
            params_json = json.dumps(search_parameters) if search_parameters else None
            
            result = self.conn.execute('''
                INSERT INTO search_queries (query_image_path, results_count, processing_time_ms, top_similarity, search_parameters)
                VALUES (?, ?, ?, ?, ?)
                RETURNING id
            ''', [query_image_path, results_count, processing_time_ms, top_similarity, params_json])
            
            query_id = result.fetchone()[0]
            return query_id
            
        except Exception as e:
            logging.error(f"Error logging search query: {e}")
            return -1
    
    def start_crawl_session(self, target_websites: List[str]) -> int:
        """Start a new crawling session"""
        try:
            websites_json = json.dumps(target_websites)
            
            result = self.conn.execute('''
                INSERT INTO crawl_sessions (target_websites)
                VALUES (?)
                RETURNING id
            ''', [websites_json])
            
            session_id = result.fetchone()[0]
            logging.info(f"Started crawl session {session_id}")
            return session_id
            
        except Exception as e:
            logging.error(f"Error starting crawl session: {e}")
            return -1
    
    def update_crawl_session(self, session_id: int, images_crawled: int = None, 
                            faces_detected: int = None, errors_count: int = None, 
                            status: str = None):
        """Update crawling session statistics"""
        try:
            updates = []
            params = []
            
            if images_crawled is not None:
                updates.append("images_crawled = ?")
                params.append(images_crawled)
            
            if faces_detected is not None:
                updates.append("faces_detected = ?")
                params.append(faces_detected)
            
            if errors_count is not None:
                updates.append("errors_count = ?")
                params.append(errors_count)
            
            if status is not None:
                updates.append("status = ?")
                params.append(status)
                
                if status in ['completed', 'failed']:
                    updates.append("session_end = CURRENT_TIMESTAMP")
            
            if updates:
                params.append(session_id)
                query = f"UPDATE crawl_sessions SET {', '.join(updates)} WHERE id = ?"
                self.conn.execute(query, params)
                
                logging.debug(f"Updated crawl session {session_id}")
            
        except Exception as e:
            logging.error(f"Error updating crawl session: {e}")
    
    def get_image_by_path(self, file_path: str) -> Optional[Dict]:
        """Get image record by file path"""
        try:
            result = self.conn.execute('SELECT * FROM images WHERE file_path = ?', [file_path])
            row = result.fetchone()
            
            if row:
                columns = [desc[0] for desc in result.description]
                return dict(zip(columns, row))
            
            return None
            
        except Exception as e:
            logging.error(f"Error getting image by path: {e}")
            return None
    
    def get_faces_by_image_id(self, image_id: int) -> List[Dict]:
        """Get all faces for a specific image"""
        try:
            result = self.conn.execute('SELECT * FROM faces WHERE image_id = ? ORDER BY face_index', [image_id])
            rows = result.fetchall()
            
            if rows:
                columns = [desc[0] for desc in result.description]
                return [dict(zip(columns, row)) for row in rows]
            
            return []
            
        except Exception as e:
            logging.error(f"Error getting faces for image {image_id}: {e}")
            return []
    
    def get_face_embedding(self, face_id: int) -> Optional[np.ndarray]:
        """Get face embedding by face ID"""
        try:
            result = self.conn.execute('SELECT embedding_vector FROM faces WHERE id = ?', [face_id])
            row = result.fetchone()
            
            if row and row[0]:
                # Deserialize embedding
                embedding = np.frombuffer(row[0], dtype=np.float32)
                return embedding
            
            return None
            
        except Exception as e:
            logging.error(f"Error getting face embedding: {e}")
            return None
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        try:
            stats = {}
            
            # Image statistics
            result = self.conn.execute('SELECT COUNT(*), COUNT(CASE WHEN processing_status = ? THEN 1 END) FROM images', ['completed'])
            total_images, processed_images = result.fetchone()
            stats['total_images'] = total_images
            stats['processed_images'] = processed_images
            
            # Face statistics
            result = self.conn.execute('SELECT COUNT(*), AVG(confidence) FROM faces')
            total_faces, avg_confidence = result.fetchone()
            stats['total_faces'] = total_faces
            stats['avg_confidence'] = float(avg_confidence) if avg_confidence else 0.0
            
            # Source breakdown
            result = self.conn.execute('SELECT source, COUNT(*) FROM images GROUP BY source')
            stats['images_by_source'] = dict(result.fetchall())
            
            # Recent activity
            result = self.conn.execute("SELECT COUNT(*) FROM search_queries WHERE search_timestamp > (CURRENT_TIMESTAMP - INTERVAL '24 hours')")
            stats['searches_last_24h'] = result.fetchone()[0]
            
            # Crawling sessions
            result = self.conn.execute('SELECT COUNT(*), COUNT(CASE WHEN status = ? THEN 1 END) FROM crawl_sessions', ['completed'])
            total_sessions, completed_sessions = result.fetchone()
            stats['total_crawl_sessions'] = total_sessions
            stats['completed_crawl_sessions'] = completed_sessions
            
            return stats
            
        except Exception as e:
            logging.error(f"Error getting database stats: {e}")
            return {}
    
    def cleanup_old_data(self, days: int = 30):
        """Clean up old data from the database"""
        try:
            cutoff_date = f"datetime('now', '-{days} days')"
            
            # Remove old search queries
            result = self.conn.execute(f'DELETE FROM search_queries WHERE search_timestamp < {cutoff_date}')
            deleted_queries = result.rowcount
            
            # Remove old crawl sessions
            result = self.conn.execute(f'DELETE FROM crawl_sessions WHERE session_start < {cutoff_date}')
            deleted_sessions = result.rowcount
            
            logging.info(f"Cleaned up {deleted_queries} old queries and {deleted_sessions} old sessions")
            
        except Exception as e:
            logging.error(f"Error cleaning up old data: {e}")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logging.info("Database connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()