"""
PostgreSQL database models for the automated face search system
"""
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, LargeBinary, Boolean, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import ARRAY
import numpy as np
import json

Base = declarative_base()

class Image(Base):
    """Model for storing image metadata"""
    __tablename__ = 'images'
    
    id = Column(Integer, primary_key=True)
    file_path = Column(String(500), unique=True, nullable=False)
    filename = Column(String(255), nullable=False)
    source = Column(String(100), nullable=False)  # 'upload', 'unsplash', 'pexels', etc.
    url = Column(Text)  # Original URL if crawled
    file_size = Column(Integer)  # File size in bytes
    width = Column(Integer)  # Image width in pixels
    height = Column(Integer)  # Image height in pixels
    format = Column(String(20))  # Image format (JPEG, PNG, etc.)
    processing_status = Column(String(50), default='pending')  # pending, processing, completed, failed
    face_count = Column(Integer, default=0)  # Number of faces detected
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)
    
    # Relationships
    faces = relationship("Face", back_populates="image", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_images_source', 'source'),
        Index('idx_images_status', 'processing_status'),
        Index('idx_images_created', 'created_at'),
    )

class Face(Base):
    """Model for storing face detection and embedding data"""
    __tablename__ = 'faces'
    
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('images.id'), nullable=False)
    bbox_x = Column(Float, nullable=False)  # Bounding box coordinates
    bbox_y = Column(Float, nullable=False)
    bbox_width = Column(Float, nullable=False)
    bbox_height = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)  # Detection confidence
    landmarks = Column(Text)  # JSON string of facial landmarks
    embedding = Column(LargeBinary)  # Face embedding as binary data
    faiss_index = Column(Integer)  # Index in FAISS vector database
    thumbnail_path = Column(String(500))  # Path to face thumbnail
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    image = relationship("Image", back_populates="faces")
    
    # Indexes
    __table_args__ = (
        Index('idx_faces_image_id', 'image_id'),
        Index('idx_faces_confidence', 'confidence'),
        Index('idx_faces_faiss_index', 'faiss_index'),
    )
    
    def set_embedding(self, embedding_array):
        """Store numpy array as binary data"""
        if embedding_array is not None:
            self.embedding = embedding_array.tobytes()
    
    def get_embedding(self):
        """Retrieve embedding as numpy array"""
        if self.embedding:
            return np.frombuffer(self.embedding, dtype=np.float32)
        return None
    
    def set_landmarks(self, landmarks_array):
        """Store landmarks as JSON string"""
        if landmarks_array is not None:
            self.landmarks = json.dumps(landmarks_array.tolist())
    
    def get_landmarks(self):
        """Retrieve landmarks as numpy array"""
        if self.landmarks:
            return np.array(json.loads(self.landmarks))
        return None

class SearchQuery(Base):
    """Model for logging search queries and analytics"""
    __tablename__ = 'search_queries'
    
    id = Column(Integer, primary_key=True)
    query_image_path = Column(String(500), nullable=False)
    results_count = Column(Integer, nullable=False)
    processing_time_ms = Column(Integer, nullable=False)
    top_similarity = Column(Float)  # Highest similarity score
    search_parameters = Column(Text)  # JSON string of search parameters
    user_ip = Column(String(45))  # IPv4/IPv6 address
    user_agent = Column(Text)
    search_timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_search_timestamp', 'search_timestamp'),
        Index('idx_search_similarity', 'top_similarity'),
    )

class CrawlSession(Base):
    """Model for tracking web crawling sessions"""
    __tablename__ = 'crawl_sessions'
    
    id = Column(Integer, primary_key=True)
    target_websites = Column(Text, nullable=False)  # JSON array of target websites
    session_start = Column(DateTime, default=datetime.utcnow)
    session_end = Column(DateTime)
    status = Column(String(50), default='running')  # running, completed, failed, cancelled
    images_crawled = Column(Integer, default=0)
    faces_detected = Column(Integer, default=0)
    errors_count = Column(Integer, default=0)
    error_log = Column(Text)  # JSON array of error messages
    
    # Indexes
    __table_args__ = (
        Index('idx_crawl_status', 'status'),
        Index('idx_crawl_start', 'session_start'),
    )

class DatabaseManager:
    """PostgreSQL database manager for the face search system"""
    
    def __init__(self, database_url=None):
        """Initialize database connection"""
        if database_url is None:
            database_url = os.environ.get('DATABASE_URL')
        
        if not database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        self.engine = create_engine(database_url, pool_size=10, max_overflow=20)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Create all tables
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get a database session"""
        return self.SessionLocal()
    
    def add_image(self, file_path, filename, source, url=None, file_size=None, 
                  width=None, height=None, format=None):
        """Add a new image record"""
        with self.get_session() as session:
            image = Image(
                file_path=file_path,
                filename=filename,
                source=source,
                url=url,
                file_size=file_size,
                width=width,
                height=height,
                format=format
            )
            session.add(image)
            session.commit()
            return image.id
    
    def add_face(self, image_id, bbox, confidence, landmarks=None, embedding=None, faiss_index=None):
        """Add a face record"""
        with self.get_session() as session:
            face = Face(
                image_id=image_id,
                bbox_x=bbox[0],
                bbox_y=bbox[1],
                bbox_width=bbox[2],
                bbox_height=bbox[3],
                confidence=confidence,
                faiss_index=faiss_index
            )
            
            if landmarks is not None:
                face.set_landmarks(landmarks)
            
            if embedding is not None:
                face.set_embedding(embedding)
            
            session.add(face)
            session.commit()
            return face.id
    
    def update_image_processing_status(self, image_id, status, face_count=None):
        """Update image processing status"""
        with self.get_session() as session:
            image = session.query(Image).filter(Image.id == image_id).first()
            if image:
                image.processing_status = status
                if face_count is not None:
                    image.face_count = face_count
                if status == 'completed':
                    image.processed_at = datetime.utcnow()
                session.commit()
    
    def log_search_query(self, query_image_path, results_count, processing_time_ms, 
                        top_similarity=None, search_parameters=None, user_ip=None, user_agent=None):
        """Log a search query"""
        with self.get_session() as session:
            query = SearchQuery(
                query_image_path=query_image_path,
                results_count=results_count,
                processing_time_ms=processing_time_ms,
                top_similarity=top_similarity,
                search_parameters=json.dumps(search_parameters) if search_parameters else None,
                user_ip=user_ip,
                user_agent=user_agent
            )
            session.add(query)
            session.commit()
            return query.id
    
    def start_crawl_session(self, target_websites):
        """Start a new crawling session"""
        with self.get_session() as session:
            crawl_session = CrawlSession(
                target_websites=json.dumps(target_websites)
            )
            session.add(crawl_session)
            session.commit()
            return crawl_session.id
    
    def update_crawl_session(self, session_id, images_crawled=None, faces_detected=None, 
                           errors_count=None, status=None):
        """Update crawling session statistics"""
        with self.get_session() as session:
            crawl_session = session.query(CrawlSession).filter(CrawlSession.id == session_id).first()
            if crawl_session:
                if images_crawled is not None:
                    crawl_session.images_crawled = images_crawled
                if faces_detected is not None:
                    crawl_session.faces_detected = faces_detected
                if errors_count is not None:
                    crawl_session.errors_count = errors_count
                if status is not None:
                    crawl_session.status = status
                    if status in ['completed', 'failed', 'cancelled']:
                        crawl_session.session_end = datetime.utcnow()
                session.commit()
    
    def get_database_stats(self):
        """Get comprehensive database statistics"""
        with self.get_session() as session:
            stats = {}
            
            # Image statistics
            total_images = session.query(Image).count()
            processed_images = session.query(Image).filter(Image.processing_status == 'completed').count()
            stats['total_images'] = total_images
            stats['processed_images'] = processed_images
            
            # Face statistics
            total_faces = session.query(Face).count()
            stats['total_faces'] = total_faces
            
            # Average confidence
            avg_confidence = session.query(Face.confidence).filter(Face.confidence.isnot(None)).all()
            if avg_confidence:
                stats['avg_confidence'] = sum(c[0] for c in avg_confidence) / len(avg_confidence)
            else:
                stats['avg_confidence'] = 0
            
            # Source breakdown
            sources = session.query(Image.source, session.query(Image).filter(Image.source == Image.source).count()).distinct().all()
            stats['images_by_source'] = {source: count for source, count in sources}
            
            # Recent activity (last 24 hours)
            from datetime import timedelta
            yesterday = datetime.utcnow() - timedelta(hours=24)
            recent_searches = session.query(SearchQuery).filter(SearchQuery.search_timestamp > yesterday).count()
            stats['searches_last_24h'] = recent_searches
            
            # Crawling sessions
            total_sessions = session.query(CrawlSession).count()
            completed_sessions = session.query(CrawlSession).filter(CrawlSession.status == 'completed').count()
            stats['total_crawl_sessions'] = total_sessions
            stats['completed_crawl_sessions'] = completed_sessions
            
            return stats
    
    def get_face_by_faiss_index(self, faiss_index):
        """Get face record by FAISS index"""
        with self.get_session() as session:
            face = session.query(Face).filter(Face.faiss_index == faiss_index).first()
            if face:
                return {
                    'id': face.id,
                    'image_id': face.image_id,
                    'bbox': [face.bbox_x, face.bbox_y, face.bbox_width, face.bbox_height],
                    'confidence': face.confidence,
                    'thumbnail_path': face.thumbnail_path,
                    'image_path': face.image.file_path if face.image else None,
                    'image_source': face.image.source if face.image else None
                }
            return None
    
    def cleanup_old_data(self, days=30):
        """Clean up old data"""
        with self.get_session() as session:
            from datetime import timedelta
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Delete old search queries
            session.query(SearchQuery).filter(SearchQuery.search_timestamp < cutoff_date).delete()
            
            # Delete old completed crawl sessions
            session.query(CrawlSession).filter(
                CrawlSession.session_end < cutoff_date,
                CrawlSession.status == 'completed'
            ).delete()
            
            session.commit()