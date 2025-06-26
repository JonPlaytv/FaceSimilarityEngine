# Automated Face Search System

## Overview

This is an advanced automated face search system that crawls real face images from public websites, generates high-quality embeddings using InsightFace, and provides lightning-fast similarity search using FAISS vector database. The system stores comprehensive metadata in DuckDB and automatically builds searchable face datasets from public image sources.

## System Architecture

### Backend Architecture
- **Web Framework**: Flask with Gunicorn for production deployment
- **AI Face Processing**: InsightFace with ONNX models for state-of-the-art face detection and embedding generation
- **Vector Search**: FAISS (Facebook AI Similarity Search) for high-performance similarity search
- **Database**: DuckDB for fast analytical queries and metadata storage
- **Web Crawling**: Automated crawler using Selenium and BeautifulSoup for image collection
- **Image Processing**: PIL (Pillow) for image manipulation and thumbnail generation

### Frontend Architecture
- **UI Framework**: Bootstrap with dark theme
- **Interactive Features**: Drag-and-drop file upload, automated crawling controls
- **Icons**: Font Awesome for visual elements
- **Responsive Design**: Mobile-friendly interface with advanced statistics display

### Data Storage
- **Vector Database**: FAISS index for 512-dimensional face embeddings
- **Metadata Database**: DuckDB for images, faces, search queries, and crawling sessions
- **File Storage**: Organized filesystem for crawled images, uploads, and thumbnails
- **Model Storage**: InsightFace ONNX models for face detection and recognition

## Key Components

### 1. Face Detection (`face_detector.py`)
- Uses OpenCV's Haar Cascade classifiers for robust face detection
- Implements LBPHFaceRecognizer for feature extraction
- Configurable detection parameters (scale factor, minimum neighbors, minimum size)

### 2. Similarity Engine (`similarity_engine.py`)
- Manages face embeddings database
- Implements similarity search algorithms
- Handles database persistence and loading
- Combines multiple similarity metrics for accurate matching

### 3. Image Processor (`image_processor.py`)
- Creates thumbnails for efficient display
- Handles image preprocessing and format conversion
- Maintains aspect ratios and applies consistent sizing

### 4. Main Application (`app.py`)
- Flask route handlers for upload and search functionality
- File validation and security measures
- Error handling and user feedback
- Integration of all components

## Data Flow

1. **Image Upload**: User uploads image via drag-and-drop or file selection
2. **Face Detection**: OpenCV detects faces in the uploaded image
3. **Feature Extraction**: System generates multi-modal feature vectors
4. **Similarity Search**: Compares features against database using combined metrics
5. **Results Display**: Returns ranked similar faces with confidence scores
6. **Thumbnail Generation**: Creates optimized thumbnails for display

## External Dependencies

### Core Libraries
- **Flask**: Web framework and HTTP handling
- **OpenCV**: Computer vision and face detection
- **NumPy**: Numerical computations for feature vectors
- **Pillow**: Image processing and thumbnail generation
- **Gunicorn**: Production WSGI server

### Development Dependencies
- **Werkzeug**: WSGI utilities and development server
- **psycopg2-binary**: PostgreSQL adapter (for potential future database integration)

## Deployment Strategy

### Production Configuration
- **Server**: Gunicorn with autoscaling deployment target
- **Port Binding**: 0.0.0.0:5000 with reuse-port option
- **Process Management**: Parallel workflow execution
- **File Limits**: 16MB maximum upload size
- **Security**: ProxyFix middleware for reverse proxy compatibility

### Development Setup
- **Environment**: Python 3.11 with Nix package management
- **Hot Reload**: Automatic server restart on file changes
- **Debug Mode**: Comprehensive logging and error reporting

### File Structure
```
/
├── static/
│   ├── dataset/         # Sample face images
│   ├── uploads/         # Temporary uploaded files
│   ├── thumbnails/      # Generated thumbnails
│   └── js/             # Frontend JavaScript
├── templates/          # HTML templates
├── data/              # Face embeddings database
└── *.py              # Python modules
```

## Recent Changes

- **June 26, 2025**: Complete system transformation to automated face search
  - Integrated InsightFace AI models for state-of-the-art face detection and embedding generation
  - Implemented FAISS vector database for high-performance similarity search
  - Added DuckDB for comprehensive metadata storage and analytics
  - Built automated web crawler targeting Unsplash, Pexels, and Pixabay
  - Created advanced UI with crawling controls and detailed statistics
  - Fixed database concurrency issues for Windows compatibility
  - System successfully tested and deployed on both Linux and Windows platforms

## Changelog

- June 26, 2025. Initial setup and complete rebuild with AI components

## User Preferences

Preferred communication style: Simple, everyday language.