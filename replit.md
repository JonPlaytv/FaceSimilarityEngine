# Face Similarity Search System

## Overview

This is an ethical face similarity search system built with Flask and OpenCV that allows users to upload photos and find visually similar faces from a sample dataset. The application focuses on privacy-first design, using only user-provided sample datasets without web crawling or unauthorized data collection.

## System Architecture

### Backend Architecture
- **Web Framework**: Flask with Gunicorn for production deployment
- **Computer Vision**: OpenCV for face detection and feature extraction
- **Image Processing**: PIL (Pillow) for image manipulation and thumbnail generation
- **Feature Extraction**: Multi-modal approach combining:
  - Local Binary Patterns (LBP)
  - Intensity Histograms
  - Edge Features
  - Geometric Properties
- **Similarity Matching**: Combination of cosine similarity and Euclidean distance

### Frontend Architecture
- **UI Framework**: Bootstrap with dark theme
- **Interactive Features**: Drag-and-drop file upload
- **Icons**: Font Awesome for visual elements
- **Responsive Design**: Mobile-friendly interface

### Data Storage
- **File Storage**: Local filesystem for uploaded images and thumbnails
- **Embeddings Database**: JSON file (`data/face_embeddings.json`) for face feature vectors
- **Temporary Storage**: Upload folder for processing images
- **Static Assets**: Organized directory structure for dataset images and thumbnails

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

## Changelog

- June 26, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.