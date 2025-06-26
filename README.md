# Face Similarity Search System

An ethical face similarity search system built with Flask and OpenCV that allows users to upload photos and find visually similar faces from a sample dataset.

## Features

- **Face Detection**: Uses OpenCV's Haar Cascade classifiers for robust face detection
- **Feature Extraction**: Generates embeddings using multiple computer vision techniques:
  - Local Binary Patterns (LBP)
  - Intensity Histograms
  - Edge Features
  - Geometric Properties
- **Similarity Search**: Combines cosine similarity and Euclidean distance for accurate matching
- **Web Interface**: Clean, responsive Bootstrap UI with drag-and-drop upload
- **Ethical Design**: No web crawling - uses only sample datasets provided by users

## Ethical Considerations

This system is designed with privacy and ethics in mind:

- ✅ **No Web Crawling**: Uses only sample datasets, no unauthorized data collection
- ✅ **Privacy-Focused**: Local processing, no permanent storage of uploaded images
- ✅ **Transparent**: Open-source algorithms, no black-box AI models
- ✅ **Educational**: Designed for research and learning purposes
- ✅ **User Control**: Users provide their own dataset images

## Installation & Setup

### Prerequisites

- Python 3.8+
- OpenCV with face recognition module
- Flask and other Python dependencies

### Quick Start

1. **Clone/Download the application files**

2. **Install dependencies**:
   ```bash
   pip install flask opencv-python opencv-contrib-python pillow numpy
   ```

3. **Add sample images**:
   - Place sample face images in the `static/dataset/` directory
   - Supported formats: PNG, JPG, JPEG, GIF, BMP
   - Use clear, well-lit photos with visible faces

4. **Run the application**:
   ```bash
   python main.py
   