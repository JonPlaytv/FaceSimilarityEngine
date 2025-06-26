import os
import logging
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import cv2
import json
from face_detector import FaceDetector
from similarity_engine import SimilarityEngine
from image_processor import ImageProcessor

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
DATASET_FOLDER = 'static/dataset'
THUMBNAILS_FOLDER = 'static/thumbnails'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Ensure directories exist
for folder in [UPLOAD_FOLDER, DATASET_FOLDER, THUMBNAILS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Initialize components
face_detector = FaceDetector()
similarity_engine = SimilarityEngine()
image_processor = ImageProcessor()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page with upload form"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and face similarity search"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(url_for('index'))
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('index'))
        
        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload an image file (PNG, JPG, JPEG, GIF, BMP)', 'error')
            return redirect(url_for('index'))
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the uploaded image
        try:
            # Load and validate image
            image = cv2.imread(filepath)
            if image is None:
                flash('Unable to process the uploaded image. Please try a different file.', 'error')
                os.remove(filepath)
                return redirect(url_for('index'))
            
            # Detect faces in uploaded image
            faces = face_detector.detect_faces(image)
            
            if not faces:
                flash('No faces detected in the uploaded image. Please try a different photo.', 'error')
                os.remove(filepath)
                return redirect(url_for('index'))
            
            # Generate embedding for the first detected face
            query_embedding = face_detector.generate_embedding(image, faces[0])
            
            if query_embedding is None:
                flash('Unable to generate face features from the uploaded image.', 'error')
                os.remove(filepath)
                return redirect(url_for('index'))
            
            # Find similar faces
            similar_faces = similarity_engine.find_similar_faces(query_embedding)
            
            # Generate thumbnail for uploaded image
            thumbnail_path = image_processor.create_thumbnail(filepath, filename)
            
            return render_template('results.html', 
                                 query_image=thumbnail_path,
                                 similar_faces=similar_faces,
                                 total_faces=len(similar_faces))
            
        except Exception as e:
            logging.error(f"Error processing image: {str(e)}")
            flash('An error occurred while processing your image. Please try again.', 'error')
            if os.path.exists(filepath):
                os.remove(filepath)
            return redirect(url_for('index'))
    
    except Exception as e:
        logging.error(f"Upload error: {str(e)}")
        flash('An unexpected error occurred. Please try again.', 'error')
        return redirect(url_for('index'))

@app.route('/initialize_dataset', methods=['POST'])
def initialize_dataset():
    """Initialize the face dataset from images in the dataset folder"""
    try:
        dataset_images = []
        dataset_path = DATASET_FOLDER
        
        # Get all image files from dataset folder
        for filename in os.listdir(dataset_path):
            if allowed_file(filename):
                filepath = os.path.join(dataset_path, filename)
                dataset_images.append(filepath)
        
        if not dataset_images:
            return jsonify({
                'success': False, 
                'message': 'No images found in dataset folder. Please add some sample face images to static/dataset/'
            })
        
        # Process dataset images
        processed_count = similarity_engine.build_dataset(dataset_images)
        
        return jsonify({
            'success': True,
            'message': f'Successfully processed {processed_count} images and built face dataset.'
        })
        
    except Exception as e:
        logging.error(f"Dataset initialization error: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error initializing dataset: {str(e)}'
        })

@app.route('/dataset_status')
def dataset_status():
    """Check if dataset is initialized"""
    dataset_size = similarity_engine.get_dataset_size()
    return jsonify({
        'initialized': dataset_size > 0,
        'size': dataset_size
    })

@app.errorhandler(413)
def too_large(e):
    flash('File is too large. Please upload an image smaller than 16MB.', 'error')
    return redirect(url_for('index'))

@app.errorhandler(404)
def not_found(e):
    return render_template('index.html'), 404

@app.errorhandler(500)
def server_error(e):
    logging.error(f"Server error: {str(e)}")
    flash('An internal server error occurred. Please try again.', 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
