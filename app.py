import os
import logging
import time
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import cv2
import numpy as np
from insight_face_engine import InsightFaceEngine
from faiss_search_engine import FAISSSearchEngine
from database_manager import DatabaseManager
from web_crawler import WebCrawler
from image_processor import ImageProcessor
from models import DatabaseManager as PostgreSQLManager

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

# Initialize advanced components
face_engine = InsightFaceEngine()
search_engine = FAISSSearchEngine(embedding_dim=512)
db_manager = DatabaseManager()  # DuckDB for analytics

# Initialize PostgreSQL if available
postgres_db = None
try:
    postgres_db = PostgreSQLManager()  # PostgreSQL for structured data
    logging.info("PostgreSQL database connected successfully")
except Exception as e:
    logging.warning(f"PostgreSQL not available, using DuckDB only: {e}")

web_crawler = WebCrawler()
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
    """Handle file upload and face similarity search using InsightFace and FAISS"""
    start_time = time.time()
    
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(url_for('index'))
        
        file = request.files['file']
        
        if file.filename == '' or file.filename is None:
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
            
            # Preprocess image
            image = face_engine.preprocess_image(image)
            
            # Detect faces using InsightFace
            faces = face_engine.detect_faces(image)
            
            if not faces:
                flash('No faces detected in the uploaded image. Please try a different photo.', 'error')
                os.remove(filepath)
                return redirect(url_for('index'))
            
            # Get embedding for the best face (highest confidence)
            best_face = max(faces, key=lambda x: x.get('confidence', 0))
            query_embedding = best_face.get('embedding')
            
            if query_embedding is None:
                flash('Unable to generate face features from the uploaded image.', 'error')
                os.remove(filepath)
                return redirect(url_for('index'))
            
            # Search for similar faces using FAISS
            similar_results = search_engine.search_similar(query_embedding, top_k=20, threshold=0.3)
            
            # Process results for display
            similar_faces = []
            for result in similar_results:
                metadata = result.get('metadata', {})
                similar_faces.append({
                    'id': result['index'],
                    'thumbnail': metadata.get('thumbnail_path', ''),
                    'source_image': metadata.get('filename', ''),
                    'similarity_score': round(result['similarity'] * 100, 1),
                    'confidence': metadata.get('confidence', 0),
                    'age': metadata.get('age'),
                    'gender': metadata.get('gender'),
                    # Add detailed similarity metrics if available
                    'cosine_similarity': round(result.get('cosine', result['similarity']) * 100, 1) if 'cosine' in result or 'similarity' in result else None,
                    'euclidean_similarity': round(result.get('euclidean', 1 - result['similarity']) * 100, 1) if 'euclidean' in result or 'similarity' in result else None
                })
            
            # Generate thumbnail for uploaded image
            thumbnail_path = image_processor.create_thumbnail(filepath, filename)
            
            # Log search query
            processing_time = int((time.time() - start_time) * 1000)
            top_similarity = similar_faces[0]['similarity_score'] / 100 if similar_faces else 0
            db_manager.log_search_query(
                query_image_path=filepath,
                results_count=len(similar_faces),
                processing_time_ms=processing_time,
                top_similarity=top_similarity
            )
            
            return render_template('results.html', 
                                 query_image=thumbnail_path,
                                 similar_faces=similar_faces,
                                 total_faces=len(similar_faces),
                                 processing_time=processing_time,
                                 face_details=best_face)
            
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

@app.route('/crawl_images', methods=['POST'])
def crawl_images():
    """Start automated web crawling for face images"""
    try:
        max_images = request.json.get('max_images', 50) if request.is_json else 50
        
        # Start crawling session
        target_websites = [w['name'] for w in web_crawler.get_target_websites()]
        session_id = db_manager.start_crawl_session(target_websites)
        
        # Perform web crawling
        logging.info(f"Starting automated crawling for {max_images} images")
        crawl_stats = web_crawler.crawl_websites(max_images=max_images)
        
        # Get crawled images
        crawled_images = web_crawler.get_crawled_images()
        
        if not crawled_images:
            db_manager.update_crawl_session(session_id, status='failed', errors_count=1)
            return jsonify({
                'success': False,
                'message': 'No images were successfully crawled. Please check network connection.'
            })
        
        # Process crawled images with InsightFace
        processed_count = process_crawled_images(crawled_images, session_id)
        
        # Update crawl session
        db_manager.update_crawl_session(
            session_id, 
            images_crawled=len(crawled_images),
            faces_detected=processed_count,
            status='completed'
        )
        
        return jsonify({
            'success': True,
            'message': f'Successfully crawled {len(crawled_images)} images and detected {processed_count} faces.',
            'stats': crawl_stats
        })
        
    except Exception as e:
        logging.error(f"Crawling error: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error during crawling: {str(e)}'
        })

@app.route('/initialize_dataset', methods=['POST'])
def initialize_dataset():
    """Initialize the face dataset from local images and crawled data"""
    try:
        # Process local dataset images
        dataset_images = []
        dataset_path = DATASET_FOLDER
        
        logging.info(f"Scanning dataset folder: {dataset_path}")
        
        # Get all image files from dataset folder
        if os.path.exists(dataset_path):
            for filename in os.listdir(dataset_path):
                if allowed_file(filename):
                    filepath = os.path.join(dataset_path, filename)
                    dataset_images.append(filepath)
                    logging.info(f"Found dataset image: {filename}")
        
        # Process crawled images if available
        crawled_images = []
        try:
            crawled_images = web_crawler.get_crawled_images()
        except Exception as e:
            logging.warning(f"Could not get crawled images: {e}")
        
        all_images = dataset_images + crawled_images
        
        if not all_images:
            return jsonify({
                'success': False, 
                'message': 'No images found. Use "Crawl Images" to get data or add images to static/dataset/'
            })
        
        logging.info(f"Processing {len(all_images)} images total")
        
        # Process all images with InsightFace and store in FAISS/DuckDB
        processed_count = process_image_dataset(all_images)
        
        # Get updated statistics
        faiss_stats = search_engine.get_stats()
        db_stats = db_manager.get_database_stats()
        
        return jsonify({
            'success': True,
            'message': f'Successfully processed {processed_count} faces from {len(all_images)} images.',
            'dataset_images': len(dataset_images),
            'crawled_images': len(crawled_images),
            'total_faces': processed_count,
            'faiss_vectors': faiss_stats.get('total_vectors', 0)
        })
        
    except Exception as e:
        logging.error(f"Dataset initialization error: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error initializing dataset: {str(e)}'
        })

def process_crawled_images(image_paths, session_id):
    """Process crawled images with InsightFace and store in FAISS/DuckDB"""
    processed_count = 0
    embeddings = []
    metadata = []
    
    for image_path in image_paths:
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            # Get image info
            file_size = os.path.getsize(image_path)
            height, width = image.shape[:2]
            
            # Add to database
            image_id = db_manager.add_image(
                file_path=image_path,
                filename=os.path.basename(image_path),
                source='crawled',
                file_size=file_size,
                width=width,
                height=height
            )
            
            # Extract face data using InsightFace
            face_data_list = face_engine.extract_face_embeddings(image)
            
            for face_data in face_data_list:
                try:
                    # Create face thumbnail
                    bbox = face_data['bbox']
                    thumbnail_name = f"face_{image_id}_{face_data['face_id']}.jpg"
                    thumbnail_path = image_processor.create_face_thumbnail(image, bbox, thumbnail_name)
                    
                    # Prepare metadata for FAISS
                    face_metadata = {
                        'image_id': image_id,
                        'filename': os.path.basename(image_path),
                        'thumbnail_path': thumbnail_path,
                        'confidence': face_data['confidence'],
                        'age': face_data.get('age'),
                        'gender': face_data.get('gender'),
                        'source': 'crawled'
                    }
                    
                    # Add to database
                    face_id = db_manager.add_face(image_id, face_data, face_data['embedding'], len(embeddings))
                    
                    # Store for FAISS
                    embeddings.append(face_data['embedding'])
                    metadata.append(face_metadata)
                    processed_count += 1
                    
                except Exception as e:
                    logging.warning(f"Error processing face in {image_path}: {e}")
                    continue
            
            # Update image processing status
            db_manager.update_image_processing_status(image_id, 'completed', len(face_data_list))
            
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}")
            continue
    
    # Add embeddings to FAISS index
    if embeddings:
        search_engine.add_embeddings(embeddings, metadata)
        search_engine.save_index()
    
    return processed_count

def process_image_dataset(image_paths):
    """Process local and crawled images for the dataset"""
    processed_count = 0
    embeddings = []
    metadata = []
    
    for image_path in image_paths:
        try:
            # Check if already processed
            existing_record = db_manager.get_image_by_path(image_path)
            if existing_record and existing_record.get('processing_status') == 'completed':
                # Get existing faces for this image
                existing_faces = db_manager.get_faces_by_image_id(existing_record['id'])
                for face in existing_faces:
                    embedding = db_manager.get_face_embedding(face['id'])
                    if embedding is not None:
                        face_metadata = {
                            'image_id': face['image_id'],
                            'filename': existing_record['filename'],
                            'thumbnail_path': face['thumbnail_path'],
                            'confidence': face['confidence'],
                            'age': face['age'],
                            'gender': face['gender'],
                            'source': existing_record['source']
                        }
                        embeddings.append(embedding)
                        metadata.append(face_metadata)
                        processed_count += 1
                continue
            
            # Load and process new image
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            # Determine source
            source = 'dataset' if DATASET_FOLDER in image_path else 'crawled'
            
            # Add to database if not exists
            if not existing_record:
                file_size = os.path.getsize(image_path)
                height, width = image.shape[:2]
                image_id = db_manager.add_image(
                    file_path=image_path,
                    filename=os.path.basename(image_path),
                    source=source,
                    file_size=file_size,
                    width=width,
                    height=height
                )
            else:
                image_id = existing_record['id']
            
            # Extract face data
            face_data_list = face_engine.extract_face_embeddings(image)
            
            for face_data in face_data_list:
                try:
                    # Create face thumbnail
                    bbox = face_data['bbox']
                    thumbnail_name = f"face_{image_id}_{face_data['face_id']}.jpg"
                    thumbnail_path = image_processor.create_face_thumbnail(image, bbox, thumbnail_name)
                    
                    # Prepare metadata
                    face_metadata = {
                        'image_id': image_id,
                        'filename': os.path.basename(image_path),
                        'thumbnail_path': thumbnail_path,
                        'confidence': face_data['confidence'],
                        'age': face_data.get('age'),
                        'gender': face_data.get('gender'),
                        'source': source
                    }
                    
                    # Add to database
                    face_id = db_manager.add_face(image_id, face_data, face_data['embedding'], len(embeddings))
                    
                    # Store for FAISS
                    embeddings.append(face_data['embedding'])
                    metadata.append(face_metadata)
                    processed_count += 1
                    
                except Exception as e:
                    logging.warning(f"Error processing face in {image_path}: {e}")
                    continue
            
            # Update processing status
            db_manager.update_image_processing_status(image_id, 'completed', len(face_data_list))
            
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}")
            continue
    
    # Update FAISS index
    if embeddings:
        # Clear and rebuild index to avoid duplicates
        search_engine.clear_index()
        search_engine.add_embeddings(embeddings, metadata)
        search_engine.save_index()
    
    return processed_count

@app.route('/dataset_status')
def dataset_status():
    """Check if dataset is initialized using FAISS and database"""
    try:
        # Get FAISS statistics
        faiss_stats = search_engine.get_stats()
        
        # Get database statistics
        db_stats = db_manager.get_database_stats()
        
        total_faces = faiss_stats.get('total_vectors', 0)
        
        return jsonify({
            'initialized': total_faces > 0,
            'size': total_faces,
            'total_images': db_stats.get('total_images', 0),
            'processed_images': db_stats.get('processed_images', 0),
            'avg_confidence': db_stats.get('avg_confidence', 0),
            'images_by_source': db_stats.get('images_by_source', {}),
            'faiss_ready': faiss_stats.get('total_vectors', 0) > 0
        })
        
    except Exception as e:
        logging.error(f"Error getting dataset status: {e}")
        return jsonify({
            'initialized': False,
            'size': 0,
            'error': str(e)
        })

@app.route('/crawl_status')
def crawl_status():
    """Get web crawling statistics"""
    try:
        db_stats = db_manager.get_database_stats()
        crawler_stats = web_crawler.get_stats()
        
        return jsonify({
            'crawl_sessions': db_stats.get('total_crawl_sessions', 0),
            'completed_sessions': db_stats.get('completed_crawl_sessions', 0),
            'recent_searches': db_stats.get('searches_last_24h', 0),
            'crawler_stats': crawler_stats
        })
        
    except Exception as e:
        logging.error(f"Error getting crawl status: {e}")
        return jsonify({
            'error': str(e)
        })

@app.errorhandler(413)
def too_large(e):
    flash('File is too large. Please upload an image smaller than 16MB.', 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)
