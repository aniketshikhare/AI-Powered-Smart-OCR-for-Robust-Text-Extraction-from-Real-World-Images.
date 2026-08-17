"""
Flask Web Application for AI-Powered Smart OCR
Provides REST API and web interface for text extraction
"""

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import uuid
from pathlib import Path
import logging
from datetime import datetime

# Import our OCR processor
from ocr_processor import OCRProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'data/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize OCR processor (lazy loading)
ocr_processor = None


def convert_to_serializable(obj):
    """Convert numpy types to regular Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

def get_ocr_processor(languages=None, use_gpu=True):
    """Get or create OCR processor instance"""
    global ocr_processor
    if ocr_processor is None:
        ocr_processor = OCRProcessor(languages=languages or ['en'], use_gpu=use_gpu)
    return ocr_processor


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Render the main web interface"""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'AI-Powered Smart OCR'
    })


@app.route('/api/extract', methods=['POST'])
def extract_text():
    """
    Extract text from uploaded image
    
    Expected form data:
    - file: Image file
    - languages: (optional) Comma-separated language codes
    - use_gpu: (optional) Boolean for GPU usage
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save file
        file.save(filepath)
        
        # Get parameters
        languages = request.form.get('languages', 'en').split(',')
        use_gpu = request.form.get('use_gpu', 'true').lower() == 'true'
        
        # Process image
        processor = get_ocr_processor(languages=languages, use_gpu=use_gpu)
        result = processor.process_image_file(filepath)
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        # Add metadata
        result['timestamp'] = datetime.now().isoformat()
        result['success'] = 'error' not in result
        
        return jsonify(convert_to_serializable(result))
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch', methods=['POST'])
def batch_extract():
    """
    Extract text from multiple images
    
    Expected form data:
    - files: Multiple image files
    - languages: (optional) Comma-separated language codes
    - use_gpu: (optional) Boolean for GPU usage
    """
    try:
        # Check if files are present
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        
        if not files or len(files) == 0:
            return jsonify({'error': 'No files selected'}), 400
        
        # Get parameters
        languages = request.form.get('languages', 'en').split(',')
        use_gpu = request.form.get('use_gpu', 'true').lower() == 'true'
        
        # Process images
        processor = get_ocr_processor(languages=languages, use_gpu=use_gpu)
        results = []
        
        for file in files:
            if file and allowed_file(file.filename):
                # Generate unique filename
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4()}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                
                # Save file
                file.save(filepath)
                
                # Process image
                result = processor.process_image_file(filepath)
                result['original_filename'] = filename
                result['timestamp'] = datetime.now().isoformat()
                result['success'] = 'error' not in result
                
                results.append(result)
                
                # Clean up uploaded file
                try:
                    os.remove(filepath)
                except:
                    pass
        
        return jsonify(convert_to_serializable({
            'total_files': len(results),
            'successful': sum(1 for r in results if r['success']),
            'failed': sum(1 for r in results if not r['success']),
            'results': results
        }))
        
    except Exception as e:
        logger.error(f"Error processing batch request: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/languages', methods=['GET'])
def get_supported_languages():
    """Get list of supported languages"""
    # Common language codes supported by EasyOCR
    languages = {
        'en': 'English',
        'hi': 'Hindi',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'ja': 'Japanese',
        'ko': 'Korean',
        'zh': 'Chinese',
        'ar': 'Arabic',
        'th': 'Thai',
        'vi': 'Vietnamese'
    }
    
    return jsonify({
        'languages': languages,
        'note': 'EasyOCR supports 80+ languages. These are the most common ones.'
    })


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Run the application
    app.run(host='0.0.0.0', port=5000, debug=True)