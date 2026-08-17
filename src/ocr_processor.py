"""
AI-Powered Smart OCR Processor
A robust text extraction system for real-world images
"""

import cv2
import numpy as np
import easyocr
import pytesseract
from PIL import Image
import imutils
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
from performance_optimizer import PerformanceOptimizer, MemoryOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCRProcessor:
    """
    Advanced OCR processor with multiple engines and preprocessing capabilities
    """
    
    def __init__(self, languages: List[str] = None, use_gpu: bool = True, 
                 enable_cache: bool = True, enable_memory_opt: bool = False):
        """
        Initialize the OCR processor
        
        Args:
            languages: List of language codes (e.g., ['en', 'hi'])
            use_gpu: Whether to use GPU acceleration
            enable_cache: Whether to enable result caching
            enable_memory_opt: Whether to enable memory optimization
        """
        self.languages = languages or ['en']
        self.use_gpu = use_gpu
        self.easyocr_reader = None
        self._initialize_engines()
        
        # Performance optimization
        self.performance_optimizer = PerformanceOptimizer(enable_cache=enable_cache)
        self.memory_optimizer = MemoryOptimizer() if enable_memory_opt else None
        self.enable_cache = enable_cache
        self.enable_memory_opt = enable_memory_opt
        
    def _initialize_engines(self):
        """Initialize OCR engines"""
        try:
            logger.info("Initializing EasyOCR reader...")
            self.easyocr_reader = easyocr.Reader(
                self.languages, 
                gpu=self.use_gpu,
                verbose=False
            )
            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            self.easyocr_reader = None
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply advanced preprocessing to improve OCR accuracy
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive thresholding for better text extraction
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
        
        # Apply morphological operations
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        
        return processed
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality for better OCR results
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def extract_text_easyocr(self, image: np.ndarray, detail: int = 1) -> List[Dict]:
        """
        Extract text using EasyOCR engine
        
        Args:
            image: Input image
            detail: Level of detail (0 for text only, 1 for full details)
            
        Returns:
            List of extracted text with bounding boxes and confidence
        """
        if self.easyocr_reader is None:
            logger.error("EasyOCR reader not initialized")
            return []
        
        try:
            # Preprocess image
            processed = self.preprocess_image(image)
            
            # Extract text
            results = self.easyocr_reader.readtext(
                processed, 
                detail=detail,
                paragraph=False
            )
            
            # Format results
            formatted_results = []
            for result in results:
                if detail == 1:
                    bbox, text, confidence = result
                    formatted_results.append({
                        'text': text,
                        'bbox': bbox,
                        'confidence': float(confidence),
                        'engine': 'easyocr'
                    })
                else:
                    formatted_results.append({
                        'text': result,
                        'engine': 'easyocr'
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return []
    
    def extract_text_tesseract(self, image: np.ndarray) -> List[Dict]:
        """
        Extract text using Tesseract OCR engine (fallback)
        
        Args:
            image: Input image
            
        Returns:
            List of extracted text with confidence
        """
        try:
            # Preprocess image
            processed = self.preprocess_image(image)
            
            # Configure Tesseract
            config = r'--oem 3 --psm 6'
            
            # Extract text with confidence data
            data = pytesseract.image_to_data(
                processed, 
                config=config,
                output_type=pytesseract.Output.DICT
            )
            
            # Process results
            results = []
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                text = data['text'][i].strip()
                if text and int(data['conf'][i]) > 0:
                    results.append({
                        'text': text,
                        'bbox': [
                            [data['left'][i], data['top'][i]],
                            [data['left'][i] + data['width'][i], data['top'][i]],
                            [data['left'][i] + data['width'][i], data['top'][i] + data['height'][i]],
                            [data['left'][i], data['top'][i] + data['height'][i]]
                        ],
                        'confidence': float(data['conf'][i]) / 100.0,
                        'engine': 'tesseract'
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return []
    
    def extract_text(self, image: np.ndarray, use_fallback: bool = True) -> Dict:
        """
        Extract text using the best available OCR engine
        
        Args:
            image: Input image
            use_fallback: Whether to use fallback engines if primary fails
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        # Enhance image
        enhanced = self.enhance_image(image)
        
        # Try EasyOCR first
        results = self.extract_text_easyocr(enhanced)
        
        # Use fallback if needed
        if use_fallback and (not results or len(results) == 0):
            logger.info("EasyOCR failed, trying Tesseract fallback")
            results = self.extract_text_tesseract(enhanced)
        
        # Calculate overall confidence
        if results:
            avg_confidence = np.mean([r['confidence'] for r in results])
        else:
            avg_confidence = 0.0
        
        return {
            'results': results,
            'full_text': ' '.join([r['text'] for r in results]),
            'average_confidence': avg_confidence,
            'total_detections': len(results),
            'languages': self.languages
        }
    
    def process_image_file(self, image_path: str, options: Dict = None) -> Dict:
        """
        Process an image file and extract text
        
        Args:
            image_path: Path to the image file
            options: Dictionary of processing options
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        options = options or {}
        
        # Check cache first
        if self.enable_cache:
            cached_result = self.performance_optimizer.get_cached_result(
                image_path, tuple(self.languages), options
            )
            if cached_result:
                return cached_result
        
        # Memory optimization check
        if self.enable_memory_opt and self.memory_optimizer:
            if not self.memory_optimizer.can_process_image(image_path):
                logger.warning(f"Memory constraints prevent processing {image_path}")
                return {
                    'error': 'Memory constraints prevent processing',
                    'results': [],
                    'full_text': '',
                    'average_confidence': 0.0
                }
            self.memory_optimizer.allocate_memory(image_path)
        
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Extract text
            result = self.extract_text(image)
            result['image_path'] = image_path
            result['image_size'] = image.shape[:2]
            
            # Cache the result
            if self.enable_cache and 'error' not in result:
                self.performance_optimizer.cache_result(
                    image_path, tuple(self.languages), options, result
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process image file: {e}")
            return {
                'error': str(e),
                'results': [],
                'full_text': '',
                'average_confidence': 0.0
            }
        finally:
            # Free memory
            if self.enable_memory_opt and self.memory_optimizer:
                self.memory_optimizer.free_memory(image_path)
    
    def batch_process(self, image_paths: List[str]) -> List[Dict]:
        """
        Process multiple images in batch
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of results for each image
        """
        results = []
        for image_path in image_paths:
            result = self.process_image_file(image_path)
            results.append(result)
        
        return results


def main():
    """Example usage of the OCR processor"""
    # Initialize processor
    processor = OCRProcessor(languages=['en'], use_gpu=False)
    
    # Example image processing
    image_path = "example.jpg"
    result = processor.process_image_file(image_path)
    
    print(f"Extracted text: {result['full_text']}")
    print(f"Average confidence: {result['average_confidence']:.2f}")
    print(f"Total detections: {result['total_detections']}")


if __name__ == "__main__":
    main()