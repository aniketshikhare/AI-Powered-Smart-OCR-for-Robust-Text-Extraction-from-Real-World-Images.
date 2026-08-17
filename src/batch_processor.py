"""
Batch Processing Module for AI-Powered Smart OCR
Handles large-scale text extraction operations
"""

import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm

from ocr_processor import OCRProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Advanced batch processing for multiple images
    """
    
    def __init__(self, languages: List[str] = None, use_gpu: bool = True, max_workers: int = 4):
        """
        Initialize batch processor
        
        Args:
            languages: List of language codes
            use_gpu: Whether to use GPU acceleration
            max_workers: Maximum number of parallel workers
        """
        self.languages = languages or ['en']
        self.use_gpu = use_gpu
        self.max_workers = max_workers
        self.processor = OCRProcessor(languages=languages, use_gpu=use_gpu)
        
    def process_directory(self, directory: str, 
                         output_dir: Optional[str] = None,
                         output_format: str = 'json') -> Dict:
        """
        Process all images in a directory
        
        Args:
            directory: Path to directory containing images
            output_dir: Directory to save results (optional)
            output_format: Output format ('json', 'csv', 'txt')
            
        Returns:
            Dictionary containing batch results
        """
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            raise ValueError(f"Invalid directory: {directory}")
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(dir_path.glob(f'*{ext}'))
            image_paths.extend(dir_path.glob(f'*{ext.upper()}'))
        
        image_paths = [str(p) for p in image_paths]
        
        if not image_paths:
            logger.warning(f"No image files found in {directory}")
            return {
                'total_files': 0,
                'successful': 0,
                'failed': 0,
                'results': []
            }
        
        logger.info(f"Found {len(image_paths)} images in {directory}")
        
        # Process images
        results = self.process_batch(image_paths)
        
        # Save results if output directory specified
        if output_dir:
            self.save_results(results, output_dir, output_format)
        
        return results
    
    def process_batch(self, image_paths: List[str]) -> Dict:
        """
        Process multiple images in parallel
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Dictionary containing batch results
        """
        results = []
        successful = 0
        failed = 0
        
        # Process images in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.processor.process_image_file, path): path 
                for path in image_paths
            }
            
            # Process results as they complete
            for future in tqdm(as_completed(future_to_path), 
                             total=len(image_paths), 
                             desc="Processing images"):
                path = future_to_path[future]
                try:
                    result = future.result()
                    result['image_path'] = path
                    result['timestamp'] = datetime.now().isoformat()
                    result['success'] = 'error' not in result
                    
                    if result['success']:
                        successful += 1
                    else:
                        failed += 1
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Failed to process {path}: {e}")
                    failed += 1
                    results.append({
                        'image_path': path,
                        'error': str(e),
                        'success': False,
                        'timestamp': datetime.now().isoformat()
                    })
        
        return {
            'total_files': len(image_paths),
            'successful': successful,
            'failed': failed,
            'results': results,
            'languages': self.languages
        }
    
    def save_results(self, batch_results: Dict, output_dir: str, output_format: str = 'json'):
        """
        Save batch results to files
        
        Args:
            batch_results: Dictionary containing batch results
            output_dir: Directory to save results
            output_format: Output format ('json', 'csv', 'txt')
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_format == 'json':
            output_file = output_path / f"ocr_results_{timestamp}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(batch_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_file}")
            
        elif output_format == 'csv':
            output_file = output_path / f"ocr_results_{timestamp}.csv"
            with open(output_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Image Path', 'Full Text', 'Confidence', 
                               'Detections', 'Success', 'Timestamp', 'Error'])
                for result in batch_results['results']:
                    writer.writerow([
                        result.get('image_path', ''),
                        result.get('full_text', ''),
                        result.get('average_confidence', 0),
                        result.get('total_detections', 0),
                        result.get('success', False),
                        result.get('timestamp', ''),
                        result.get('error', '')
                    ])
            logger.info(f"Results saved to {output_file}")
            
        elif output_format == 'txt':
            output_file = output_path / f"ocr_results_{timestamp}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Batch OCR Results - {timestamp}\n")
                f.write(f"{'='*60}\n")
                f.write(f"Total Files: {batch_results['total_files']}\n")
                f.write(f"Successful: {batch_results['successful']}\n")
                f.write(f"Failed: {batch_results['failed']}\n")
                f.write(f"Languages: {', '.join(batch_results['languages'])}\n")
                f.write(f"{'='*60}\n\n")
                
                for result in batch_results['results']:
                    f.write(f"Image: {result.get('image_path', 'unknown')}\n")
                    f.write(f"Success: {result.get('success', False)}\n")
                    if result.get('success'):
                        f.write(f"Text: {result.get('full_text', '')}\n")
                        f.write(f"Confidence: {result.get('average_confidence', 0):.2%}\n")
                        f.write(f"Detections: {result.get('total_detections', 0)}\n")
                    else:
                        f.write(f"Error: {result.get('error', 'Unknown error')}\n")
                    f.write("-" * 60 + "\n\n")
            logger.info(f"Results saved to {output_file}")
    
    def generate_summary_report(self, batch_results: Dict, output_path: str):
        """
        Generate a summary report of batch processing
        
        Args:
            batch_results: Dictionary containing batch results
            output_path: Path to save the summary report
        """
        # Calculate statistics
        confidences = [r.get('average_confidence', 0) for r in batch_results['results'] if r.get('success')]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        detection_counts = [r.get('total_detections', 0) for r in batch_results['results'] if r.get('success')]
        avg_detections = sum(detection_counts) / len(detection_counts) if detection_counts else 0
        
        # Generate report
        report = {
            'summary': {
                'total_files': batch_results['total_files'],
                'successful': batch_results['successful'],
                'failed': batch_results['failed'],
                'success_rate': batch_results['successful'] / batch_results['total_files'] if batch_results['total_files'] > 0 else 0,
                'average_confidence': avg_confidence,
                'average_detections': avg_detections,
                'languages': batch_results['languages']
            },
            'failed_files': [
                r['image_path'] for r in batch_results['results'] if not r.get('success')
            ],
            'high_confidence_files': [
                r['image_path'] for r in batch_results['results'] 
                if r.get('success') and r.get('average_confidence', 0) > 0.8
            ],
            'low_confidence_files': [
                r['image_path'] for r in batch_results['results'] 
                if r.get('success') and r.get('average_confidence', 0) < 0.5
            ]
        }
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Summary report saved to {output_path}")
        return report


def main():
    """Example usage of batch processor"""
    processor = BatchProcessor(languages=['en'], use_gpu=False, max_workers=2)
    
    # Process directory
    results = processor.process_directory(
        directory='./data/images',
        output_dir='./data/results',
        output_format='json'
    )
    
    # Generate summary report
    processor.generate_summary_report(results, './data/results/summary.json')
    
    print(f"Processed {results['total_files']} files: {results['successful']} successful, {results['failed']} failed")


if __name__ == '__main__':
    main()