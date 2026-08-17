"""
Command Line Interface for AI-Powered Smart OCR
Allows batch processing and command-line usage
"""

import argparse
import sys
import json
import csv
from pathlib import Path
from typing import List
import logging
from tqdm import tqdm

from ocr_processor import OCRProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_single_image(args):
    """Process a single image file"""
    processor = OCRProcessor(languages=args.languages.split(','), use_gpu=args.gpu)
    result = processor.process_image_file(args.input)
    
    if args.output:
        # Save to file
        output_path = Path(args.output)
        if output_path.suffix == '.json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        elif output_path.suffix == '.txt':
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result['full_text'])
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result['full_text'])
        logger.info(f"Results saved to {args.output}")
    else:
        # Print to console
        print(f"\n{'='*60}")
        print(f"OCR Results for: {args.input}")
        print(f"{'='*60}")
        print(f"Extracted Text:\n{result['full_text']}")
        print(f"\nConfidence: {result['average_confidence']:.2%}")
        print(f"Detections: {result['total_detections']}")
        print(f"{'='*60}\n")


def process_batch(args):
    """Process multiple images in batch"""
    processor = OCRProcessor(languages=args.languages.split(','), use_gpu=args.gpu)
    
    # Get image paths
    if args.input:
        # Single directory
        input_path = Path(args.input)
        if input_path.is_file():
            image_paths = [str(input_path)]
        else:
            image_paths = list(input_path.glob('*.[pj][np][g]')) + \
                         list(input_path.glob('*.jpeg')) + \
                         list(input_path.glob('*.bmp')) + \
                         list(input_path.glob('*.tiff')) + \
                         list(input_path.glob('*.webp'))
            image_paths = [str(p) for p in image_paths]
    else:
        # Multiple files
        image_paths = args.files
    
    if not image_paths:
        logger.error("No image files found")
        return
    
    logger.info(f"Processing {len(image_paths)} images...")
    
    # Process images with progress bar
    results = []
    for image_path in tqdm(image_paths, desc="Processing images"):
        result = processor.process_image_file(image_path)
        result['image_path'] = image_path
        results.append(result)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        if output_path.suffix == '.json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        elif output_path.suffix == '.csv':
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Image Path', 'Full Text', 'Confidence', 'Detections', 'Success'])
                for result in results:
                    writer.writerow([
                        result.get('image_path', ''),
                        result.get('full_text', ''),
                        result.get('average_confidence', 0),
                        result.get('total_detections', 0),
                        'error' not in result
                    ])
        else:
            # Default to text file
            with open(output_path, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(f"Image: {result.get('image_path', 'unknown')}\n")
                    f.write(f"Text: {result.get('full_text', '')}\n")
                    f.write(f"Confidence: {result.get('average_confidence', 0):.2%}\n")
                    f.write("-" * 60 + "\n\n")
        
        logger.info(f"Results saved to {args.output}")
    else:
        # Print summary
        successful = sum(1 for r in results if 'error' not in r)
        logger.info(f"Processed {len(results)} images: {successful} successful, {len(results)-successful} failed")
        
        # Print individual results
        for result in results:
            print(f"\n{'='*60}")
            print(f"Image: {result.get('image_path', 'unknown')}")
            print(f"{'='*60}")
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Text: {result.get('full_text', '')}")
                print(f"Confidence: {result.get('average_confidence', 0):.2%}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='AI-Powered Smart OCR - Robust Text Extraction from Real-World Images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image
  python cli.py -i image.jpg -l en
  
  # Process with GPU
  python cli.py -i image.jpg -l en --gpu
  
  # Batch process directory
  python cli.py -i ./images/ -o results.json
  
  # Multiple languages
  python cli.py -i image.jpg -l en,hi,es
  
  # Save to CSV
  python cli.py -i ./images/ -o results.csv
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-i', '--input', help='Input image file or directory')
    input_group.add_argument('-f', '--files', nargs='+', help='Multiple image files')
    
    # Output options
    parser.add_argument('-o', '--output', help='Output file (json, csv, txt)')
    
    # Processing options
    parser.add_argument('-l', '--languages', default='en',
                       help='Comma-separated language codes (default: en)')
    parser.add_argument('--gpu', action='store_true', default=False,
                       help='Use GPU acceleration (default: False)')
    parser.add_argument('--batch', action='store_true',
                       help='Batch processing mode')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input path does not exist: {args.input}")
            sys.exit(1)
    
    if args.files:
        for file_path in args.files:
            if not Path(file_path).exists():
                logger.error(f"File does not exist: {file_path}")
                sys.exit(1)
    
    # Process
    try:
        if args.batch or (args.input and Path(args.input).is_dir()) or args.files:
            process_batch(args)
        else:
            process_single_image(args)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()