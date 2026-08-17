"""
Test performance optimization features
"""

import sys
import os
import time

# Change to src directory
os.chdir('src')
sys.path.insert(0, os.getcwd())

from ocr_processor import OCRProcessor
from performance_optimizer import PerformanceOptimizer, MemoryOptimizer

def test_caching():
    """Test caching functionality"""
    print("Testing Caching Performance...")
    
    # Create processor with caching enabled
    processor = OCRProcessor(languages=['en'], use_gpu=False, enable_cache=True)
    
    # First run (no cache)
    start_time = time.time()
    result1 = processor.process_image_file('../data/images/test1.jpg')
    first_run_time = time.time() - start_time
    
    # Second run (should use cache)
    start_time = time.time()
    result2 = processor.process_image_file('../data/images/test1.jpg')
    second_run_time = time.time() - start_time
    
    print(f"First run time: {first_run_time:.2f}s")
    print(f"Second run time (cached): {second_run_time:.2f}s")
    print(f"Speed improvement: {(first_run_time / second_run_time):.2f}x")
    
    # Get cache stats
    stats = processor.performance_optimizer.get_cache_stats()
    print(f"Cache stats: {stats}")
    
    print("Caching test completed")

def test_memory_optimization():
    """Test memory optimization"""
    print("\nTesting Memory Optimization...")
    
    memory_optimizer = MemoryOptimizer(max_memory_mb=1024)
    
    # Test memory estimation
    test_image = '../data/images/test1.jpg'
    estimated_memory = memory_optimizer.estimate_image_memory(test_image)
    print(f"Estimated memory for test image: {estimated_memory:.2f} MB")
    
    # Test memory allocation
    can_process = memory_optimizer.can_process_image(test_image)
    print(f"Can process image: {can_process}")
    
    if can_process:
        memory_optimizer.allocate_memory(test_image)
        status = memory_optimizer.get_memory_status()
        print(f"Memory status after allocation: {status}")
        
        memory_optimizer.free_memory(test_image)
        status = memory_optimizer.get_memory_status()
        print(f"Memory status after freeing: {status}")
    
    print("Memory optimization test completed")

def test_batch_with_optimization():
    """Test batch processing with optimization"""
    print("\nTesting Batch Processing with Optimization...")
    
    from batch_processor import BatchProcessor
    
    # Create batch processor with optimization
    batch_processor = BatchProcessor(
        languages=['en'], 
        use_gpu=False, 
        max_workers=2,
        enable_cache=True,
        enable_memory_opt=False
    )
    
    # Process directory
    start_time = time.time()
    results = batch_processor.process_directory(
        directory='../data/images',
        output_dir='../data/results',
        output_format='json'
    )
    total_time = time.time() - start_time
    
    print(f"Processed {results['total_files']} files in {total_time:.2f}s")
    print(f"Average time per file: {total_time / results['total_files']:.2f}s")
    print(f"Success rate: {results['successful']}/{results['total_files']} ({(results['successful']/results['total_files'])*100:.1f}%)")
    
    print("Batch processing test completed")

if __name__ == "__main__":
    print("Testing Performance Optimization Features\n")
    
    try:
        test_caching()
        test_memory_optimization()
        test_batch_with_optimization()
        
        print("\nAll performance tests completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()