#!/usr/bin/env python3
"""
Example: Memory-Efficient Data Processing with Streaming

This example demonstrates how to use the new streaming data loader
for memory-efficient processing of large files.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from services.file_io_service import FileIOService
from utils.memory_manager import get_memory_stats, start_memory_monitoring
from utils.streaming_loader import StreamingDataLoader

def create_sample_large_csv(file_path: str, num_rows: int = 50000):
    """Create a sample large CSV file for testing."""
    print(f"Creating sample CSV with {num_rows} rows...")
    
    # Generate sample data
    data = {
        'id': range(1, num_rows + 1),
        'text_column': [f"This is sample text for row {i}" for i in range(1, num_rows + 1)],
        'category': np.random.choice(['A', 'B', 'C', 'D'], num_rows),
        'numeric_value': np.random.randn(num_rows),
        'date_column': pd.date_range('2023-01-01', periods=num_rows, freq='H')
    }
    
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"Created {file_path} ({df.memory_usage(deep=True).sum() / 1024**2:.1f} MB)")
    return file_path

def example_basic_streaming():
    """Example 1: Basic streaming file loading."""
    print("\n" + "="*60)
    print("Example 1: Basic Streaming File Loading")
    print("="*60)
    
    # Start memory monitoring
    start_memory_monitoring()
    
    # Create sample file
    sample_file = "temp_large_sample.csv"
    create_sample_large_csv(sample_file, 20000)
    
    try:
        # Initialize FileIO service with streaming enabled
        file_service = FileIOService(enable_streaming=True, streaming_chunk_size=5000)
        
        print(f"\n📊 Initial memory usage: {get_memory_stats().process_mb:.1f} MB")
        
        # Stream the file
        chunk_count = 0
        total_rows = 0
        
        for chunk, metadata in file_service.load_file_streaming(sample_file):
            chunk_count += 1
            total_rows += len(chunk)
            
            if metadata:
                print(f"\n📄 File metadata:")
                print(f"   • File: {metadata.file_name}")
                print(f"   • Format: {metadata.file_format}")
                print(f"   • Columns: {len(metadata.columns)}")
                print(f"   • Supports streaming: {metadata.supports_streaming}")
                print(f"   • Estimated rows: {metadata.row_count}")
            
            # Show progress every few chunks
            if chunk_count % 2 == 0:
                current_memory = get_memory_stats()
                print(f"   Chunk {chunk_count}: {len(chunk)} rows, "
                      f"Memory: {current_memory.process_mb:.1f} MB")
        
        final_memory = get_memory_stats()
        print(f"\n✅ Streaming completed:")
        print(f"   • Total chunks: {chunk_count}")
        print(f"   • Total rows processed: {total_rows}")
        print(f"   • Final memory usage: {final_memory.process_mb:.1f} MB")
        
    finally:
        # Cleanup
        if os.path.exists(sample_file):
            os.remove(sample_file)

def example_memory_efficient_processing():
    """Example 2: Memory-efficient data processing."""
    print("\n" + "="*60)
    print("Example 2: Memory-Efficient Data Processing")
    print("="*60)
    
    # Create sample file
    sample_file = "temp_processing_sample.csv"
    create_sample_large_csv(sample_file, 30000)
    
    try:
        file_service = FileIOService(enable_streaming=True, streaming_chunk_size=8000)
        
        print(f"\n📊 Initial memory: {get_memory_stats().process_mb:.1f} MB")
        
        # Define a processing function
        def analyze_chunk(chunk):
            """Analyze a chunk and return summary statistics."""
            return {
                'row_count': len(chunk),
                'avg_numeric': chunk['numeric_value'].mean(),
                'category_counts': chunk['category'].value_counts().to_dict(),
                'text_length_avg': chunk['text_column'].str.len().mean()
            }
        
        # Define a function to combine results
        def combine_results(results):
            """Combine results from all chunks."""
            total_rows = sum(r['row_count'] for r in results)
            avg_numeric = np.mean([r['avg_numeric'] for r in results])
            
            # Combine category counts
            all_categories = {}
            for r in results:
                for cat, count in r['category_counts'].items():
                    all_categories[cat] = all_categories.get(cat, 0) + count
            
            avg_text_length = np.mean([r['text_length_avg'] for r in results])
            
            return {
                'total_rows': total_rows,
                'overall_avg_numeric': avg_numeric,
                'category_distribution': all_categories,
                'avg_text_length': avg_text_length
            }
        
        # Process file in chunks
        results = file_service.process_file_in_chunks(
            sample_file, 
            analyze_chunk, 
            combine_results
        )
        
        print(f"\n📈 Processing Results:")
        print(f"   • Total rows: {results['total_rows']}")
        print(f"   • Average numeric value: {results['overall_avg_numeric']:.3f}")
        print(f"   • Average text length: {results['avg_text_length']:.1f}")
        print(f"   • Category distribution: {results['category_distribution']}")
        
        final_memory = get_memory_stats()
        print(f"\n💾 Final memory usage: {final_memory.process_mb:.1f} MB")
        
    finally:
        # Cleanup
        if os.path.exists(sample_file):
            os.remove(sample_file)

def example_file_sampling():
    """Example 3: Efficient file sampling."""
    print("\n" + "="*60)
    print("Example 3: Efficient File Sampling")
    print("="*60)
    
    # Create sample file
    sample_file = "temp_sampling_sample.csv"
    create_sample_large_csv(sample_file, 100000)
    
    try:
        file_service = FileIOService(enable_streaming=True)
        
        print(f"\n📊 Initial memory: {get_memory_stats().process_mb:.1f} MB")
        
        # Load just a sample for preview
        sample_df, metadata = file_service.load_file_sample(sample_file, sample_size=500)
        
        print(f"\n📋 Sample Results:")
        print(f"   • Sample size: {len(sample_df)} rows")
        print(f"   • Is sample: {metadata.is_sample}")
        print(f"   • Sample size in metadata: {metadata.sample_size}")
        print(f"   • File supports streaming: {metadata.supports_streaming}")
        print(f"   • Estimated total rows: {metadata.row_count}")
        
        # Show sample data
        print(f"\n📄 Sample data preview:")
        print(sample_df.head().to_string())
        
        # Show memory usage
        sample_memory = sample_df.memory_usage(deep=True).sum() / 1024**2
        final_memory = get_memory_stats()
        print(f"\n💾 Sample memory usage: {sample_memory:.2f} MB")
        print(f"💾 Total memory usage: {final_memory.process_mb:.1f} MB")
        
    finally:
        # Cleanup
        if os.path.exists(sample_file):
            os.remove(sample_file)

def example_memory_comparison():
    """Example 4: Memory usage comparison."""
    print("\n" + "="*60)
    print("Example 4: Memory Usage Comparison")
    print("="*60)
    
    # Create sample file
    sample_file = "temp_comparison_sample.csv"
    create_sample_large_csv(sample_file, 40000)
    
    try:
        print(f"\n📊 Initial memory: {get_memory_stats().process_mb:.1f} MB")
        
        # Method 1: Traditional loading
        print("\n🔸 Method 1: Traditional Loading")
        traditional_service = FileIOService(enable_streaming=False)
        
        before_memory = get_memory_stats().process_mb
        df_traditional, _ = traditional_service.load_file(sample_file)
        after_memory = get_memory_stats().process_mb
        
        traditional_increase = after_memory - before_memory
        print(f"   Memory increase: {traditional_increase:.1f} MB")
        print(f"   DataFrame memory: {df_traditional.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Clean up
        del df_traditional
        import gc
        gc.collect()
        
        # Method 2: Streaming with sample
        print("\n🔸 Method 2: Streaming with Sample")
        streaming_service = FileIOService(enable_streaming=True)
        
        before_memory = get_memory_stats().process_mb
        sample_df, _ = streaming_service.load_file_sample(sample_file, sample_size=1000)
        after_memory = get_memory_stats().process_mb
        
        streaming_increase = after_memory - before_memory
        print(f"   Memory increase: {streaming_increase:.1f} MB")
        print(f"   Sample memory: {sample_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        print(f"\n📊 Comparison:")
        print(f"   Traditional: {traditional_increase:.1f} MB")
        print(f"   Streaming sample: {streaming_increase:.1f} MB")
        print(f"   Memory savings: {traditional_increase - streaming_increase:.1f} MB")
        print(f"   Reduction: {((traditional_increase - streaming_increase) / traditional_increase * 100):.1f}%")
        
    finally:
        # Cleanup
        if os.path.exists(sample_file):
            os.remove(sample_file)

def main():
    """Run all examples."""
    print("🚀 Streaming Data Loader Examples")
    print("=" * 50)
    
    try:
        example_basic_streaming()
        example_memory_efficient_processing()
        example_file_sampling()
        example_memory_comparison()
        
        print("\n" + "="*60)
        print("✅ All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        raise

if __name__ == "__main__":
    main()