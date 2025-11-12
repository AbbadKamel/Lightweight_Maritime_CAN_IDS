#!/usr/bin/env python3
"""
Merge all N2K frame files into one CSV
"""
import pandas as pd
from pathlib import Path
import glob

def merge_n2k_files(input_dir, output_file, max_messages=100000):
    """
    Merge multiple N2K txt files into one CSV
    
    Args:
        input_dir: Directory with Frame*.txt files
        output_file: Output CSV file path
        max_messages: Maximum number of messages to include (default 100k)
    """
    print("="*60)
    print("Merging N2K Files")
    print("="*60)
    
    # Find all Frame files
    input_path = Path(input_dir)
    files = sorted(input_path.glob("Frame251016*.txt"))
    
    print(f"\nFound {len(files)} files")
    print(f"Will merge up to {max_messages:,} messages")
    
    # Read and concatenate
    all_data = []
    total_read = 0
    
    for i, file in enumerate(files):
        if total_read >= max_messages:
            break
            
        print(f"\rProcessing file {i+1}/{len(files)}: {file.name[:30]}...", end='')
        
        try:
            # Read TSV file (tab-separated)
            df = pd.read_csv(file, sep='\t', encoding='utf-8')
            
            # Take only what we need
            remaining = max_messages - total_read
            if len(df) > remaining:
                df = df.head(remaining)
            
            all_data.append(df)
            total_read += len(df)
            
        except Exception as e:
            print(f"\nError reading {file.name}: {e}")
            continue
    
    print(f"\n\nConcatenating {len(all_data)} dataframes...")
    merged_df = pd.concat(all_data, ignore_index=True)
    
    # Clean up column names (remove extra spaces)
    merged_df.columns = merged_df.columns.str.strip()
    
    # Save as CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Merged {len(merged_df):,} messages")
    print(f"✓ Saved to: {output_path}")
    print(f"{'='*60}")
    
    # Show stats
    print(f"\nData Statistics:")
    print(f"  - Total messages: {len(merged_df):,}")
    print(f"  - Unique CAN IDs: {merged_df['ID'].nunique()}")
    print(f"  - Time range: {merged_df['Timesteamp'].iloc[0]} to {merged_df['Timesteamp'].iloc[-1]}")
    print(f"\nTop 10 CAN IDs:")
    print(merged_df['ID'].value_counts().head(10))
    
    return merged_df


if __name__ == "__main__":
    # Merge files
    # Start with 100k messages for initial testing
    df = merge_n2k_files(
        input_dir="Frame brute cantest",
        output_file="data/raw/n2k/normal/n2k_real_data_100k.csv",
        max_messages=100000
    )
    
    print("\n✅ Ready to use!")
    print("\nNext step: python scripts/02_preprocess_data.py")
