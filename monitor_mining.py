import os
import json
from pathlib import Path
from collections import defaultdict

def monitor_progress():
    MINED_DIR = "./storage/mined_knowledge"
    
    if not os.path.exists(MINED_DIR):
        print(f"❌ Storage directory not found at {MINED_DIR}")
        return

    # Tracking data
    stats = defaultdict(lambda: {"files": 0, "chunks": 0})
    total_files = 0
    total_chunks = 0

    # Scan the directory
    for root, dirs, files in os.walk(MINED_DIR):
        # We only care about files in product subfolders
        path_parts = Path(root).parts
        if len(path_parts) < 3:
            continue
            
        product_id = path_parts[-1]
        
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(root, file)
                
                # Count lines (each line is a chunk)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        line_count = sum(1 for _ in f)
                    
                    stats[product_id]["files"] += 1
                    stats[product_id]["chunks"] += line_count
                    total_files += 1
                    total_chunks += line_count
                except Exception:
                    continue

    # Display Output
    print("\n" + "="*50)
    print(f"{'PRODUCT ID':<20} | {'FILES':<8} | {'CHUNKS'}")
    print("-" * 50)
    
    # Sort by products with most chunks
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['chunks'], reverse=True)
    
    for product, data in sorted_stats:
        print(f"{product:<20} | {data['files']:<8} | {data['chunks']}")

    print("-" * 50)
    print(f"{'TOTAL PROGRESS':<20} | {total_files:<8} | {total_chunks}")
    print("="*50 + "\n")

if __name__ == "__main__":
    monitor_progress()