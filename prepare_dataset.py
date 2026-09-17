import os
import sys
import json
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

def get_thai_char(code_str):
    try:
        val = int(code_str)
        return bytes([val]).decode('tis-620')
    except Exception:
        return code_str

def prepare_data(data_dir=r"ThaiCharacter Dataset/round2", output_dir=".", seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    classes = sorted(
        [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d != '__MACOSX'],
        key=lambda x: int(x) if x.isdigit() else x
    )
    
    print(f"Found {len(classes)} classes in {data_dir}")
    
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for i, c in enumerate(classes)}
    
    char_mapping = {}
    for c in classes:
        char_th = get_thai_char(c)
        char_mapping[c] = {
            "index": class_to_idx[c],
            "thai_char": char_th,
            "code": c
        }
    
    records = []
    class_counts = {}
    
    for c in classes:
        c_path = os.path.join(data_dir, c)
        files = [f for f in os.listdir(c_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        class_counts[c] = len(files)
        char_th = get_thai_char(c)
        
        # Split 80% train, 20% val
        shuffled = files.copy()
        random.shuffle(shuffled)
        n = len(shuffled)
        
        if n == 1:
            # 1 sample: put in train, also copy to val for evaluation representation
            records.append({
                "filepath": os.path.join(c_path, shuffled[0]).replace("\\", "/"),
                "filename": shuffled[0],
                "class_code": c,
                "class_idx": class_to_idx[c],
                "thai_char": char_th,
                "split": "train"
            })
            records.append({
                "filepath": os.path.join(c_path, shuffled[0]).replace("\\", "/"),
                "filename": shuffled[0],
                "class_code": c,
                "class_idx": class_to_idx[c],
                "thai_char": char_th,
                "split": "val"
            })
        else:
            n_val = max(1, int(round(n * 0.2)))
            n_train = n - n_val
            
            for f in shuffled[:n_train]:
                records.append({
                    "filepath": os.path.join(c_path, f).replace("\\", "/"),
                    "filename": f,
                    "class_code": c,
                    "class_idx": class_to_idx[c],
                    "thai_char": char_th,
                    "split": "train"
                })
            for f in shuffled[n_train:]:
                records.append({
                    "filepath": os.path.join(c_path, f).replace("\\", "/"),
                    "filename": f,
                    "class_code": c,
                    "class_idx": class_to_idx[c],
                    "thai_char": char_th,
                    "split": "val"
                })
                
    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "dataset_metadata.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"Saved metadata to {csv_path} (Total samples: {len(df)})")
    
    # Summary of split
    train_count = len(df[df['split'] == 'train'])
    val_count = len(df[df['split'] == 'val'])
    print(f"Train samples: {train_count} ({train_count/len(df)*100:.1f}%)")
    print(f"Val samples:   {val_count} ({val_count/len(df)*100:.1f}%)")
    
    # Save mappings
    with open(os.path.join(output_dir, "char_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(char_mapping, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)
        
    print("Saved char_mapping.json and classes.json")
    
    # Generate distribution plot
    plt.figure(figsize=(14, 6))
    labels = [f"{c}\n({get_thai_char(c)})" for c in classes]
    counts = [class_counts[c] for c in classes]
    
    bars = plt.bar(range(len(classes)), counts, color="#3498db", edgecolor="#2980b9")
    plt.xticks(range(len(classes)), labels, fontsize=9, rotation=0)
    plt.ylabel("Number of Images", fontsize=12)
    plt.xlabel("Thai Character Class (TIS-620 Code)", fontsize=12)
    plt.title(f"Dataset Class Distribution - {len(classes)} Classes (Total {sum(counts):,} Images)\nExtreme Imbalance: Min {min(counts)} vs Max {max(counts):,}", fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        if height < 100:
            plt.annotate(f'{height}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=8, color='red', fontweight='bold')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "dataset_distribution.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved distribution chart to {plot_path}")
    
    return df, char_mapping

if __name__ == "__main__":
    prepare_data()
