import os
import sys
import json
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as transforms

from models import get_model

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

def load_checkpoint(checkpoint_path, device):
    """
    Loads saved checkpoint and reconstructs model, classes, and char mapping.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("model_name", "resnet50")
    num_classes = checkpoint.get("num_classes", 72)
    classes = checkpoint.get("classes", [])
    char_mapping = checkpoint.get("char_mapping", {})
    img_size = checkpoint.get("img_size", 224)
    
    model = get_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return model, transform, classes, char_mapping, checkpoint

def visualize_batch_predictions(model, transform, classes, char_mapping, device, metadata_file="dataset_metadata.csv", num_samples=12, output_img=None):
    """
    Selects random validation images and plots prediction grid with Thai characters.
    """
    if output_img is None:
        output_img = "figures/sample_predictions.png" if os.path.exists("figures") else "sample_predictions.png"

    import pandas as pd
    if not os.path.exists(metadata_file):
        print(f"Metadata {metadata_file} not found. Skipping visualization.")
        return
        
    df = pd.read_csv(metadata_file)
    val_df = df[df['split'] == 'val'].sample(min(num_samples, len(df[df['split'] == 'val'])), random_state=42)
    
    cols = 4
    rows = (len(val_df) + cols - 1) // cols
    plt.figure(figsize=(15, rows * 3.5))
    
    for idx, (_, row) in enumerate(val_df.iterrows()):
        img_path = row['filepath']
        true_char = row['thai_char']
        true_code = str(row['class_code'])
        
        preds = predict_single_image(img_path, model, transform, classes, char_mapping, device, topk=1)
        pred_char = preds[0]['thai_char']
        pred_code = preds[0]['class_code']
        conf = preds[0]['percentage']
        
        is_correct = (pred_code == true_code)
        color = "green" if is_correct else "red"
        
        img = Image.open(img_path).convert('RGB')
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(f"True: {true_code} ({true_char})\nPred: {pred_code} ({pred_char}) [{conf}]",
                  fontsize=11, color=color, fontweight='bold')
                  
    plt.tight_layout()
    plt.savefig(output_img, dpi=200)
    plt.close()
    print(f"Saved prediction grid to: {output_img}")

def predict_single_image(image_input, model, transform, classes, char_mapping, device, topk=3, use_tta=False):
    """
    Predicts Thai character for a single PIL image or image file path.
    Supports optional Test-Time Augmentation (TTA).
    """
    if isinstance(image_input, str):
        image = Image.open(image_input).convert('RGB')
    else:
        image = image_input.convert('RGB')
        
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if use_tta:
            # TTA: original, slight angles, slight scales
            import torchvision.transforms.functional as TF
            angles = [-5, 0, 5]
            all_probs = []
            for ang in angles:
                rot_img = TF.rotate(tensor, ang, fill=1.0)
                out = model(rot_img)
                all_probs.append(torch.softmax(out, dim=1))
            probs = torch.stack(all_probs).mean(dim=0).squeeze(0)
        else:
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1).squeeze(0)
        
    top_probs, top_indices = torch.topk(probs, min(topk, len(classes)))
    
    results = []
    for rank in range(min(topk, len(classes))):
        idx = top_indices[rank].item()
        code = classes[idx]
        prob = top_probs[rank].item()
        info = char_mapping.get(str(code), {})
        thai_char = info.get("char", "N/A")
        desc = info.get("description", "")
        results.append({
            "rank": rank + 1,
            "class_code": code,
            "thai_char": thai_char,
            "description": desc,
            "probability": prob,
            "percentage": f"{prob * 100:.2f}%"
        })
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Inference for Thai Character Classification (72 Classes)")
    parser.add_argument("--checkpoint", type=str, default="best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--image", type=str, default=None, help="Path to single image to classify")
    parser.add_argument("--dir", type=str, default=None, help="Path to directory of images to classify")
    parser.add_argument("--output", type=str, default="inference_results.csv", help="Output CSV path for batch inference")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation for higher test accuracy")
    parser.add_argument("--demo", action="store_true", help="Run visual evaluation on random validation samples")
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint {args.checkpoint} not found. Please ensure best_model.pt exists.")
        sys.exit(1)
        
    model, transform, classes, char_mapping, chkpt = load_checkpoint(args.checkpoint, device)
    print(f"Loaded {chkpt.get('model_name', 'model')} trained with Best Val Acc: {chkpt.get('val_top1_acc', 0.0):.2f}% ({len(classes)} classes)")
    if args.tta:
        print("Test-Time Augmentation (TTA) enabled.")
    
    if args.image:
        results = predict_single_image(args.image, model, transform, classes, char_mapping, device, topk=3, use_tta=args.tta)
        print(f"\nPrediction for: {args.image}")
        print("=" * 60)
        for i, res in enumerate(results):
            marker = ">>>" if i == 0 else "   "
            print(f"{marker} Top-{i+1}: Class {res['class_code']:>3} | Character: {res['thai_char']} ({res['description']}) | Confidence: {res['percentage']}")
            
    elif args.dir:
        import pandas as pd
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        files = []
        for root, _, filenames in os.walk(args.dir):
            for fn in filenames:
                if fn.lower().endswith(valid_exts):
                    files.append(os.path.join(root, fn))
                    
        if not files:
            print(f"No image files found in {args.dir}")
            return
            
        print(f"\nFound {len(files)} images in {args.dir}. Running inference...")
        results_list = []
        top1_correct = 0
        top3_correct = 0
        total_eval = 0
        has_ground_truth = False
        
        for idx, fp in enumerate(files):
            top3 = predict_single_image(fp, model, transform, classes, char_mapping, device, topk=3, use_tta=args.tta)
            res = top3[0]
            
            # Check if ground truth is inferrable from parent folder name or filename
            parent_dir = os.path.basename(os.path.dirname(fp))
            true_code = None
            if parent_dir in classes:
                true_code = parent_dir
            elif any(fp_part in classes for fp_part in os.path.basename(fp).split('_')):
                for p in os.path.basename(fp).split('_'):
                    if p in classes:
                        true_code = p
                        break
                        
            row_data = {
                "filename": os.path.basename(fp),
                "filepath": fp,
                "predicted_code": res["class_code"],
                "predicted_char": res["thai_char"],
                "char_description": res["description"],
                "confidence": res["probability"],
                "confidence_pct": res["percentage"],
                "top2_code": top3[1]["class_code"] if len(top3) > 1 else "",
                "top2_char": top3[1]["thai_char"] if len(top3) > 1 else "",
                "top3_code": top3[2]["class_code"] if len(top3) > 2 else "",
                "top3_char": top3[2]["thai_char"] if len(top3) > 2 else ""
            }
            
            if true_code is not None:
                has_ground_truth = True
                row_data["ground_truth_code"] = true_code
                is_correct = (res["class_code"] == true_code)
                is_top3 = any(t["class_code"] == true_code for t in top3)
                row_data["correct"] = is_correct
                if is_correct:
                    top1_correct += 1
                if is_top3:
                    top3_correct += 1
                total_eval += 1
                
            results_list.append(row_data)
            
            if (idx + 1) % 50 == 0 or (idx + 1) == len(files):
                print(f"Processed [{idx + 1}/{len(files)}] images...")
                
        out_csv = args.output
        pd.DataFrame(results_list).to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"\n[DONE] Saved complete inference results to: {out_csv}")
        
        if has_ground_truth and total_eval > 0:
            top1_acc = (top1_correct / total_eval) * 100.0
            top3_acc = (top3_correct / total_eval) * 100.0
            print("=" * 60)
            print(f"Ground Truth Evaluation on {total_eval} labeled images:")
            print(f"  --> Top-1 Accuracy: {top1_acc:.2f}% ({top1_correct}/{total_eval})")
            print(f"  --> Top-3 Accuracy: {top3_acc:.2f}% ({top3_correct}/{total_eval})")
            print("=" * 60)
            
    elif args.demo:
        visualize_batch_predictions(model, transform, classes, char_mapping, device)
    else:
        # Default: test with sample image from dataset
        import glob
        samples = glob.glob("ThaiCharacter Dataset/round2/*/*.jpg")
        if samples:
            sample_img = samples[0]
            print(f"No --image provided. Running test inference on sample: {sample_img}")
            results = predict_single_image(sample_img, model, transform, classes, char_mapping, device, topk=3, use_tta=args.tta)
            print("=" * 60)
            for i, res in enumerate(results):
                marker = ">>>" if i == 0 else "   "
                print(f"{marker} Top-{i+1}: Class {res['class_code']:>3} | Character: {res['thai_char']} ({res['description']}) | Confidence: {res['percentage']}")
        visualize_batch_predictions(model, transform, classes, char_mapping, device)

if __name__ == "__main__":
    main()
