import json
import time
import torch
import torch.nn.functional as F  # <-- NEW: Imported for Softmax confidence calculations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix

# ── Import the Schema to map structured outputs back to readable tokens ──
from command_schema import TOKEN_OF

# Update these directories to match where your custom model and new validation data were saved
MODEL_DIR = './custom_das3_Emilutz_SLM_AI_Model'
VAL_FILE  = 'valnewvalue.json'

def evaluate_model():
    print(f"Loading custom classifier from {MODEL_DIR}...")
    try:
        # Load our custom tokenizer and the trained classification model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}\nDid you run train_model.py first?")
        return

    print(f"Loading validation data from {VAL_FILE}...")
    try:
        with open(VAL_FILE, encoding='utf-8') as f:
            val_data = json.load(f)
    except FileNotFoundError:
        print(f"File {VAL_FILE} not found!")
        return
    
    # For the detailed report
    y_true_labels = []
    y_pred_labels = []
    
    exact_match_count = 0
    total_time = 0.0

    print(f"\nStarting evaluation on {len(val_data)} validation samples...\n")

    for item in tqdm(val_data, desc="Evaluating Model", unit="sample"):
        input_text = item['input']
        target_text = item['target'].strip()

        # Tokenize input using our custom micro-dictionary
        inputs = tokenizer(input_text, return_tensors='pt', max_length=64, truncation=True)
        
        # Measure Latency (how fast the AI categorizes the text)
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
            
            # 1. Convert the raw output scores (logits) into percentages (0.0 to 1.0)
            probs = F.softmax(outputs.logits, dim=-1)
            
            # 2. Grab the highest percentage and its corresponding ID
            max_prob, pred_id = torch.max(probs, dim=-1)
            
            confidence_score = max_prob.item()
            pred_id_value = pred_id.item()
            
        t1 = time.perf_counter()
        total_time += (t1 - t0)
        
        # 3. THE UNKNOWN TRAP: If the AI is less than 75% sure, force it to UNKNOWN
        CONFIDENCE_THRESHOLD = 0.75
        
        if confidence_score < CONFIDENCE_THRESHOLD:
            pred_text = "UNKNOWN_CONFIDENCE_TOO_LOW"
            pred_token = "UNKNOWN"
        else:
            # Convert the predicted ID back to the schema string using the model's internal config
            pred_text = model.config.id2label[pred_id_value]
            pred_token = TOKEN_OF.get(pred_text, "UNKNOWN")
            
        # Map the complex schema string back to a readable label for the true target
        true_token = TOKEN_OF.get(target_text, "UNKNOWN")
        
        y_true_labels.append(true_token)
        y_pred_labels.append(pred_token)

        if pred_text == target_text:
            exact_match_count += 1
        else:
            # Show mistakes to understand AI confusion
            tqdm.write(f"[MISTAKE] Input: '{input_text}'")
            tqdm.write(f"   AI guessed: {pred_token} (Confidence: {confidence_score:.2f})")
            tqdm.write(f"   Expected:   {true_token}\n")

    # ─── CALCULATE METRICS ───
    n = len(val_data)
    exact_match_pct = (exact_match_count / n) * 100
    avg_latency_ms = (total_time / n) * 1000

    print("\n" + "=" * 60)
    print("📊 CUSTOM SLM EVALUATION RESULTS (CLASSIFIER)")
    print("=" * 60)
    
    # General Metrics
    print(f"1. Exact-Match Accuracy : {exact_match_pct:.2f}% ({exact_match_count}/{n})")
    print(f"2. Avg Latency per text : {avg_latency_ms:.2f} ms") 
    print("-" * 60)
    
    # Detailed Classification Report
    print("3. Detailed Classification Report:")
    # We use zero_division=0 so we don't get warnings if a class was completely missed
    report_text = classification_report(y_true_labels, y_pred_labels, zero_division=0)
    print(report_text)
    print("=" * 60)

    # ─── GENERATE GRAPHICS ───
    print("\nGenerating visual graphics... (Close the windows to exit the script)")
    
    report_dict = classification_report(y_true_labels, y_pred_labels, zero_division=0, output_dict=True)
    labels = [label for label in report_dict.keys() if label not in ['accuracy', 'macro avg', 'weighted avg']]
    
    f1_scores = [report_dict[label]['f1-score'] for label in labels]
    
    # Figure 1: F1-Scores
    plt.figure(figsize=(12, 6))
    sns.barplot(x=f1_scores, y=labels, hue=labels, palette="viridis", legend=False)
    plt.title("F1-Score per Command (Classifier Model)", fontsize=14, fontweight='bold')
    plt.xlabel("F1-Score (0.0 to 1.0)", fontsize=12)
    plt.ylabel("Commands", fontsize=12)
    plt.xlim(0, 1.05)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Figure 2: Confusion Matrix
    unique_labels = sorted(list(set(y_true_labels + y_pred_labels)))
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=unique_labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title("Confusion Matrix: Actual vs Predicted Commands", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Command", fontsize=12)
    plt.ylabel("True Command", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_model()