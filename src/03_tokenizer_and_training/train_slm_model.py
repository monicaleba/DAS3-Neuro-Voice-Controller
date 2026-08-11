import json
import os
import re
import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# Import the schema lists we made earlier
from command_schema import ALL_OUTPUTS, TOKEN_OF 

# ────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION & MAPPINGS
# ────────────────────────────────────────────────────────────────────────
TOKENIZER_DIR = './custom_das3_tokenizer'
OUTPUT_DIR    = './custom_das3_Emilutz_SLM_AI_Model'
MAX_INPUT     = 64    

label2id = {label: i for i, label in enumerate(ALL_OUTPUTS)}
id2label = {i: label for i, label in enumerate(ALL_OUTPUTS)}

# ────────────────────────────────────────────────────────────────────────
# 1.5. DATA ENHANCEMENT (NOISE CANCELLATION & PRIORITY BOOSTING)
# ────────────────────────────────────────────────────────────────────────
# 1. THE NOISE CANCELLER: Words the AI should completely ignore (Commented out for now)
STOP_WORDS = ["the", "a", "an", "to", "into", "my", 'in']

# 2. THE SIGNAL BOOSTER: Words that should guarantee a specific command
PRIORITY_WORDS = {
    'SHOULDER_UP': ['shoulder', 'up', 'upper Arm', 'upper Limb', 'arm', 'upward'],
    'SHOULDER_DOWN': ['shoulder', 'down', 'upper Arm', 'upper Limb', 'arm', 'downward'],
    'SHOULDER_LEFT': ['shoulder', 'left', 'upper Arm', 'upper Limb', 'arm', 'leftward'],
    'SHOULDER_RIGHT': ['shoulder', 'right', 'upper Arm', 'upper Limb', 'arm', 'rightward'],
    
    'ELBOW_UP': ['elbow', 'up', 'forearm', 'lower arm', 'lower limb', 'in', 'upward'],
    'ELBOW_DOWN': ['elbow', 'down', 'forearm', 'lower arm', 'lower limb', 'out', 'downward'],
    
    'ROTATE_WRIST': ['spin', 'twist', 'roll', 'swivel', 'gyrate'],
    
    'STOP': ['halt', 'freeze', 'abort', 'cancel', 'cease', 'stop', 'do not', "don't", 'End', 'Break', 'quit', 'pause'],
    
    'REST': ['zero', 'home', 'sleep', 'baseline', 'dormant', 'rest', 'relax', 'neutral', 'default', 'calm', 'steady', 'nap'],
}

def load_json(path, is_train=False):
    with open(path, encoding='utf-8') as f:
        raw_data = json.load(f)
        
    if not is_train:
        # If it's validation data, just load it normally. We don't want to boost tests.
        return Dataset.from_list(raw_data)
        
    print(f"\nApplying Data Enhancements to {path}...")
    enhanced_data = []
    
    for item in raw_data:
        text = item['input']
        target_command = item['target']
        target_token = TOKEN_OF.get(target_command, "")

        # --- STEP A: NOISE CANCELLER (Currently Commented Out) ---
        # If you want to use this, uncomment the next 4 lines.
        #words = text.split()
        #clean_words = [w for w in words if w.lower() not in STOP_WORDS]
        #text = " ".join(clean_words)
        #item['input'] = text
        
        enhanced_data.append(item)

        # --- STEP B: COMMAND-WEIGHTED PRIORITY BOOSTER ---
        boost_copies = 0
        if target_token in PRIORITY_WORDS:
            
            # 1. Establish the Strict Hierarchy Multiplier
            if target_token == 'STOP':
                weight = 15  # Absolute highest priority
            elif target_token == 'REST':
                weight = 8   # Second highest priority
            else:
                weight = 2   # Standard priority for the remaining 7 commands
            
            # 2. Search for the words (Case-Insensitive)
            for golden_word in PRIORITY_WORDS[target_token]:
                # \b enforces exact word boundaries.
                pattern = r'\b' + re.escape(golden_word) + r'\b'
                
                # re.IGNORECASE makes sure "End", "END", and "end" are all treated the same
                if re.search(pattern, text, flags=re.IGNORECASE):
                    # Add copies based on the command's rank
                    boost_copies += weight 
                    
        # If any golden words were found, clone the item that many times
        if boost_copies > 0:
            enhanced_data.extend([item.copy() for _ in range(boost_copies)])
                    
    print(f"Dataset expanded from {len(raw_data)} to {len(enhanced_data)} items via Priority Boosting.")
    return Dataset.from_list(enhanced_data)

# A helper function to calculate accuracy during training
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

def train_slm():
    print(f"\n--- 1. LOADING CUSTOM TOKENIZER ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    except OSError:
        print(f"Error: Tokenizer not found at {TOKENIZER_DIR}. Run build_tokenizer.py first!")
        return

    print("\n--- 2. INITIALIZING A BLANK MODEL ARCHITECTURE ---")
    config = AutoConfig.from_pretrained(
        "distilbert-base-uncased",
        vocab_size=len(tokenizer),
        num_labels=len(ALL_OUTPUTS),
        label2id=label2id,
        id2label=id2label
    )
    
    model = AutoModelForSequenceClassification.from_config(config)
    print(f"Model initialized with {model.num_parameters():,} parameters.")

    print("\n--- 3. LOADING & PROCESSING DATASET ---")
    train_ds = load_json('train.json', is_train=True)  # <-- Enhancements applied here
    val_ds   = load_json('val.json', is_train=False)   # <-- Validation untouched

    def tokenise(batch):
        model_inputs = tokenizer(batch['input'], max_length=MAX_INPUT, truncation=True)
        model_inputs['labels'] = [label2id[target_text] for target_text in batch['target']]
        return model_inputs

    train_tok = train_ds.map(tokenise, batched=True, remove_columns=['input', 'target'])
    val_tok   = val_ds.map(tokenise, batched=True, remove_columns=['input', 'target'])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print("\n--- 4. CONFIGURING TRAINING HYPERPARAMETERS ---")
    training_args = TrainingArguments(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = 30,          
        per_device_train_batch_size = 8,           
        per_device_eval_batch_size  = 16,          
        learning_rate               = 5e-5,        
        warmup_steps                = 100,         
        weight_decay                = 0.05,        
        evaluation_strategy         = 'epoch',     
        save_strategy               = 'epoch',
        load_best_model_at_end      = True,        
        metric_for_best_model       = 'accuracy',
        fp16                        = torch.cuda.is_available(), 
        logging_steps               = 10,
        report_to                   = 'none',
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_tok,
        eval_dataset    = val_tok,
        tokenizer       = tokenizer,
        data_collator   = data_collator,
        compute_metrics = compute_metrics, 
    )

    print("\n--- 5. FINE-TUNING THE AI (Training from Scratch) ---")
    trainer.train()

    print(f"\n--- 6. SAVING CUSTOM SLM TO {OUTPUT_DIR} ---")
    trainer.save_model(OUTPUT_DIR)    
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training Complete! You have successfully built your own AI brain.")

if __name__ == "__main__":
    train_slm()