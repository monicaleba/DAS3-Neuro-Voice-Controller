import json
import random
import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split

# ────────────────────────────────────────────────────────────────────────
# 1. GENERATE THE SCHEMA FILE
# ────────────────────────────────────────────────────────────────────────
schema_code = """# Maps command token -> canonical structured output string
COMMANDS = {
    'STOP'             : 'EMERGENCY(action=stop)',
    'REST'             : 'STATE(target=zero_pose)',
    'SHOULDER_UP'      : 'INCREMENT(axis=GH_z, direction=positive, amount=10)',
    'SHOULDER_DOWN'    : 'INCREMENT(axis=GH_z, direction=negative, amount=10)',
    'SHOULDER_LEFT'    : 'INCREMENT(axis=GH_y, direction=positive, amount=10)',
    'SHOULDER_RIGHT'   : 'INCREMENT(axis=GH_y, direction=negative, amount=10)',
    'ELBOW_UP'         : 'INCREMENT(axis=EL_x, direction=flex, amount=10)',
    'ELBOW_DOWN'       : 'INCREMENT(axis=EL_x, direction=extend, amount=10)',
    'ROTATE_WRIST'     : 'INCREMENT(axis=PS_y, direction=turn, amount=90)',
}
TOKEN_OF = {v: k for k, v in COMMANDS.items()}
ALL_OUTPUTS = list(COMMANDS.values())
"""
with open('command_schema.py', 'w', encoding='utf-8') as f:
    f.write(schema_code)

# ────────────────────────────────────────────────────────────────────────
# 2. FILE PATH MAPPING
# ────────────────────────────────────────────────────────────────────────
file_paths = {
    'STOP':              'data_files/stop.txt',
    'REST':              'data_files/rest.txt',
    'SHOULDER_UP':       'data_files/move_up.txt',
    'SHOULDER_DOWN':     'data_files/move_down.txt',
    'SHOULDER_LEFT':     'data_files/move_left.txt',
    'SHOULDER_RIGHT':    'data_files/move_right.txt',
    'ELBOW_UP':          'data_files/pull.txt',
    'ELBOW_DOWN':        'data_files/push.txt',
    'ROTATE_WRIST':      'data_files/turn.txt',
}

# ────────────────────────────────────────────────────────────────────────
# 3. READ FILES AND BUILD THE RAW DATASET 
# ────────────────────────────────────────────────────────────────────────
from command_schema import COMMANDS

raw_data = []
print("Loading data from text files...\n")

for token, filepath in file_paths.items():
    if not os.path.exists(filepath):
        print(f"⚠️  WARNING: File not found -> {filepath} (Skipping {token})")
        continue

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, skipinitialspace=True)
        for row in reader:
            for expr in row:
                clean_expr = expr.strip(' \n\r\t"\'')
                if clean_expr:
                    # We now track the 'source' file so we can report on it later
                    raw_data.append({
                        'input': clean_expr, 
                        'target': COMMANDS[token], 
                        'source': filepath
                    })

# ────────────────────────────────────────────────────────────────────────
# 4. DUPLICATE ANALYSIS & DEEP MIND METHODOLOGY
# ────────────────────────────────────────────────────────────────────────
print("-" * 50)
print("Running Duplicate Analysis...")

# Convert to Pandas DataFrame for advanced manipulation
df = pd.DataFrame(raw_data)
initial_count = len(df)

# Find all duplicates (keep=False ensures we capture every instance of a duplicate)
all_duplicates = df[df.duplicated(subset=['input'], keep=False)]

if all_duplicates.empty:
    print("✅ No duplicate expressions found across any files! Dataset is clean.")
else:
    print("⚠️  WARNING: Duplicate Expressions Found!")
    print("Review the following expressions (they will be automatically removed):")
    
    # Group by the expression to list all files it appeared in
    grouped_dupes = all_duplicates.groupby('input')
    for expr, group in grouped_dupes:
        files = group['source'].tolist()
        # Create a clean string of files (e.g., "data_files/pull.txt, data_files/push.txt")
        file_list_str = ", ".join(files)
        print(f"   - '{expr}' ➡️ Found in: [{file_list_str}]")

print("\nApplying DeepMind Lab Data Processing Standards...")

# Step A: Remove exact duplicates (prevents data leakage)
df = df.drop_duplicates(subset=['input'])
removed_count = initial_count - len(df)
if removed_count > 0:
    print(f"🧹 Removed {removed_count} duplicate expressions from the training pool.")

# Step B: Remove the 'source' column so it doesn't get saved into the final JSON files
df = df.drop(columns=['source'])

# Step C: Strict Class Balancing
min_class_count = df['target'].value_counts().min()
df_balanced = df.groupby('target').sample(n=min_class_count, random_state=42)
print(f"⚖️  Balanced all classes to exactly {min_class_count} samples each.")

# Step D: Stratified Split (80% Train / 20% Validation)
train_df, val_df = train_test_split(
    df_balanced, 
    test_size=0.2, 
    stratify=df_balanced['target'], 
    random_state=42
)

# Convert back to list of dicts
train_data = train_df.to_dict(orient='records')
val_data = val_df.to_dict(orient='records')

# ────────────────────────────────────────────────────────────────────────
# 5. SAVE EXPORTS
# ────────────────────────────────────────────────────────────────────────
with open('train.json', 'w', encoding='utf-8') as f: 
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open('val.json', 'w', encoding='utf-8') as f: 
    json.dump(val_data, f, ensure_ascii=False, indent=2)

print("\n" + "="*50)
print("✅ DATASET PREPARATION COMPLETE!")
print(f"   - Generated: command_schema.py")
print(f"   - Saved: train.json ({len(train_data)} total samples)")
print(f"   - Saved: val.json ({len(val_data)} total samples)")
print("="*50)
print("You can now run your training script!")